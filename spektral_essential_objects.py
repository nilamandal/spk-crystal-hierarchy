from spektral.data import Graph, Dataset, DisjointLoader
from spektral.data.utils import to_batch
from spektral.utils import reorder, sp_matrix_to_sp_tensor
from spektral.layers import CrystalConv, DiffPool, ops, GlobalSumPool, GlobalAvgPool, Disjoint2Batch
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Multiply, Masking, LayerNormalization
from tensorflow.keras.losses import MeanSquaredError, SparseCategoricalCrossentropy
from tensorflow.keras.metrics import sparse_categorical_accuracy
from tensorflow.keras.regularizers import L2
import numpy as np
import pandas as pd
import os
import sys
from pymatgen.core.structure import Structure
import json
import argparse
import time
import scipy.sparse as sp
from tensorflow.keras import backend as K
from tensorflow.keras import activations
from keras import initializers
from spektral.layers.ops.scatter import deserialize_scatter
from tensorflow.keras.callbacks import Callback
from tensorflow.python.ops.linalg.sparse import sparse_csr_matrix_ops

class GaussianDistance(object):
    """
    Expands the distance by Gaussian basis.
    Unit: angstrom
    """
    def __init__(self, dmin, dmax, step, var=None):
        """
        Parameters
        ----------
        dmin: float
          Minimum interatomic distance
        dmax: float
          Maximum interatomic distance
        step: float
          Step size for the Gaussian filter
        """
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax+step, step)
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):
        """Apply Gaussian disntance filter to a numpy distance array"""
        return np.exp(-(distances[..., np.newaxis] - self.filter)**2 /
                      self.var**2)



class MyDataset(Dataset):
    def __init__(self, df, datadir, r_a, num_nbrs, task):
        self.dataframe=df
        self.datadir= datadir
        self.radius_angstroms= r_a
        self.num_nbrs= num_nbrs
        self.task= task
        self.electronegativity_lookup= {1:2.2, 3:0.98, 4:1.57, 11:0.93, 12:1.31, 13:1.61, 14:1.9, 15:2.19, 19:0.82,
            20:1, 25:1.55, 30:1.65, 31:1.81, 32:2.01, 33:2.18, 37:0.82, 38:0.95, 48:1.69, 49:1.69, 50:1.96, 51:2.05,
            55:0.79, 56:0.89, 70:1.1, 80:2, 81:1.62, 82:2.33, 83:2.02}

        super().__init__()

    def read(self):
        df = self.dataframe.sample(frac=1).reset_index(drop=True)
        allgraphs=[]
        cifs=list(df['id'])
        #num_symmetric=0
        #num_asymmetric=0
        self.cifs=cifs
        all_atomic_numbers=[]
        for c in cifs:
            c=str(c)
            try:
                crystal= Structure.from_file(os.path.join(self.datadir,c))
            except:
                crystal= Structure.from_file(os.path.join(self.datadir,c+'.cif'))
            num_atoms=len(crystal)

            atomic_numbers=[crystal[i].specie.number for i in range(len(crystal))]

            atom_fea=[]
            all_atomic_numbers= all_atomic_numbers + atomic_numbers
            for atom in crystal:
                group_encoding= np.zeros(18)
                row_encoding= np.zeros(9)
                group_encoding[atom.specie.group-1]=1
                row_encoding[atom.specie.row-1]=1
                electronegativity= [self.electronegativity_lookup[atom.specie.number]]
                atom_hot=np.concatenate((group_encoding, row_encoding, electronegativity))
                atom_fea.append(atom_hot)

            atom_fea= np.vstack(atom_fea)

            all_nbrs = crystal.get_all_neighbors(self.radius_angstroms, include_index=True)
            all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
            nbr_fea_idx, nbr_fea = [], []
            for nbr in all_nbrs:
                if len(nbr) < self.num_nbrs:
                    warnings.warn('{} not find enough neighbors to build graph. '
                                  'If it happens frequently, consider increase '
                                  'radius.'.format(cif_id))
                    nbr_fea_idx.append(list(map(lambda x: x[2], nbr)) +
                                       [0] * (self.num_nbrs - len(nbr)))
                    nbr_fea.append(list(map(lambda x: x[1], nbr)) +
                                   [self.radius_angstroms + 1.] * (self.num_nbrs -
                                                         len(nbr)))
                else:
                    nbr_fea_idx.append(list(map(lambda x: x[2],
                                                nbr[:self.num_nbrs])))
                    nbr_fea.append(list(map(lambda x: x[1],
                                            nbr[:self.num_nbrs])))
            df_MG=df[df['id'].astype(str)==c]
            gdf = GaussianDistance(dmin=0, dmax=8, step=0.2)
            nbr_fea = gdf.expand(np.array(nbr_fea))
            adj = np.zeros((num_atoms, num_atoms))
            edges= np.zeros((num_atoms, num_atoms, 41))
            edgeidxtemp=[]
            edgefeat=[]
            for i in range(len(nbr_fea_idx)):
                for j in range(len(nbr_fea_idx[i])):
                    k=nbr_fea_idx[i][j]
                    adj[i,k]+=1
                    if adj[i,k]==1:
                        edgeidxtemp.append((i,k))
                        edgefeat.append(nbr_fea[i][j])

            adj=sp.csr_matrix(adj)

            edge_idx, edges= reorder(edge_index=np.array(edgeidxtemp), edge_features=np.array(edgefeat))

            if self.task=='c':
                MG=Graph(x=atom_fea, a=adj, e=edges, y=int(df_MG['target'].values[0]))
                MG._atomlist=set(atomic_numbers)
                MG._cif=c
            elif self.task=='r':
                MG=Graph(x=atom_fea, a=adj, e=edges, y=float(df_MG['target'].values[0]))
                MG._atomlist=set(atomic_numbers)
                MG._cif=c
            else:
                print(self.task, ' is not c or r.')
            allgraphs.append(MG)
        self.all_atomic_numbers= set(all_atomic_numbers)

        return allgraphs

    def get_cifs(self):
        return np.asarray(self.cifs , dtype=object)


class RegularizedDiffPool(DiffPool):
    def __init__(self, k, beta=1, channels=None, return_selection=False, activation='relu', kernel_initializer="glorot_uniform",
        kernel_regularizer=None, kernel_constraint=None,  path='./', **kwargs):

        self.k=tf.constant(k)
        self.beta= tf.constant(beta, dtype=tf.float32)

        super().__init__(k, channels=channels, return_selection=return_selection, activation=activation,
                kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer, kernel_constraint=kernel_constraint,
                **kwargs)
        self.saveindex=1
        self.savepath= path
        self.assignment_fc= Dense(self.k, use_bias=False)
        self.masker= Masking(mask_value=np.zeros(self.k))

    def build(self, input_shape):
        in_channels = input_shape[0][-1]
        if self.channels is None:
            self.channels = in_channels
        super(DiffPool, self).build(input_shape)

    def call(self, inputs, mask=None):
        x, a, i = self.get_inputs(inputs)

        # Graph filter for GNNs
        if K.is_sparse(a):
            #i_n = tf.sparse.eye(self.n_nodes, dtype=a.dtype)
            i_sub=tf.one_hot(list(range(self.n_nodes)),depth=self.n_nodes)
            i_n= tf.sparse.from_dense(tf.stack([i_sub]*x.shape[0]))
            a_ = tf.sparse.add(a, i_n)
        else:
            i_n = tf.eye(self.n_nodes, dtype=a.dtype)
            a_ = a + i_n
        fltr = ops.normalize_A(a_)

        output = self.pool(x, a, i, fltr=fltr, mask=mask)
        return output

    def select(self, x, a, i, fltr=None, mask=None):
        x = ops.modal_dot(fltr, x)
        s = self.assignment_fc(x)

        masked_s = self.masker(s)

        masked_tensor= tf.ragged.boolean_mask(masked_s, masked_s._keras_mask)
        means= tf.reduce_mean(masked_tensor, axis=1)
        stdev= tf.math.add(tf.math.reduce_std(masked_tensor, axis=1), K.epsilon()) #+epsilon in case of std=0 (when column is all the same value)
        mean_stack=tf.stack([means]*s.shape[1], axis=1)
        stdev_stack= tf.stack([stdev]*s.shape[1], axis=1)

        ###### version with beta
        s_interim= tf.math.divide(tf.subtract(s, mean_stack), stdev_stack)
        s_interim= s_interim*self.beta

        normalized_crystal= tf.ragged.boolean_mask(s_interim, masked_s._keras_mask)
        normalized_crystal = activations.softmax(normalized_crystal, axis=-1)

        s= normalized_crystal.to_tensor(default_value=0.)

        if mask is not None:
            s *= mask[0]

        return s

    def reduce(self, x, s, fltr=None):
        x = ops.modal_dot(fltr, x)
        return ops.modal_dot(s, x, transpose_a=True)

    def connect(self, a, s, **kwargs):
        return ops.matmul_at_b_a(s, a)




class ModifiedCrystalConv(CrystalConv):
     def __init__(self, activation= None, kernel_initializer= None, **kwargs):
         super().__init__(self, activation=activation, kernel_initializer=kernel_initializer, **kwargs)
         self.agg = deserialize_scatter('sum')
         self.bn1= BatchNormalization()
         self.bn2= BatchNormalization()


     def message(self, x, e=None):
        x_i = self.get_targets(x)
        x_j = self.get_sources(x)

        to_concat = [x_i, x_j]
        if e is not None:
            to_concat += [e]
        z = K.concatenate(to_concat, axis=-1)
        z= self.bn1(z)
        self.nbr_sumed = self.dense_s(z) * self.dense_f(z)
        output= self.bn2(self.nbr_sumed)
        return output


class SuperCgcnn(CrystalConv):
    def __init__(self, activation= None, kernel_initializer= None, **kwargs):
        super().__init__(self, activation=activation, kernel_initializer=kernel_initializer, **kwargs)
        self.transfer_weights=kwargs['transfers']
        self.transfer_idx=kwargs['transfer_idx']


    def build(self, input_shape):
        assert len(input_shape) >= 2
        layer_kwargs = dict(
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint,
            dtype=self.dtype,
        )
        channels = input_shape[0][-1] * 2

        self.dense_fc = Dense(channels, **layer_kwargs)
        #self.dense_s = Dense(channels, activation=self.activation, **layer_kwargs)

        bn1_w= 'bn1_w_'+self.transfer_idx
        bn1_b= 'bn1_b_'+self.transfer_idx
        bn2_w= 'bn2_w_'+self.transfer_idx
        bn2_b= 'bn2_b_'+self.transfer_idx
        #initializers.constant(self.transfer_weights[bn1_w])
        self.agg = deserialize_scatter('sum')
        self.bn1= BatchNormalization(beta_initializer=initializers.constant(self.transfer_weights[bn1_w]) ,gamma_initializer=initializers.constant(self.transfer_weights[bn1_b]))
        self.bn2= BatchNormalization(beta_initializer=initializers.constant(self.transfer_weights[bn2_w]) ,gamma_initializer=initializers.constant(self.transfer_weights[bn2_b]))
        self.built = True

    def message(self, x, e=None):
       x_i = self.get_targets(x)
       x_j = self.get_sources(x)

       to_concat = [x_i, x_j]
       if e is not None:
           to_concat += [e]
       z = K.concatenate(to_concat, axis=-1)
       z= self.dense_fc(z)
       z= self.bn1(z)
       #print(z.shape)
       nbr_filter, nbr_core= tf.split(z, 2, axis=1)
       nbr_filter= tf.sigmoid(nbr_filter)
       nbr_core= tf.keras.activations.softplus(nbr_core)
       nbr_sumed=nbr_filter * nbr_core
       output= tf.keras.activations.softplus(tf.math.add(x_i, self.bn2(nbr_sumed)))
       return output
#


class HNetDoubleJanossy(Model):
    def __init__(self, task, num_classes, embedding_size=52, cgcnn_num=3, d1=0.578, el=427, cl=265, fc_num=1, fc_size=23, fc_num2=1, fc_size2=23, regularizer='l2', return_s=False,  random_seed=0, path='./', **kwargs):
        super().__init__()
        glorot_initializer= initializers.glorot_uniform(seed=random_seed)
        he_initializer= initializers.he_uniform(seed=random_seed)

        self.return_s=return_s
        self.task=task
        self.num_classes=num_classes

        self.embedding= Dense(embedding_size, kernel_initializer=glorot_initializer, kernel_regularizer=regularizer)

        self.conv_list=[]
        for i in range(cgcnn_num):
            conv= ModifiedCrystalConv(activation= 'softplus', kernel_initializer=he_initializer)#does this l2 have a lambda
            self.conv_list.append(conv)
        #conv= ModifiedCrystalConv(activation= 'tanh', kernel_initializer=glorot_initializer)#does this l2 have a lambda
        #self.conv_list.append(conv)

        self.disjoint2batch= Disjoint2Batch()
        #self.dropout1= Dropout(d1)

        self.pool= DoubleJanossyDiffPool(k=2, kernel_initializer=he_initializer, column_lambda=cl, entr_lambda=el, return_selection=True, path=path)
        #self.bn1= BatchNormalization()
        self.janossy_orange_list=[]
        for i in range(fc_num):
            fc= Dense(fc_size, activation='softplus', kernel_initializer=he_initializer)
            self.janossy_orange_list.append(fc)

        self.janossy_green_list=[]
        for i in range(fc_num):
            fc= Dense(fc_size, activation='softplus', kernel_initializer=he_initializer)
            self.janossy_green_list.append(fc)

        self.janossy_2_list=[]
        for i in range(fc_num2):
            fc= Dense(fc_size2, activation='softplus', kernel_initializer=he_initializer)
            self.janossy_2_list.append(fc)

        self.meanpool= GlobalAvgPool()
        #we should have a dropout after aggregating crystal features
        if self.task=='c':
            self.out_layer= Dense(self.num_classes, activation='softmax', kernel_initializer=glorot_initializer)
        elif self.task=='r':
            self.out_layer= Dense(1, kernel_initializer=glorot_initializer)
        self.saveindex=1
        self.savepath=path

    def call(self, inputs):
        x, a, e, i = inputs
        element_idx=np.empty((len(x)))
        for id in range(len(x)):
            temp=np.nonzero(x[id])[0]
            element_idx[id]=int(str(temp[0])+str(temp[1]))

        x= self.embedding(x)

        for cgcnn in self.conv_list:
            x= cgcnn([x, a, e])
            x= tf.nn.softplus(x)

        #filename= self.savepath+'/x_before_bn'+str(self.saveindex)
        #np.savez(filename, x=cgcnn.nbr_sumed)

        #x= self.dropout1(x)

        batch_X, batch_A= self.disjoint2batch([x, a, i])
        #filename= self.savepath+'/x_after_cgcnn_no_dropout'+str(self.saveindex)
        #np.savez(filename, x=batch_X)

        x_pool_all, a, i, s= self.pool([batch_X, batch_A, i, element_idx])
        #self.saveindex+=1
        x_pool_p0=x_pool_all[:,:,0]
        x_pool_p1=x_pool_all[:,:,1]

        x_2o= tf.stack([x_pool_p0[:,0],x_pool_p0[:,2],x_pool_p0[:,1]], axis=1)
        x_3o= tf.stack([x_pool_p0[:,1],x_pool_p0[:,0],x_pool_p0[:,2]], axis=1)
        x_4o= tf.stack([x_pool_p0[:,1],x_pool_p0[:,2],x_pool_p0[:,0]], axis=1)
        x_5o= tf.stack([x_pool_p0[:,2],x_pool_p0[:,0],x_pool_p0[:,1]], axis=1)
        x_6o= tf.stack([x_pool_p0[:,2],x_pool_p0[:,1],x_pool_p0[:,0]], axis=1)

        x_1o=tf.reshape(x_pool_p0, [x_pool_p0.shape[0],x_pool_p0.shape[1]*x_pool_p0.shape[2]])
        x_2o=tf.reshape(x_2o, [x_2o.shape[0],x_2o.shape[1]*x_2o.shape[2]])
        x_3o=tf.reshape(x_3o, [x_3o.shape[0],x_3o.shape[1]*x_3o.shape[2]])
        x_4o=tf.reshape(x_4o, [x_4o.shape[0],x_4o.shape[1]*x_4o.shape[2]])
        x_5o=tf.reshape(x_5o, [x_5o.shape[0],x_5o.shape[1]*x_5o.shape[2]])
        x_6o=tf.reshape(x_6o, [x_6o.shape[0],x_6o.shape[1]*x_6o.shape[2]])

        x_2g= tf.stack([x_pool_p1[:,0],x_pool_p1[:,2],x_pool_p1[:,1]], axis=1)
        x_3g= tf.stack([x_pool_p1[:,1],x_pool_p1[:,0],x_pool_p1[:,2]], axis=1)
        x_4g= tf.stack([x_pool_p1[:,1],x_pool_p1[:,2],x_pool_p1[:,0]], axis=1)
        x_5g= tf.stack([x_pool_p1[:,2],x_pool_p1[:,0],x_pool_p1[:,1]], axis=1)
        x_6g= tf.stack([x_pool_p1[:,2],x_pool_p1[:,1],x_pool_p1[:,0]], axis=1)


        x_1g=tf.reshape(x_pool_p1, [x_pool_p1.shape[0],x_pool_p1.shape[1]*x_pool_p1.shape[2]])
        x_2g=tf.reshape(x_2g, [x_2g.shape[0],x_2g.shape[1]*x_2g.shape[2]])
        x_3g=tf.reshape(x_3g, [x_3g.shape[0],x_3g.shape[1]*x_3g.shape[2]])
        x_4g=tf.reshape(x_4g, [x_4g.shape[0],x_4g.shape[1]*x_4g.shape[2]])
        x_5g=tf.reshape(x_5g, [x_5g.shape[0],x_5g.shape[1]*x_5g.shape[2]])
        x_6g=tf.reshape(x_6g, [x_6g.shape[0],x_6g.shape[1]*x_6g.shape[2]])


        for layer in self.janossy_orange_list:
            x_1o= layer(x_1o)
            x_2o= layer(x_2o)
            x_3o= layer(x_3o)
            x_4o= layer(x_4o)
            x_5o= layer(x_5o)
            x_6o= layer(x_6o)

        for layer in self.janossy_green_list:
            x_1g= layer(x_1g)
            x_2g= layer(x_2g)
            x_3g= layer(x_3g)
            x_4g= layer(x_4g)
            x_5g= layer(x_5g)
            x_6g= layer(x_6g)

        x_o=tf.stack([x_1o,x_2o,x_3o,x_4o,x_5o,x_6o],axis=-2)
        x_g=tf.stack([x_1g,x_2g,x_3g,x_4g,x_5g,x_6g],axis=-2)


        x_o= self.meanpool(x_o)
        x_g= self.meanpool(x_g)

        x_og= tf.concat([x_o, x_g], axis=1)
        x_go= tf.concat([x_g, x_o], axis=1)

        for layer in self.janossy_2_list:
            x_og = layer(x_og)
            x_go = layer(x_go)

        x_final= tf.stack([x_og, x_go], axis=-2)

        x_final=self.meanpool(x_final)

        x=self.out_layer(x_final)

        if self.return_s:
            return x, s
        else:
            return x

class SparseEdgepool(Model):
    def __init__(self, task, num_classes, embedding_size=52, cgcnn_num=3, cgcnn_num2=3, regularizer='l2', return_s=False, softmax_beta=1, random_seed=0, path='./', k= 2, **kwargs):
        super().__init__()
        glorot_initializer= initializers.glorot_uniform(seed=random_seed)
        he_initializer= initializers.he_uniform(seed=random_seed)

        self.return_s=return_s
        self.task=task
        self.num_classes=num_classes
        self.k= k

        self.embedding= Dense(embedding_size, kernel_initializer=glorot_initializer, kernel_regularizer=regularizer)

        self.conv_list=[]
        for i in range(cgcnn_num):
            conv= ModifiedCrystalConv(activation= 'softplus', kernel_initializer=he_initializer)#does this l2 have a lambda
            self.conv_list.append(conv)

        self.pool= RegularizedDiffPool(k=self.k, beta= softmax_beta, kernel_initializer=he_initializer, return_selection=True, path=path)

        self.conv_list2=[]
        for i in range(cgcnn_num2):
            conv= ModifiedCrystalConv(activation= 'softplus', kernel_initializer=he_initializer)
            self.conv_list2.append(conv)

        if self.task=='c':
            self.out_layer= Dense(self.num_classes, activation='softmax', kernel_initializer=glorot_initializer)
        elif self.task=='r':
            self.out_layer= Dense(1, kernel_initializer=glorot_initializer)
        self.saveindex=1
        self.savepath=path


    def call(self, inputs):
        x, a, e, i = inputs
        x= self.embedding(x)
        #print(e)
        for cgcnn in self.conv_list:
            x= cgcnn([x, a, e])
            x= tf.nn.softplus(x)

        batch_X = ops.disjoint_signal_to_batch(x, i)
        batch_A= self.disjoint_adjacency_to_batch(a, i)

        x_pool, a_pool, i_pool, s= self.pool([batch_X, batch_A, i])
        x_pool= tf.reshape(x_pool, [x_pool.shape[0]*x_pool.shape[1], x_pool.shape[2]]) #reshape to disjoint form

        e_pool= self.edgepool(e, batch_A, s, i)

        disjoint_a, edges= self.batch2disjoint(a_pool, e_pool, len(i_pool))

        for cgcnn2 in self.conv_list2:
            x_pool= cgcnn2([x_pool, disjoint_a, edges])
            x_pool= tf.nn.softplus(x_pool)

        x_pool= tf.reshape(x_pool, [int(x_pool.shape[0]/self.k), int(x_pool.shape[1]*self.k)])

        x=self.out_layer(x_pool)

        if self.return_s:
            return x, s
        else:
            return x

    def disjoint_adjacency_to_batch(self, A, I):#sparse version
        I = tf.cast(I, tf.int64)
        indices = A.indices
        values = A.values
        i_nodes, j_nodes = indices[:, 0], indices[:, 1]

        graph_sizes = tf.math.segment_sum(tf.ones_like(I), I)
        max_n_nodes = tf.reduce_max(graph_sizes)
        n_graphs = tf.shape(graph_sizes)[0]

        offset = tf.gather(I, i_nodes)
        offset = tf.gather(tf.cumsum(graph_sizes, exclusive=True), offset)

        relative_j_nodes = j_nodes - offset
        relative_i_nodes = i_nodes - offset
        real_new_indices= tf.stack([tf.gather(I, i_nodes),relative_i_nodes,relative_j_nodes], axis=1)

        batch = tf.sparse.SparseTensor(real_new_indices,values,(n_graphs, max_n_nodes, max_n_nodes))

        return batch


    def edgepool(self, e, batch_a, s, i):
        indices = batch_a.indices
        graph_sizes = tf.math.segment_sum(tf.ones_like(i), i)
        max_n_nodes = tf.cast(tf.reduce_max(graph_sizes), tf.int32)
        n_graphs = tf.cast(tf.shape(graph_sizes)[0], tf.int32)

        batch_edge_placeholder = tf.sparse.SparseTensor(indices,tf.reduce_sum(e, axis=1),(n_graphs, max_n_nodes, max_n_nodes))
        s_sparse= tf.sparse.from_dense(s)

        part_1= self.sparse_multiply(tf.sparse.transpose(s_sparse, perm=[0,2,1]), batch_edge_placeholder)
        batch_e= self.sparse_multiply(part_1,s_sparse)

        return batch_e

    def sparse_multiply(self, a: tf.SparseTensor, b: tf.SparseTensor):
        a_sm = sparse_csr_matrix_ops.sparse_tensor_to_csr_sparse_matrix(
            a.indices, a.values, a.dense_shape
        )

        b_sm = sparse_csr_matrix_ops.sparse_tensor_to_csr_sparse_matrix(
            b.indices, b.values, b.dense_shape
        )

        c_sm = sparse_csr_matrix_ops.sparse_matrix_sparse_mat_mul(
            a=a_sm, b=b_sm, type=tf.float32
        )

        c = sparse_csr_matrix_ops.csr_sparse_matrix_to_sparse_tensor(
            c_sm, tf.float32
        )

        return tf.SparseTensor(
            c.indices, c.values, dense_shape=c.dense_shape
        )

    def batch2disjoint(self, batch_adj, batch_edge, total_nodes):
        #adj

        temp_a= tf.unstack(batch_adj)
        #print(temp_a)
        disjoint_adj= np.zeros((total_nodes, total_nodes))
        begin=0
        step=len(temp_a[0])
        end=begin+step

        for j in temp_a:
            disjoint_adj[begin:end, begin:end]= j#np.ones((self.k,self.k))
            begin= begin+step
            end= begin+step
        #print(disjoint_adj)
        #print('----')
        disjoint_adj= tf.sparse.from_dense(disjoint_adj)

        #edge
        adj_indices=disjoint_adj.indices
        disjoint_e= np.zeros((total_nodes, total_nodes))

        #temp_e= tf.unstack(batch_edge)
        edge_idx= batch_edge.indices
        edge_vals= batch_edge.values

        edge_idx, edges= reorder(edge_index=np.array(adj_indices), edge_features=np.array(edge_vals))

        return disjoint_adj, np.reshape(edges, [edges.shape[0],1])
