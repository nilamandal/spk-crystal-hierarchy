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

        super().__init__()

    def read(self):
        df = self.dataframe.sample(frac=1).reset_index(drop=True)
        allgraphs=[]
        cifs=list(df['id'])
        num_symmetric=0
        num_asymmetric=0
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
                atom_hot=np.concatenate((group_encoding, row_encoding))
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
        kernel_regularizer=None, kernel_constraint=None, path='./', **kwargs):

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
        if np.any(tf.math.is_nan(x)):
            raise Exception('x input into call is nan')
        # Graph filter for GNNs
        if K.is_sparse(a):
            #i_n = tf.sparse.eye(self.n_nodes, dtype=a.dtype)
            i_sub=tf.one_hot(list(range(self.n_nodes)),depth=self.n_nodes)
            i_n= tf.sparse.from_dense(tf.stack([i_sub]*x.shape[0]))
            a_ = tf.sparse.add(a, i_n)
            #print(a_)
        else:
            i_n = tf.eye(self.n_nodes, dtype=a.dtype)
            a_ = a + i_n
        fltr = ops.normalize_A(a_)

        output = self.pool(x, a, i, fltr=fltr, mask=mask)
        return output

    def select(self, x, a, i, fltr=None, mask=None):
        x = ops.modal_dot(fltr, x)
        s = self.assignment_fc(x)
        #if np.any(tf.math.is_nan(s)):
        #  raise Exception('s self assignment is nan')
        masked_s = self.masker(s)

        masked_tensor= tf.ragged.boolean_mask(masked_s, masked_s._keras_mask)

        means= tf.reduce_mean(masked_tensor, axis=1)
        stdev= tf.math.add(tf.math.reduce_std(masked_tensor, axis=1), K.epsilon()) #+epsilon in case of std=0 (when column is all the same value)
        mean_stack=tf.stack([means]*s.shape[1], axis=1) #meanstack
        stdev_stack= tf.stack([stdev]*s.shape[1], axis=1)
        s_interim= tf.math.divide(tf.subtract(s, mean_stack), stdev_stack)
        if np.any(tf.math.is_nan(s_interim)):
          raise Exception('s_interim before beta is nan')
        s_interim= s_interim*self.beta
        if np.any(tf.math.is_nan(s_interim)):
          raise Exception('s_interim after beta is nan')

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


class HNetSingleJanossy(Model):
    def __init__(self, task, num_classes, beta=1, embedding_size=52, cgcnn_num=3, cgcnn_num2=3, regularizer='l2', return_s=False,  random_seed=0, path='./', **kwargs):
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

        self.disjoint2batch= Disjoint2Batch()

        self.pool= RegularizedDiffPool(k=2, beta=beta, kernel_initializer=he_initializer, return_selection=True, path=path)

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
        element_idx=np.empty((len(x)))
        for id in range(len(x)):
            temp=np.nonzero(x[id])[0]
            element_idx[id]=int(str(temp[0])+str(temp[1]))

        x= self.embedding(x)

        for cgcnn in self.conv_list:
            x= cgcnn([x, a, e])
            x= tf.nn.softplus(x)
#        if np.any(tf.math.is_nan(x)):
#           raise Exception('cgcnn series 1 is the problem')
        batch_X, batch_A= self.disjoint2batch([x, a, i])
#        if np.any(tf.math.is_nan(batch_X)):
#           raise Exception('batch_X is the problem')
        x_pool, a_pool, i, s= self.pool([batch_X, batch_A, i])
        if np.any(tf.math.is_nan(x_pool)):
           raise Exception('x_pool is the problem')
        x_pool= tf.reshape(x_pool, [x_pool.shape[0]*x_pool.shape[1], x_pool.shape[2]]) #reshape to disjoint form


        temp_a= tf.unstack(a_pool)
        total_nodes= len(i)
        disjoint_a= np.zeros((total_nodes, total_nodes))
        begin=0
        step=len(temp_a[0])
        end=begin+step

        for j in temp_a:
            disjoint_a[begin:end, begin:end]=j
            begin= begin+step
            end= begin+step
        disjoint_a= tf.sparse.from_dense(disjoint_a)

        for cgcnn2 in self.conv_list2:
            x_pool= cgcnn2([x_pool, disjoint_a])
            x_pool= tf.nn.softplus(x_pool)

        x_pool= tf.reshape(x_pool, [int(x_pool.shape[0]/2), int(x_pool.shape[1]*2)])

        x=self.out_layer(x_pool)

        if self.return_s:
            return x, s
        else:
            return x

class Edgepool(Model):
    def __init__(self, task, num_classes, embedding_size=52, cgcnn_num=3, cgcnn_num2=3, el=427, cl=265, softmax_beta= 1, regularizer='l2', return_s=False,  random_seed=0, path='./', k= 2, **kwargs):
        super().__init__()
        glorot_initializer= initializers.glorot_uniform(seed=random_seed)
        he_initializer= initializers.he_uniform(seed=random_seed)

        self.return_s=return_s
        self.task=task
        self.num_classes=num_classes
        self.k= k
        self.beta= softmax_beta

        self.embedding= Dense(embedding_size, kernel_initializer=glorot_initializer, kernel_regularizer=regularizer)

        self.conv_list=[]
        for i in range(cgcnn_num):
            conv= ModifiedCrystalConv(activation= 'softplus', kernel_initializer=he_initializer)#does this l2 have a lambda
            self.conv_list.append(conv)

        self.disjoint2batch= Disjoint2Batch()
        self.pool= RegularizedDiffPool(k=self.k, beta= self.beta, kernel_initializer=he_initializer, column_lambda=cl, entr_lambda=el, return_selection=True, path=path)

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

        for cgcnn in self.conv_list:
            x= cgcnn([x, a, e])
            x= tf.nn.softplus(x)

        batch_X, batch_A= self.disjoint2batch([x, a, i])

        x_pool, a_pool, i_pool, s= self.pool([batch_X, batch_A, i])
        #x_pool= tf.reshape(x_pool, [x_pool.shape[0]*x_pool.shape[1], x_pool.shape[2]]) #reshape to disjoint form

        #e_pool= self.edgepool(e, batch_A, s, i)
        e_pool= self.edgepool(e, a, s, i, a_pool)
        x_pool, a_pool, e_pool, i_pool = self.batch2disjoint(x_pool, e_pool, a_pool)
        # temp_a= tf.unstack(a_pool)

        for cgcnn2 in self.conv_list2:
             x_pool= cgcnn2([x_pool, a_pool, e_pool])
             x_pool= tf.nn.softplus(x_pool)
        #
        x_pool= tf.reshape(x_pool, [int(x_pool.shape[0]/2), int(x_pool.shape[1]*2)])
        #
        x=self.out_layer(x_pool)
        #
        if self.return_s:
             return x, s
        else:
             return x

    def edgepool(self, e, a, s, i, a_pooled):
        indices = a.indices
        #values = a.values
        i_nodes, j_nodes = indices[:, 0], indices[:, 1]

        graph_sizes = tf.math.segment_sum(tf.ones_like(i), i)
        max_n_nodes = tf.reduce_max(graph_sizes)
        n_graphs = tf.shape(graph_sizes)[0]
        relative_j_nodes = j_nodes - self._vectorised_get_cum_graph_size(j_nodes, graph_sizes)

        new_indices = tf.transpose(tf.stack([i_nodes, relative_j_nodes]))

        new_indices = tf.cast(new_indices, tf.int32)
        n_graphs = tf.cast(n_graphs, tf.int32)
        max_n_nodes = tf.cast(max_n_nodes, tf.int32)

        dense_edge = tf.scatter_nd(
            new_indices, e, (n_graphs * max_n_nodes, max_n_nodes, 41)
        )

        batch_edge = tf.reshape(dense_edge, (n_graphs, max_n_nodes, max_n_nodes, 41))
        batch_edge = tf.cast(batch_edge, tf.float32)

        temp=tf.einsum('bijk,bil->bilk',batch_edge,s)
        e_pooled=tf.einsum('bmn,bilk->bnlk',s,temp)

        a_extended= tf.stack([a_pooled] * 41, axis=3)
        e_pooled= tf.divide(e_pooled, a_extended)

        return e_pooled

    def batch2disjoint(self, batch_x, batch_e, batch_a):
        x_shape=batch_x.shape
        i= []
        for j in range(x_shape[0]):
            for k in range(x_shape[1]):
                i.append(j)
        disjoint_x= tf.concat(tf.unstack(batch_x), axis=0)

        temp_a= tf.unstack(batch_a)
        total_nodes=disjoint_x.shape[0]
        disjoint_a= np.zeros((total_nodes, total_nodes))
        begin=0
        step=len(temp_a[0])
        end=begin+step

        for j in temp_a:
            disjoint_a[begin:end, begin:end]=j
            begin= begin+step
            end= begin+step
        disjoint_a= tf.sparse.from_dense(disjoint_a)

        disjoint_e= np.zeros((total_nodes, total_nodes, batch_e.shape[-1]))

        temp_e= tf.unstack(batch_e)
        begin=0
        step=len(temp_e[0])
        end=begin+step
        for j in temp_e:
            disjoint_e[begin:end, begin:end]=j
            begin= begin+step
            end= begin+step
        dummy_e= []
        adj_indices=disjoint_a.indices

        for idx in adj_indices:
            dummy_e.append(disjoint_e[idx[0], idx[1]])

        edge_idx, edges= reorder(edge_index=np.array(adj_indices), edge_features=np.array(dummy_e))

        #drop zero padding
        #can we use learned clusters to identify similar structures
        return disjoint_x, disjoint_a, edges, tf.cast(i, tf.int32)

    def _vectorised_get_cum_graph_size(self, nodes, graph_sizes):
        """Takes a list of node ids and graph sizes ordered by segment ID and returns the number of nodes contained in graphs with smaller segment ID.
        :param nodes: List of node ids of shape (nodes)
        :param graph_sizes: List of graph sizes (i.e. tf.math.segment_sum(tf.ones_like(I), I) where I are the segment IDs).
        :return: A list of shape (nodes) where each entry corresponds to the number of nodes contained in graphs with smaller segment ID for each node.
        """
        def get_cum_graph_size(node):
            cum_graph_sizes = tf.cumsum(graph_sizes, exclusive=True)
            indicator_if_smaller = tf.cast(node - cum_graph_sizes >= 0, tf.int32)
            graph_id = tf.reduce_sum(indicator_if_smaller) - 1
            return tf.cumsum(graph_sizes, exclusive=True)[graph_id]

        return tf.map_fn(get_cum_graph_size, nodes)
