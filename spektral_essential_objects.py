from spektral.data import Graph, Dataset, DisjointLoader
from spektral.data.utils import to_batch
from spektral.utils import reorder
from spektral.layers import CrystalConv, DiffPool, ops, GlobalSumPool, GlobalAvgPool, Disjoint2Batch
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
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

# class HNet(Model):
#     def __init__(self, task, num_classes, return_s=False):
#         super().__init__()
#         self.return_s=return_s
#         self.task=task
#         self.num_classes=num_classes
#         self.embedding= Dense(64)
#         self.conv1= CrystalConv()
#         self.conv2= CrystalConv()
#         self.conv3= CrystalConv()
#
#         self.assign_embedding= Dense(64)
#         self.assign_conv1= CrystalConv()
#         self.assign_conv2= CrystalConv()
#         self.assign_conv3= CrystalConv()
#
#         self.pool= DiffPool(k=3, return_selection=True)
#         self.conv4= CrystalConv()
#         self.avgpool= GlobalAvgPool()
#         self.avgpool.data_mode='disjoint'
#         if self.task=='c':
#             self.out_layer= Dense(self.num_classes, activation='softmax')
#         elif self.task=='r':
#             self.out_layer= Dense(1)
#
#
#
#     def call(self, inputs):
#         x, a, e, i = inputs
#
#         x_assign= self.assign_embedding(x)
#
#         x_assign= self.assign_conv1([x_assign, a, e])
#         x_assign= self.assign_conv2([x_assign, a, e])
#         x_assign= self.assign_conv3([x_assign, a, e])
#
#         x= self.embedding(x)
#         x= self.conv1([x, a, e])
#         x= self.conv2([x, a, e])
#         x= self.conv3([x, a, e])
#
#         batch_X = ops.disjoint_signal_to_batch(x, i)
#         batch_assignfeats= ops.disjoint_signal_to_batch(x_assign, i)
#         batch_A, batch_E = self.local_disjoint_adjacency_to_batch(e, a, i)#had to rewrite
#
#         x_assign, a, s= self.pool([batch_assignfeats, batch_A])
#
#         x_temp=tf.einsum('bij,bmn->bjn',s,batch_X)
#
#         i=tf.convert_to_tensor([k for k in range(0,x_temp.shape[0]) for j in range(0,3)])
#         x= tf.reshape(x_temp, (x_temp.shape[0]*x_temp.shape[1],x_temp.shape[2]))
#
#         temp=tf.einsum('bijk,bil->bilk',batch_E,s)
#         e_new=tf.einsum('bmn,bilk->bnlk',s,temp)
#
#         temp_e=tf.math.reduce_max(e_new, axis=3)
#         a_count=tf.math.count_nonzero(a)
#         e_count=tf.math.count_nonzero(temp_e)
#         zero = tf.constant(0, dtype=tf.float32)
#
#         if a_count!=e_count:
#             where2=tf.not_equal(a, zero)
#             indices_a= tf.where(where2)
#             e=tf.gather_nd(e_new, indices_a)
#
#         else:
#             e= tf.reshape(e_new, (e_new.shape[0]*e_new.shape[1]*e_new.shape[2],e_new.shape[3]))
#
#             temp2=tf.math.reduce_max(e, axis=1)
#             where = tf.not_equal(temp2, zero)
#             indices = tf.where(where)
#
#             temp3=tf.gather(e, indices, axis=0)
#             e=tf.reshape(temp3, (temp3.shape[0],temp3.shape[2]))
#
#         #need to turn adj. back to disjoing mode.
#         adj_empty = np.zeros((x.shape[0], x.shape[0]))
#         idx=0
#         for j in a:
#             adj_empty[idx:idx+3, idx:idx+3]=j
#             idx+=3
#
#         a_new= tf.sparse.from_dense(adj_empty)
#
#         x=self.conv4([x, a_new, e])
#
#         x=self.avgpool([x, i])
#         x=self.out_layer(x)
#
#         if self.return_s:
#             return x, s
#         else:
#             return x
#
#     def local_disjoint_adjacency_to_batch(self, E, A, I):
#         I = tf.cast(I, tf.int64)
#         E = tf.cast(E, tf.float32)
#         A = tf.cast(A, tf.float32)
#         indices = A.indices
#         values = tf.cast(A.values, tf.int64)
#         i_nodes, j_nodes = indices[:, 0], indices[:, 1]
#
#         graph_sizes = tf.math.segment_sum(tf.ones_like(I), I)
#         max_n_nodes = tf.reduce_max(graph_sizes)
#         n_graphs = tf.shape(graph_sizes)[0]
#         relative_j_nodes = j_nodes - self._vectorised_get_cum_graph_size(j_nodes, graph_sizes)
#         #relative_i_nodes = i_nodes - self._vectorised_get_cum_graph_size(i_nodes, graph_sizes)
#
#         #spaced_i_nodes = I * max_n_nodes + relative_i_nodes
#         new_indices = tf.transpose(tf.stack([i_nodes, relative_j_nodes]))
#
#         new_indices = tf.cast(new_indices, tf.int32)
#         n_graphs = tf.cast(n_graphs, tf.int32)
#         max_n_nodes = tf.cast(max_n_nodes, tf.int32)
#
#         #adj
#         dense_adjacency = tf.scatter_nd(
#             new_indices, values, (n_graphs * max_n_nodes, max_n_nodes)
#         )
#
#         batch_adj = tf.reshape(dense_adjacency, (n_graphs, max_n_nodes, max_n_nodes))
#         batch_adj = tf.cast(batch_adj, tf.float32)
#
#         #edge
#         dense_edge = tf.scatter_nd(
#             new_indices, E, (n_graphs * max_n_nodes, max_n_nodes, 41)
#         )
#
#         batch_edge = tf.reshape(dense_edge, (n_graphs, max_n_nodes, max_n_nodes, 41))
#
#         batch_edge = tf.cast(batch_edge, tf.float32)
#
#         return batch_adj, batch_edge
#
#
#     def _vectorised_get_cum_graph_size(self, nodes, graph_sizes):
#         """
#         Takes a list of node ids and graph sizes ordered by segment ID and returns the
#         number of nodes contained in graphs with smaller segment ID.
#         :param nodes: List of node ids of shape (nodes)
#         :param graph_sizes: List of graph sizes (i.e. tf.math.segment_sum(tf.ones_like(I), I) where I are the
#         segment IDs).
#         :return: A list of shape (nodes) where each entry corresponds to the number of nodes contained in graphs
#         with smaller segment ID for each node.
#         """
#
#         def get_cum_graph_size(node):
#             cum_graph_sizes = tf.cumsum(graph_sizes, exclusive=True)
#             indicator_if_smaller = tf.cast(node - cum_graph_sizes >= 0, tf.int32)
#             graph_id = tf.reduce_sum(indicator_if_smaller) - 1
#             return tf.cumsum(graph_sizes, exclusive=True)[graph_id]
#
#         return tf.map_fn(get_cum_graph_size, nodes)


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


class PartitionedData(Dataset):
    def __init__(self, datalist):
        self.datalist=datalist
        super().__init__()

    def read(self):
        return self.datalist

class MyDataset(Dataset):

    def __init__(self, datadir, filename, r_a, num_nbrs, task):
        self.datadir=datadir
        self.filename=filename
        self.radius_angstroms= r_a
        self.num_nbrs= num_nbrs
        self.task= task

        super().__init__()

    def read(self):
        df = pd.read_csv(os.path.join(self.datadir,self.filename), names=['id','target'], header=None)

        df = df.sample(frac=1).reset_index(drop=True)

        allgraphs=[]
        cifs=list(df['id'])
        self.cifs=cifs
        all_atomic_numbers=[]
        for c in cifs:

            c=str(c)
            try:
                crystal= Structure.from_file(os.path.join(self.datadir,c+'.cif'))
            except:
                crystal= Structure.from_file(os.path.join(self.datadir,c))
            num_atoms=len(crystal)

            #ari = AtomCustomJSONInitializer(os.path.join(self.datadir,'atom_init.json'))#check atom initializer
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

            #atom_fea = np.vstack([ari.get_atom_fea(crystal[i].specie.number) for i in range(len(crystal))]) #the features of each element in the atom, in no particular order
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
    def __init__(self, k, channels=None, return_selection=False, activation='relu', kernel_initializer="glorot_uniform",
        kernel_regularizer=None, kernel_constraint=None, column_lambda=1, entr_lambda=1, **kwargs):

        self.column_lambda= tf.constant(column_lambda, dtype=tf.float32)
        self.entr_lambda= tf.constant(entr_lambda, dtype=tf.float32)
        self.bn_reduce= BatchNormalization()
        self.k=tf.constant(k)

        super().__init__(k, channels=channels, return_selection=return_selection, activation=activation,
                kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer, kernel_constraint=kernel_constraint,
                **kwargs)


    def select(self, x, a, i, fltr=None, mask=None):
        
        s = ops.modal_dot(fltr, K.dot(x, self.kernel_pool))
        s = activations.softmax(s, axis=-1)
        if mask is not None:
            s *= mask[0]

        # Auxiliary losses
        column_loss= self.column_entropy(s)
        entr_loss = self.entropy_loss(s)

        if K.ndim(x) == 3:
            column_loss = K.mean(column_loss)
            entr_loss = K.mean(entr_loss)

        column_loss=tf.multiply(self.column_lambda,column_loss)
        entr_loss= tf.multiply(self.entr_lambda,entr_loss)

        self.add_loss(column_loss)
        self.add_loss(entr_loss)

        return s

    def reduce(self, x, s, fltr=None):
        z = ops.modal_dot(fltr, K.dot(x, self.kernel_emb))
        z = self.activation(z)
        z = self.bn_reduce(z)

        return ops.modal_dot(s, z, transpose_a=True)

    def column_entropy(self, s):
        #print(s)
        #print(s.shape)
        temp=tf.math.divide(tf.math.reduce_sum(s, axis=1),s.shape[1])#this should be corrected to average over the number of atoms!
        #we want to maximize the column entropy to encourage distributing nodes into different pools
        inv_entr = tf.reduce_sum(tf.multiply(temp, K.log(temp)), axis=-1) #this should be positive!!!!
        #print(inv_entr)
        return inv_entr

class SigmoidalDiffPool(RegularizedDiffPool):
    def __init__(self, channels=None, return_selection=False, activation='relu', kernel_initializer="glorot_uniform",
        kernel_regularizer=None, kernel_constraint=None, column_lambda=1, entr_lambda=1, **kwargs):

        super().__init__(k=1, channels=channels, return_selection=return_selection, activation=activation,
                kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer, kernel_constraint=kernel_constraint, column_lambda=column_lambda, entr_lambda=entr_lambda,
                **kwargs)

    def select(self, x, a, i, fltr=None, mask=None):
        s = ops.modal_dot(fltr, K.dot(x, self.kernel_pool))
        s = activations.sigmoid(s)

        if mask is not None:
            s *= mask[0]

        # Auxiliary losses
        entr_loss = self.entropy_loss(s)
        if K.ndim(x) == 3:
            entr_loss = K.mean(entr_loss)
        entr_loss= tf.multiply(self.entr_lambda,entr_loss)

        self.add_loss(entr_loss)
        return s

    def reduce(self, x, s, fltr=None):
        z = ops.modal_dot(fltr, K.dot(x, self.kernel_emb))
        z = self.activation(z)
        z = self.bn_reduce(z)
        #return ops.modal_dot(s, z, transpose_a=True)
        return (z, s)

    def get_outputs(self, x_pool, a_pool, i_pool, s):
        z, s= x_pool
        output = [ops.modal_dot(s, z, transpose_a=True), a_pool]
        if i_pool is not None:
            output.append(i_pool)
        if self.return_selection:
            output.append(s)
            output.append(z)
        return output


class HNetSimple(Model):
    def __init__(self, task, num_classes, embedding_size=64, d1=0, d2=0, el=1, cl=1, l2_1=0, l2_2=0, l2_3=0, regularizer='l2', return_s=False,  random_seed=0, **kwargs):
        super().__init__()

        self.return_s=return_s
        self.task=task
        self.num_classes=num_classes
        glorot_initializer= initializers.glorot_uniform(seed=random_seed)
        he_initializer= initializers.he_uniform(seed=random_seed)

        self.embedding= Dense(embedding_size, kernel_initializer=glorot_initializer, kernel_regularizer=regularizer)

        self.conv1= CrystalConv(kernel_initializer=glorot_initializer, kernel_regularizer=regularizer)#does this l2 have a lambda
        self.bn1= BatchNormalization()
        self.l2_1= tf.constant(l2_1, dtype=tf.float32)
        self.conv2= CrystalConv(kernel_initializer=glorot_initializer, kernel_regularizer=regularizer)
        self.bn2= BatchNormalization()
        self.l2_2= tf.constant(l2_2, dtype=tf.float32)
        self.conv3= CrystalConv(kernel_initializer=glorot_initializer, kernel_regularizer=regularizer)
        self.bn3= BatchNormalization()
        self.l2_3= tf.constant(l2_3, dtype=tf.float32)

        self.disjoint2batch= Disjoint2Batch()
        self.dropout1= Dropout(d1)
        self.pool= RegularizedDiffPool(k=2, kernel_initializer=he_initializer, column_lambda=cl, entr_lambda=el, return_selection=True, activation='relu')
        self.finalpool= DiffPool(k=1, kernel_initializer=he_initializer, activation='relu')

        self.dropout2= Dropout(d2)

        #we should have a dropout after aggregating crystal features
        if self.task=='c':
            self.out_layer= Dense(self.num_classes, activation='softmax', kernel_initializer=glorot_initializer)
        elif self.task=='r':
            self.out_layer= Dense(1, kernel_initializer=glorot_initializer)


    def call(self, inputs):
        x, a, e, i = inputs
        x= self.embedding(x)
        x= self.conv1([x, a, e])
        self.conv1.add_loss(tf.multiply(self.l2_1,tf.norm(x)))
        x= self.bn1(x)
        x= self.conv2([x, a, e])
        self.conv2.add_loss(tf.multiply(self.l2_2,tf.norm(x)))
        x= self.bn2(x)
        x= self.conv3([x, a, e])
        self.conv3.add_loss(tf.multiply(self.l2_3,tf.norm(x)))
        x= self.bn3(x)


        x= self.dropout1(x)
        batch_X, batch_A= self.disjoint2batch([x, a, i])

        x, a, s= self.pool([batch_X, batch_A])

        x, a = self.finalpool([x, a])
        x= self.dropout2(x)


        x=self.out_layer(x)
        if self.return_s:
            return x, s
        else:
            return x

class HNetSiamese(Model):
    def __init__(self, task, num_classes, embedding_size=64, d1=0, d2=0, el=1, cl=1, regularizer='l2', return_s=False,  random_seed=0, **kwargs):
        super().__init__()
        glorot_initializer= initializers.glorot_uniform(seed=random_seed)
        he_initializer= initializers.he_uniform(seed=random_seed)

        self.return_s=return_s
        self.task=task
        self.num_classes=num_classes

        self.embedding= Dense(embedding_size, kernel_initializer=glorot_initializer, kernel_regularizer=regularizer)

        self.conv1= CrystalConv(kernel_initializer=glorot_initializer)#does this l2 have a lambda
        self.conv2= CrystalConv(kernel_initializer=glorot_initializer)
        self.conv3= CrystalConv(kernel_initializer=glorot_initializer)

        self.disjoint2batch= Disjoint2Batch()
        self.dropout1= Dropout(d1)

        self.pool= RegularizedDiffPool(k=2, kernel_initializer=he_initializer, column_lambda=cl, entr_lambda=el, return_selection=True, activation='relu')
        self.finalpool= GlobalSumPool()

        self.dropout2= Dropout(d2)

        #we should have a dropout after aggregating crystal features
        if self.task=='c':
            self.out_layer= Dense(self.num_classes, activation='softmax', kernel_initializer=glorot_initializer)
        elif self.task=='r':
            self.out_layer= Dense(1, kernel_initializer=glorot_initializer)


    def call(self, inputs):
        x, a, e, i = inputs
        x= self.embedding(x)
        x= self.conv1([x, a, e])
        x= self.conv2([x, a, e])
        x= self.conv3([x, a, e])

        x= self.dropout1(x)
        batch_X, batch_A= self.disjoint2batch([x, a, i])
        x, a, s= self.pool([batch_X, batch_A])

        w= tf.constant([1.0,-1.0], dtype=tf.float32)
        x=tf.multiply(x, w[:,tf.newaxis])

        x = self.finalpool([x])
        x= tf.abs(x)

        x= self.dropout2(x)
        x=self.out_layer(x)

        if self.return_s:
            return x, s
        else:
            return x

class HNetSigmoid(Model):
    def __init__(self, task, num_classes, embedding_size=64, d1=0, d2=0, el=1, regularizer='l2', return_s=False,  random_seed=0, **kwargs):
        super().__init__()
        self.return_s=return_s
        self.task=task
        self.num_classes=num_classes
        glorot_initializer= initializers.glorot_uniform(seed=random_seed)
        he_initializer= initializers.he_uniform(seed=random_seed)

        self.embedding= Dense(embedding_size, kernel_initializer=glorot_initializer, kernel_regularizer=regularizer)

        self.conv1= CrystalConv(kernel_initializer=glorot_initializer)
        self.conv2= CrystalConv(kernel_initializer=glorot_initializer)
        self.conv3= CrystalConv(kernel_initializer=glorot_initializer)

        self.disjoint2batch= Disjoint2Batch()
        self.dropout1= Dropout(d1)
        self.pool= SigmoidalDiffPool(kernel_initializer=he_initializer, entr_lambda=el, return_selection=True, activation='linear')
        self.dropout2= Dropout(d2)
        if self.task=='c':
            self.out_layer= Dense(self.num_classes, activation='softmax', kernel_initializer=glorot_initializer)
        elif self.task=='r':
            self.out_layer= Dense(1, kernel_initializer=glorot_initializer)

    def call(self, inputs):
        x, a, e, i = inputs
        x= self.embedding(x)
        x= self.conv1([x, a, e])
        x= self.conv2([x, a, e])
        x= self.conv3([x, a, e])

        x= self.dropout1(x)
        batch_X, batch_A= self.disjoint2batch([x, a, i])

        x, a, s, z= self.pool([batch_X, batch_A])

        s_2= tf.subtract(1, s)
        z_2=ops.modal_dot(s_2,z, transpose_a=True)
        x= tf.abs(tf.subtract(x, z_2))

        x= self.dropout2(x)

        x=self.out_layer(x)
        if self.return_s:
            return x, s
        else:
            return x


class ModifiedCrystalConv(CrystalConv):
     def __init__(self, activation= None, kernel_initializer= None, **kwargs):
         super().__init__(self, activation=activation, kernel_initializer=kernel_initializer, **kwargs)
         #print('init complete')
         #print(self.activation)
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
        nbr_sumed = self.dense_s(z) * self.dense_f(z)
        output= self.bn2(nbr_sumed)
        #print(output.shape)
        return output



class HNetConcat(Model):
    def __init__(self, task, num_classes, embedding_size=52, d1=0, d2=0, el=1, cl=1, regularizer='l2', return_s=False,  random_seed=0, **kwargs):
        super().__init__()
        glorot_initializer= initializers.glorot_uniform(seed=random_seed)
        he_initializer= initializers.he_uniform(seed=random_seed)

        self.return_s=return_s
        self.task=task
        self.num_classes=num_classes

        self.embedding= Dense(embedding_size, kernel_initializer=glorot_initializer, kernel_regularizer=regularizer)

        self.conv1= ModifiedCrystalConv(activation= 'softplus', kernel_initializer=he_initializer)#does this l2 have a lambda
        self.conv2= ModifiedCrystalConv(activation= 'softplus', kernel_initializer=he_initializer)
        self.conv3= ModifiedCrystalConv(activation= 'softplus', kernel_initializer=he_initializer)
        # print('modified:')
        # print(self.conv3.built)
        # print(self.conv3)
        # print(self.conv3.aggregate)
        # print(self.conv3.agg)
        # print('----')
        # print(self.conv3.activation)
        # print(self.conv3.use_bias)
        # print(self.conv3.kernel_initializer)
        # print(self.conv3.bias_initializer)


        self.disjoint2batch= Disjoint2Batch()
        self.dropout1= Dropout(d1)

        self.pool= RegularizedDiffPool(k=2, kernel_initializer=he_initializer, column_lambda=cl, entr_lambda=el, return_selection=True, activation='relu')
        self.finalpool= GlobalSumPool()

        self.dropout2= Dropout(d2)
        self.fc= Dense(23, activation='softplus', kernel_initializer=he_initializer)
        #we should have a dropout after aggregating crystal features
        if self.task=='c':
            self.out_layer= Dense(self.num_classes, activation='softmax', kernel_initializer=glorot_initializer)
        elif self.task=='r':
            self.out_layer= Dense(1, kernel_initializer=glorot_initializer)


    def call(self, inputs):
        x, a, e, i = inputs
        #print(self.conv1.built)
        x= self.embedding(x)
        x= self.conv1([x, a, e])
        x= tf.nn.softplus(x)
        x= self.conv2([x, a, e])
        x= tf.nn.softplus(x)
        x= self.conv3([x, a, e])
        x= tf.nn.softplus(x)

        x= self.dropout1(x)
        batch_X, batch_A= self.disjoint2batch([x, a, i])
        x_orig, a, s= self.pool([batch_X, batch_A, i])

        w= tf.constant([1.0,-1.0], dtype=tf.float32)
        x=tf.multiply(x_orig, w[:,tf.newaxis])

        x = self.finalpool([x])
        x= tf.abs(x)

        x_orig= tf.reshape(x_orig, (x.shape[0], x_orig.shape[0], x_orig.shape[1]*x_orig.shape[2]))

        x_new=tf.concat([x,x_orig], axis=2)

        x= self.dropout2(x_new)
        x= self.fc(x)
        x=self.out_layer(x)

        if self.return_s:
            return x, s
        else:
            return x

class HNetNoSub(Model):
    def __init__(self, task, num_classes, embedding_size=52, d1=0, d2=0, el=1, cl=1, regularizer='l2', return_s=False,  random_seed=0, **kwargs):
        super().__init__()
        glorot_initializer= initializers.glorot_uniform(seed=random_seed)
        he_initializer= initializers.he_uniform(seed=random_seed)

        self.return_s=return_s
        self.task=task
        self.num_classes=num_classes

        self.embedding= Dense(embedding_size, kernel_initializer=glorot_initializer, kernel_regularizer=regularizer)

        self.conv1= ModifiedCrystalConv(activation= 'softplus', kernel_initializer=he_initializer)#does this l2 have a lambda
        self.conv2= ModifiedCrystalConv(activation= 'softplus', kernel_initializer=he_initializer)
        self.conv3= ModifiedCrystalConv(activation= 'softplus', kernel_initializer=he_initializer)

        self.disjoint2batch= Disjoint2Batch()
        self.dropout1= Dropout(d1)

        self.pool= RegularizedDiffPool(k=2, kernel_initializer=he_initializer, column_lambda=cl, entr_lambda=el, return_selection=True, activation='relu')
        self.finalpool= GlobalSumPool()

        self.dropout2= Dropout(d2)
        self.fc= Dense(23, activation='softplus', kernel_initializer=he_initializer)
        #we should have a dropout after aggregating crystal features
        if self.task=='c':
            self.out_layer= Dense(self.num_classes, activation='softmax', kernel_initializer=glorot_initializer)
        elif self.task=='r':
            self.out_layer= Dense(1, kernel_initializer=glorot_initializer)


    def call(self, inputs):
        x, a, e, i = inputs

        x= self.embedding(x)
        x= self.conv1([x, a, e])
        x= tf.nn.softplus(x)
        x= self.conv2([x, a, e])
        x= tf.nn.softplus(x)
        x= self.conv3([x, a, e])
        x= tf.nn.softplus(x)

        x= self.dropout1(x)
        batch_X, batch_A= self.disjoint2batch([x, a, i])
        x_orig, a, s= self.pool([batch_X, batch_A])
        #print(x_orig.shape)



        x= tf.reshape(x_orig, (x_orig.shape[0], x_orig.shape[1]*x_orig.shape[2]))
        #print(x.shape)
        #x_new=tf.concat([x,x_orig], axis=2)

        x= self.dropout2(x)
        x= self.fc(x)
        x=self.out_layer(x)

        if self.return_s:
            return x, s
        else:
            return x
