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


# class PartitionedData(Dataset):
#     def __init__(self, datalist):
#         self.datalist=datalist
#         super().__init__()
#
#     def read(self):
#         return self.datalist

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
        #print(cifs)
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
    def __init__(self, k, channels=None, return_selection=False, activation='relu', kernel_initializer="glorot_uniform",
        kernel_regularizer=None, kernel_constraint=None, column_lambda=1, entr_lambda=1, path='./', **kwargs):

        self.column_lambda= tf.constant(column_lambda, dtype=tf.float32)
        self.entr_lambda= tf.constant(entr_lambda, dtype=tf.float32)
        self.k=tf.constant(k)

        super().__init__(k, channels=channels, return_selection=return_selection, activation=activation,
                kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer, kernel_constraint=kernel_constraint,
                **kwargs)
        self.saveindex=1
        self.savepath= path
        self.assignment_fc= Dense(self.k, use_bias=False)
        #self.crystal_normalize= LayerNormalization()
        self.masker= Masking(mask_value=[0.0, 0.0])

    def build(self, input_shape):
        in_channels = input_shape[0][-1]
        if self.channels is None:
            self.channels = in_channels
        super(DiffPool, self).build(input_shape)

    def select(self, x, a, i, fltr=None, mask=None):
        x = ops.modal_dot(fltr, x)

        s = self.assignment_fc(x)
        masked_s = self.masker(s)
        filename= self.savepath+'/s_after_fc_'+str(self.saveindex)
        #np.savez(filename, s=s)
        masked_tensor= tf.ragged.boolean_mask(masked_s, masked_s._keras_mask)
        means= tf.reduce_mean(masked_tensor, axis=1)
        mean_stack=tf.stack([means]*s.shape[1], axis=1)
        s_interim= tf.subtract(s, mean_stack)
        normalized_crystal= tf.ragged.boolean_mask(s_interim, masked_s._keras_mask)
        s= normalized_crystal.to_tensor(default_value=0.)
        #print(s)
        #print(means)

        #filename= self.savepath+'/s_after_normalizer_'+str(self.saveindex)
        #np.savez(filename, s=s)
        s = activations.softmax(s, axis=-1)
        #filename= self.savepath+'/s_after_softmax_'+str(self.saveindex)
        #np.savez(filename, s=s)
        #self.saveindex+=1

        if mask is not None:
            s *= mask[0]

        # Auxiliary losses
        column_loss, entr_loss= self.row_e_and_column_p(s, i)

        if K.ndim(x) == 3:
            column_loss = K.mean(column_loss)
            entr_loss = K.mean(entr_loss)

        column_loss=tf.multiply(self.column_lambda,column_loss)
        entr_loss= tf.multiply(self.entr_lambda,entr_loss)
        self.add_loss(column_loss)
        self.add_loss(entr_loss)

        return s

    def reduce(self, x, s, fltr=None):
        x = ops.modal_dot(fltr, x)
        return ops.modal_dot(s, x, transpose_a=True)

    def both_entropy(self, s, i):
        batch_size= s.shape[0]
        c_stack=[]
        row_entropy_sum=0
        for g in range(batch_size):
            count= np.count_nonzero(i==g)
            s_g=s[g,:count]
            #---
            row= self.entropy_loss(s_g)
            row_entropy_sum+=row
            #---
            column_means=tf.divide(tf.reduce_sum(s_g, axis=0),s_g.shape[0])
            c_stack.append(column_means)


        c_stack= tf.stack(c_stack, axis=0) #this should give shape(num graphs, k)
        column_entropy = tf.reduce_sum(tf.reduce_sum(tf.multiply(c_stack, K.log(c_stack + 10**-30)), axis=-1), axis=-1)

        return column_entropy, row_entropy_sum

    def entropy_loss(self, s):
        entr = tf.negative(
            tf.reduce_sum(tf.multiply(s, K.log(s + 10**-30)), axis=-1)#should be
        )

        entr_loss = tf.reduce_mean(entr, axis=-1)

        return entr_loss

    def row_e_and_column_p(self, s, i):
        batch_size= s.shape[0]
        column_prod_sum=0
        row_entropy_sum=0
        #print('The function is happening')
        for g in range(batch_size):
            count= np.count_nonzero(i==g)
            s_g=s[g,:count]
            #---
            row= self.entropy_loss(s_g)
            row_entropy_sum+=row
            #---
            #column_means=tf.divide(tf.reduce_sum(s_g, axis=0),s_g.shape[0])
            column_product= tf.math.reduce_prod(tf.divide(tf.reduce_sum(s_g, axis=0),s_g.shape[0]))
            column_prod_sum+= column_product

        return -1*column_prod_sum, row_entropy_sum



class DoubleJanossyDiffPool(RegularizedDiffPool):
    def __init__(self, k, channels=None, return_selection=False, activation='relu', kernel_initializer="glorot_uniform",
        kernel_regularizer=None, kernel_constraint=None, column_lambda=1, entr_lambda=1, path='./', **kwargs):

        super().__init__(k, channels, return_selection, activation, kernel_initializer, kernel_regularizer, kernel_constraint, column_lambda, entr_lambda, path, **kwargs)

    def call(self, inputs, mask=None):
        x, a, i, element_idx = inputs
        self.n_nodes = tf.shape(x)[-2]

        # Graph filter for GNNs
        if K.is_sparse(a):
            i_n = tf.sparse.eye(self.n_nodes, dtype=a.dtype)
            a_ = tf.sparse.add(a, i_n)
        else:
            i_n = tf.eye(self.n_nodes, dtype=a.dtype)
            a_ = a + i_n
        fltr = ops.normalize_A(a_)

        output = self.pool(x, a, i, element_idx=element_idx, fltr=fltr, mask=mask)
        return output

    def reduce(self, x, s, i=[], element_idx=[], fltr=None):
        x = ops.modal_dot(fltr, x)
        all_pools=np.zeros((x.shape[0],3,2,x.shape[2]))
        ta_all = tf.TensorArray(tf.float32, size=0, dynamic_size=True,clear_after_read=False)
        start_idx=0

        for j in range(x.shape[0]):
            #current_crystal= np.zeros((3,2,x.shape[2]))
            ta_crystal = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)

            count= tf.math.count_nonzero(i==j)
            current_id=element_idx[start_idx:start_idx+count]
            start_idx+=count
            unique, u_idx, u_count= tf.unique_with_counts(current_id)
            for k in range(len(unique)):
                idx= tf.where(tf.equal(u_idx,k))[:,0]
                e_pool=ops.modal_dot(tf.gather(s[j], idx), tf.gather(x[j], idx), transpose_a=True)
                #e_pool= tf.reshape(e_pool, (1,e_pool.shape[0],e_pool.shape[1]))
                ta_crystal.write(k,e_pool).mark_used()
                #current_crystal[k]=e_pool

            if len(unique)<3:
                ta_crystal.write(2,tf.zeros((2,x.shape[2]))).mark_used()

            ta_crystal_finished=ta_crystal.stack()

            ta_all.write(j,ta_crystal_finished).mark_used()
            #print(ta_all.element_shape)

            all_pools[j]= ta_crystal_finished
        ta_all_complete= ta_all.stack()

        return ta_all_complete

#
#
# class HNetSimple(Model):
#     def __init__(self, task, num_classes, embedding_size=64, d1=0, d2=0, el=1, cl=1, l2_1=0, l2_2=0, l2_3=0, regularizer='l2', return_s=False,  random_seed=0, **kwargs):
#         super().__init__()
#
#         self.return_s=return_s
#         self.task=task
#         self.num_classes=num_classes
#         glorot_initializer= initializers.glorot_uniform(seed=random_seed)
#         he_initializer= initializers.he_uniform(seed=random_seed)
#
#         self.embedding= Dense(embedding_size, kernel_initializer=glorot_initializer, kernel_regularizer=regularizer)
#
#         self.conv1= CrystalConv(kernel_initializer=glorot_initializer, kernel_regularizer=regularizer)#does this l2 have a lambda
#         self.bn1= BatchNormalization()
#         self.l2_1= tf.constant(l2_1, dtype=tf.float32)
#         self.conv2= CrystalConv(kernel_initializer=glorot_initializer, kernel_regularizer=regularizer)
#         self.bn2= BatchNormalization()
#         self.l2_2= tf.constant(l2_2, dtype=tf.float32)
#         self.conv3= CrystalConv(kernel_initializer=glorot_initializer, kernel_regularizer=regularizer)
#         self.bn3= BatchNormalization()
#         self.l2_3= tf.constant(l2_3, dtype=tf.float32)
#
#         self.disjoint2batch= Disjoint2Batch()
#         self.dropout1= Dropout(d1)
#         self.pool= RegularizedDiffPool(k=2, kernel_initializer=he_initializer, column_lambda=cl, entr_lambda=el, return_selection=True, activation='relu')
#         self.finalpool= DiffPool(k=1, kernel_initializer=he_initializer, activation='relu')
#
#         self.dropout2= Dropout(d2)
#
#         #we should have a dropout after aggregating crystal features
#         if self.task=='c':
#             self.out_layer= Dense(self.num_classes, activation='softmax', kernel_initializer=glorot_initializer)
#         elif self.task=='r':
#             self.out_layer= Dense(1, kernel_initializer=glorot_initializer)
#
#
#     def call(self, inputs):
#         x, a, e, i = inputs
#         x= self.embedding(x)
#         x= self.conv1([x, a, e])
#         self.conv1.add_loss(tf.multiply(self.l2_1,tf.norm(x)))
#         x= self.bn1(x)
#         x= self.conv2([x, a, e])
#         self.conv2.add_loss(tf.multiply(self.l2_2,tf.norm(x)))
#         x= self.bn2(x)
#         x= self.conv3([x, a, e])
#         self.conv3.add_loss(tf.multiply(self.l2_3,tf.norm(x)))
#         x= self.bn3(x)
#
#
#         x= self.dropout1(x)
#         batch_X, batch_A= self.disjoint2batch([x, a, i])
#
#         x, a, s= self.pool([batch_X, batch_A])
#
#         x, a = self.finalpool([x, a])
#         x= self.dropout2(x)
#
#
#         x=self.out_layer(x)
#         if self.return_s:
#             return x, s
#         else:
#             return x
#
# class HNetSiamese(Model):
#     def __init__(self, task, num_classes, embedding_size=64, d1=0, d2=0, el=1, cl=1, regularizer='l2', return_s=False,  random_seed=0, **kwargs):
#         super().__init__()
#         glorot_initializer= initializers.glorot_uniform(seed=random_seed)
#         he_initializer= initializers.he_uniform(seed=random_seed)
#
#         self.return_s=return_s
#         self.task=task
#         self.num_classes=num_classes
#
#         self.embedding= Dense(embedding_size, kernel_initializer=glorot_initializer, kernel_regularizer=regularizer)
#
#         self.conv1= CrystalConv(kernel_initializer=glorot_initializer)#does this l2 have a lambda
#         self.conv2= CrystalConv(kernel_initializer=glorot_initializer)
#         self.conv3= CrystalConv(kernel_initializer=glorot_initializer)
#
#         self.disjoint2batch= Disjoint2Batch()
#         self.dropout1= Dropout(d1)
#
#         self.pool= RegularizedDiffPool(k=2, kernel_initializer=he_initializer, column_lambda=cl, entr_lambda=el, return_selection=True, activation='relu')
#         self.finalpool= GlobalSumPool()
#
#         self.dropout2= Dropout(d2)
#
#         #we should have a dropout after aggregating crystal features
#         if self.task=='c':
#             self.out_layer= Dense(self.num_classes, activation='softmax', kernel_initializer=glorot_initializer)
#         elif self.task=='r':
#             self.out_layer= Dense(1, kernel_initializer=glorot_initializer)
#
#
#     def call(self, inputs):
#         x, a, e, i = inputs
#         x= self.embedding(x)
#         x= self.conv1([x, a, e])
#         x= self.conv2([x, a, e])
#         x= self.conv3([x, a, e])
#
#         x= self.dropout1(x)
#         batch_X, batch_A= self.disjoint2batch([x, a, i])
#         x, a, s= self.pool([batch_X, batch_A])
#
#         w= tf.constant([1.0,-1.0], dtype=tf.float32)
#         x=tf.multiply(x, w[:,tf.newaxis])
#
#         x = self.finalpool([x])
#         x= tf.abs(x)
#
#         x= self.dropout2(x)
#         x=self.out_layer(x)
#
#         if self.return_s:
#             return x, s
#         else:
#             return x
#
# class HNetSigmoid(Model):
#     def __init__(self, task, num_classes, embedding_size=64, d1=0, d2=0, el=1, regularizer='l2', return_s=False,  random_seed=0, **kwargs):
#         super().__init__()
#         self.return_s=return_s
#         self.task=task
#         self.num_classes=num_classes
#         glorot_initializer= initializers.glorot_uniform(seed=random_seed)
#         he_initializer= initializers.he_uniform(seed=random_seed)
#
#         self.embedding= Dense(embedding_size, kernel_initializer=glorot_initializer, kernel_regularizer=regularizer)
#
#         self.conv1= CrystalConv(kernel_initializer=glorot_initializer)
#         self.conv2= CrystalConv(kernel_initializer=glorot_initializer)
#         self.conv3= CrystalConv(kernel_initializer=glorot_initializer)
#
#         self.disjoint2batch= Disjoint2Batch()
#         self.dropout1= Dropout(d1)
#         self.pool= SigmoidalDiffPool(kernel_initializer=he_initializer, entr_lambda=el, return_selection=True, activation='linear')
#         self.dropout2= Dropout(d2)
#         if self.task=='c':
#             self.out_layer= Dense(self.num_classes, activation='softmax', kernel_initializer=glorot_initializer)
#         elif self.task=='r':
#             self.out_layer= Dense(1, kernel_initializer=glorot_initializer)
#
#     def call(self, inputs):
#         x, a, e, i = inputs
#         x= self.embedding(x)
#         x= self.conv1([x, a, e])
#         x= self.conv2([x, a, e])
#         x= self.conv3([x, a, e])
#
#         x= self.dropout1(x)
#         batch_X, batch_A= self.disjoint2batch([x, a, i])
#
#         x, a, s, z= self.pool([batch_X, batch_A])
#
#         s_2= tf.subtract(1, s)
#         z_2=ops.modal_dot(s_2,z, transpose_a=True)
#         x= tf.abs(tf.subtract(x, z_2))
#
#         x= self.dropout2(x)
#
#         x=self.out_layer(x)
#         if self.return_s:
#             return x, s
#         else:
#             return x


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
# class HNetConcat(Model):
#     def __init__(self, task, num_classes, embedding_size=52, d1=0.578, d2=0.302, el=427, cl=265, regularizer='l2', return_s=False,  random_seed=0, **kwargs):
#         super().__init__()
#         glorot_initializer= initializers.glorot_uniform(seed=random_seed)
#         he_initializer= initializers.he_uniform(seed=random_seed)
#
#         self.return_s=return_s
#         self.task=task
#         self.num_classes=num_classes
#
#         self.embedding= Dense(embedding_size, kernel_initializer=glorot_initializer, kernel_regularizer=regularizer)
#
#         self.conv1= ModifiedCrystalConv(activation= 'softplus', kernel_initializer=he_initializer)#does this l2 have a lambda
#         self.conv2= ModifiedCrystalConv(activation= 'softplus', kernel_initializer=he_initializer)
#         self.conv3= ModifiedCrystalConv(activation= 'softplus', kernel_initializer=he_initializer)
#
#         self.disjoint2batch= Disjoint2Batch()
#         self.dropout1= Dropout(d1)
#
#         self.pool= RegularizedDiffPool(k=2, kernel_initializer=he_initializer, column_lambda=cl, entr_lambda=el, return_selection=True, activation='relu')
#         self.finalpool= GlobalSumPool()
#
#         self.dropout2= Dropout(d2)
#         self.fc= Dense(23, activation='softplus', kernel_initializer=he_initializer)
#         #we should have a dropout after aggregating crystal features
#         if self.task=='c':
#             self.out_layer= Dense(self.num_classes, activation='softmax', kernel_initializer=glorot_initializer)
#         elif self.task=='r':
#             self.out_layer= Dense(1, kernel_initializer=glorot_initializer)
#
#
#     def call(self, inputs):
#         x, a, e, i = inputs
#         print(x.shape)
#         x= self.embedding(x)
#         print(x.shape)
#         x= self.conv1([x, a, e])
#         print(x.shape)
#         x= tf.nn.softplus(x)
#         x= self.conv2([x, a, e])
#         print(x.shape)
#         x= tf.nn.softplus(x)
#         x= self.conv3([x, a, e])
#         print(x.shape)
#         x= tf.nn.softplus(x)
#         print('----')
#         x= self.dropout1(x)
#
#         batch_X, batch_A= self.disjoint2batch([x, a, i])
#
#         x_orig, a, i, s= self.pool([batch_X, batch_A, i])
#
#         w= tf.constant([1.0,-1.0], dtype=tf.float32)
#         x=tf.multiply(x_orig, w[:,tf.newaxis])
#
#         x = self.finalpool([x])
#         x= tf.abs(x)
#
#         x_orig= tf.reshape(x_orig, (x.shape[0], x_orig.shape[0], x_orig.shape[1]*x_orig.shape[2]))
#
#         x_new=tf.concat([x,x_orig], axis=2)
#
#         x= self.dropout2(x_new)
#         x= self.fc(x)
#         x=self.out_layer(x)
#         if self.return_s:
#             return x, s
#         else:
#             return x
#
#
# class HNetMultifilter(Model):
#     def __init__(self, task, num_classes, embedding_size=52, d1a=0, d1b=0, d2=0, el=1, cl=1, regularizer='l2', return_s=False,  random_seed=0, **kwargs):
#         super().__init__()
#         glorot_initializer= initializers.glorot_uniform(seed=random_seed)
#         he_initializer= initializers.he_uniform(seed=random_seed)
#         print(d1a, d1b, d2)
#         self.return_s=return_s
#         self.task=task
#         self.num_classes=num_classes
#
#         self.embedding= Dense(embedding_size, kernel_initializer=glorot_initializer, kernel_regularizer=regularizer)
#
#         self.conv1= ModifiedCrystalConv(activation= 'softplus', kernel_initializer=he_initializer)#does this l2 have a lambda
#         self.conv2= ModifiedCrystalConv(activation= 'softplus', kernel_initializer=he_initializer)
#         self.conv3= ModifiedCrystalConv(activation= 'softplus', kernel_initializer=he_initializer)
#
#         self.assign_conv1= ModifiedCrystalConv(activation= 'softplus', kernel_initializer=he_initializer)#does this l2 have a lambda
#         self.assign_conv2= ModifiedCrystalConv(activation= 'softplus', kernel_initializer=he_initializer)
#         self.assign_conv3= ModifiedCrystalConv(activation= 'softplus', kernel_initializer=he_initializer)
#
#         self.disjoint2batch= Disjoint2Batch()
#         self.dropout1= Dropout(d1a)
#
#         self.dropout1_assign= Dropout(d1b)
#
#         self.pool= MultifilterDiffPool(k=2, kernel_initializer=he_initializer, column_lambda=cl, entr_lambda=el, return_selection=True, activation='relu')
#         self.finalpool= GlobalSumPool()
#
#         self.dropout2= Dropout(d2)
#         self.fc= Dense(23, activation='softplus', kernel_initializer=he_initializer)
#         #we should have a dropout after aggregating crystal features
#         if self.task=='c':
#             self.out_layer= Dense(self.num_classes, activation='softmax', kernel_initializer=glorot_initializer)
#         elif self.task=='r':
#             self.out_layer= Dense(1, kernel_initializer=glorot_initializer)
#
#
#     def call(self, inputs):
#         x_in, a_in, e_in, i_in = inputs
#         #print(self.conv1.built)
#         x= self.embedding(x_in)
#         x= self.conv1([x, a_in, e_in])
#         x= tf.nn.softplus(x)
#         x= self.conv2([x, a_in, e_in])
#         x= tf.nn.softplus(x)
#         x= self.conv3([x, a_in, e_in])
#         x= tf.nn.softplus(x)
#
#         x_assign= self.embedding(x_in)
#         x_assign= self.assign_conv1([x_assign, a_in, e_in])
#         x_assign= tf.nn.softplus(x_assign)
#         x_assign= self.assign_conv2([x_assign, a_in, e_in])
#         x_assign= tf.nn.softplus(x_assign)
#         x_assign= self.assign_conv3([x_assign, a_in, e_in])
#         x_assign= tf.nn.softplus(x_assign)
#
#
#         x= self.dropout1(x)
#         x_assign= self.dropout1_assign(x_assign)
#
#         batch_X, batch_A= self.disjoint2batch([x, a_in, i_in])
#         batch_X_assign, batch_A_assign= self.disjoint2batch([x_assign, a_in, i_in])
#
#         x_assign_pooled, a_assign_pooled, s= self.pool([batch_X_assign, batch_A_assign])
#
#         x_orig= ops.modal_dot(s, batch_X, transpose_a=True)
#
#         w= tf.constant([1.0,-1.0], dtype=tf.float32)
#         x=tf.multiply(x_orig, w[:,tf.newaxis])
#
#         x = self.finalpool([x])
#         x= tf.abs(x)
#
#         x_orig= tf.reshape(x_orig, (x.shape[0], x_orig.shape[0], x_orig.shape[1]*x_orig.shape[2]))
#
#         x_new=tf.concat([x,x_orig], axis=2)
#
#         x= self.dropout2(x_new)
#         x= self.fc(x)
#         x=self.out_layer(x)
#
#         if self.return_s:
#             return x, s
#         else:
#             return x
#
#
# class HNetRecurrent(Model):
#     def __init__(self, task, num_classes, embedding_size=52, fc1=23, d1=0, el=1, cl=1, return_s=False, k=2, random_seed=0, **kwargs):
#         super().__init__()
#         glorot_initializer= initializers.glorot_uniform(seed=random_seed)
#         he_initializer= initializers.he_uniform(seed=random_seed)
#
#         transfer_weights= np.load('/Users/nilamandal/Desktop/spk-crystal-hierarchy/total_energy_cgcnn_params.npz')
#         #for x in transfer_weights:
#         #    print(x)
#
#         self.return_s=return_s
#         self.task=task
#         self.num_classes=num_classes
#
#         self.embedding= Dense(embedding_size, kernel_initializer=initializers.constant(transfer_weights['embed']))
#
#         self.conv1= SuperCgcnn(activation= 'softplus', kernel_initializer=initializers.constant(transfer_weights['fc_w_0']), bias_initializer=initializers.constant(transfer_weights['fc_b_0']), transfers=transfer_weights, transfer_idx='0')
#         self.conv2= SuperCgcnn(activation= 'softplus', kernel_initializer=initializers.constant(transfer_weights['fc_w_1']), bias_initializer=initializers.constant(transfer_weights['fc_b_1']), transfers=transfer_weights, transfer_idx='1')
#         self.conv3= SuperCgcnn(activation= 'softplus', kernel_initializer=initializers.constant(transfer_weights['fc_w_2']), bias_initializer=initializers.constant(transfer_weights['fc_b_2']), transfers=transfer_weights, transfer_idx='2')
#
#         self.disjoint2batch= Disjoint2Batch()
#         self.pool= RegularizedDiffPool(k=k, kernel_initializer=he_initializer, column_lambda=cl, entr_lambda=el, return_selection=True, activation='relu')
#
#         self.finalpool= GlobalSumPool()
#         self.fc_afterpool= Dense(fc1, activation='softplus', kernel_initializer=he_initializer)
#
#         if self.task=='c':
#             self.dropout= Dropout(d1)
#             self.out_layer= Dense(self.num_classes, activation='softmax', kernel_initializer=glorot_initializer)
#         elif self.task=='r':
#             self.out_layer= Dense(1, kernel_initializer=glorot_initializer)
#
#     def call(self, inputs):
#         x, a, e, i = inputs
#         #print('INITIAL VALS')
#         #print(x)
#         x= self.embedding(x)
#         #print('EMBEDDED VALS')
#         #print(x)
#         x= self.conv1([x, a, e])
#         #print('CONV1 VALS')
#         #print(x)
#         x= self.conv2([x, a, e])
#         #print('CONV2 VALS')
#         #print(x)
#         x= self.conv3([x, a, e])
#         #print('CONV3 VALS')
#         #print(x)
#
#         batch_X, batch_A= self.disjoint2batch([x, a, i])
#         #print('BATCHED')
#         #print(batch_X)
#         x_pooled, a_pooled, s= self.pool([batch_X, batch_A])
#         #print('POOLED')
#         #print(x_pooled)
#         e_pooled= self.edgepool(e, a, s, i, a_pooled)
#         x, a, e, i = self.batch2disjoint(x_pooled, e_pooled, a_pooled)
#
#         x= self.conv1([x, a, e])
#         x= self.conv2([x, a, e])
#         x= self.conv3([x, a, e])
#
#         x= self.finalpool([x, i])
#         x= self.fc_afterpool(x)
#
#         if self.task=='c':
#             x= self.dropout(x)
#         x=self.out_layer(x)
#         #print('----')
#         if self.return_s:
#             return x, s
#         else:
#             return x
#
#
#     def batch2disjoint(self, batch_x, batch_e, batch_a):
#         x_shape=batch_x.shape
#         i= []
#         for j in range(x_shape[0]):
#             for k in range(x_shape[1]):
#                 i.append(j)
#         disjoint_x= tf.concat(tf.unstack(batch_x), axis=0)
#
#         temp_a= tf.unstack(batch_a)
#         total_nodes=disjoint_x.shape[0]
#         disjoint_a= np.zeros((total_nodes, total_nodes))
#         begin=0
#         step=len(temp_a[0])
#         end=begin+step
#
#         for j in temp_a:
#             disjoint_a[begin:end, begin:end]=j
#             begin= begin+step
#             end= begin+step
#         disjoint_a= tf.sparse.from_dense(disjoint_a)
#
#         disjoint_e= np.zeros((total_nodes, total_nodes, batch_e.shape[-1]))
#
#         temp_e= tf.unstack(batch_e)
#         begin=0
#         step=len(temp_e[0])
#         end=begin+step
#         for j in temp_e:
#             disjoint_e[begin:end, begin:end]=j
#             begin= begin+step
#             end= begin+step
#         dummy_e= []
#         adj_indices=disjoint_a.indices
#
#         for idx in adj_indices:
#             dummy_e.append(disjoint_e[idx[0], idx[1]])
#
#         edge_idx, edges= reorder(edge_index=np.array(adj_indices), edge_features=np.array(dummy_e))
#
#         #drop zero padding
#         #can we use learned clusters to identify similar structures
#         return disjoint_x, disjoint_a, edges, tf.cast(i, tf.int32)
#
#     def edgepool(self, e, a, s, i, a_pooled):
#         indices = a.indices
#         #values = a.values
#         i_nodes, j_nodes = indices[:, 0], indices[:, 1]
#
#         graph_sizes = tf.math.segment_sum(tf.ones_like(i), i)
#         max_n_nodes = tf.reduce_max(graph_sizes)
#         n_graphs = tf.shape(graph_sizes)[0]
#         relative_j_nodes = j_nodes - self._vectorised_get_cum_graph_size(j_nodes, graph_sizes)
#
#         new_indices = tf.transpose(tf.stack([i_nodes, relative_j_nodes]))
#
#         new_indices = tf.cast(new_indices, tf.int32)
#         n_graphs = tf.cast(n_graphs, tf.int32)
#         max_n_nodes = tf.cast(max_n_nodes, tf.int32)
#
#         dense_edge = tf.scatter_nd(
#             new_indices, e, (n_graphs * max_n_nodes, max_n_nodes, 41)
#         )
#
#         batch_edge = tf.reshape(dense_edge, (n_graphs, max_n_nodes, max_n_nodes, 41))
#         batch_edge = tf.cast(batch_edge, tf.float32)
#
#         temp=tf.einsum('bijk,bil->bilk',batch_edge,s)
#         e_pooled=tf.einsum('bmn,bilk->bnlk',s,temp)
#
#         a_extended= tf.stack([a_pooled] * 41, axis=3)
#         e_pooled= tf.divide(e_pooled, a_extended)
#
#         return e_pooled
#
#     def _vectorised_get_cum_graph_size(self, nodes, graph_sizes):
#         """Takes a list of node ids and graph sizes ordered by segment ID and returns the number of nodes contained in graphs with smaller segment ID.
#         :param nodes: List of node ids of shape (nodes)
#         :param graph_sizes: List of graph sizes (i.e. tf.math.segment_sum(tf.ones_like(I), I) where I are the segment IDs).
#         :return: A list of shape (nodes) where each entry corresponds to the number of nodes contained in graphs with smaller segment ID for each node.
#         """
#         def get_cum_graph_size(node):
#             cum_graph_sizes = tf.cumsum(graph_sizes, exclusive=True)
#             indicator_if_smaller = tf.cast(node - cum_graph_sizes >= 0, tf.int32)
#             graph_id = tf.reduce_sum(indicator_if_smaller) - 1
#             return tf.cumsum(graph_sizes, exclusive=True)[graph_id]
#
#         return tf.map_fn(get_cum_graph_size, nodes)
#
# class HNetLasagna(HNetRecurrent):
#     def __init__(self, task, num_classes, embedding_size=52, fc1=23, d1=0, el=1, cl=1, return_s=False, k=2, random_seed=0, **kwargs):
#         super().__init__(task, num_classes, embedding_size, fc1, d1, el, cl, return_s, k, random_seed, **kwargs)
#         glorot_initializer= initializers.glorot_uniform(seed=random_seed)
#         he_initializer= initializers.he_uniform(seed=random_seed)
#
#         transfer_weights= np.load('/Users/nilamandal/Desktop/spk-crystal-hierarchy/total_energy_cgcnn_params.npz')
#
#         self.conv4= SuperCgcnn(activation= 'softplus', kernel_initializer=initializers.constant(transfer_weights['fc_w_0']), bias_initializer=initializers.constant(transfer_weights['fc_b_0']), transfers=transfer_weights, transfer_idx='0')
#         self.conv5= SuperCgcnn(activation= 'softplus', kernel_initializer=initializers.constant(transfer_weights['fc_w_1']), bias_initializer=initializers.constant(transfer_weights['fc_b_1']), transfers=transfer_weights, transfer_idx='1')
#         self.conv6= SuperCgcnn(activation= 'softplus', kernel_initializer=initializers.constant(transfer_weights['fc_w_2']), bias_initializer=initializers.constant(transfer_weights['fc_b_2']), transfers=transfer_weights, transfer_idx='2')
#
#     def call(self, inputs):
#         x, a, e, i = inputs
#         x= self.embedding(x)
#         x= self.conv1([x, a, e])
#         x= self.conv2([x, a, e])
#         x= self.conv3([x, a, e])
#
#         batch_X, batch_A= self.disjoint2batch([x, a, i])
#
#         x_pooled, a_pooled, s= self.pool([batch_X, batch_A])
#         e_pooled= self.edgepool(e, a, s, i, a_pooled)
#         x, a, e, i = self.batch2disjoint(x_pooled, e_pooled, a_pooled)
#
#         x= self.conv4([x, a, e])
#         x= self.conv5([x, a, e])
#         x= self.conv6([x, a, e])
#
#         x= self.finalpool([x, i])
#         x= self.fc_afterpool(x)
#
#         if self.task=='c':
#             x= self.dropout(x)
#         x=self.out_layer(x)
#         #print('----')
#         if self.return_s:
#             return x, s
#         else:
#             return x
#
#
#
# class HNetConcatPretrained(HNetConcat):
#     def __init__(self, task, num_classes, embedding_size=52, d1=0.578, d2=0.302, el=427, cl=265, regularizer='l2', return_s=False,  random_seed=0, **kwargs):
#         super().__init__(task, num_classes, embedding_size, d1, d2, el, cl, regularizer, return_s, random_seed)
#
#         transfer_weights= np.load('/Users/nilamandal/Desktop/spk-crystal-hierarchy/final_energy_cgcnn_params.npz')
#         self.conv1= SuperCgcnn(activation= 'softplus', kernel_initializer=initializers.constant(transfer_weights['fc_w_0']), bias_initializer=initializers.constant(transfer_weights['fc_b_0']), transfers=transfer_weights, transfer_idx='0')
#         self.conv2= SuperCgcnn(activation= 'softplus', kernel_initializer=initializers.constant(transfer_weights['fc_w_1']), bias_initializer=initializers.constant(transfer_weights['fc_b_1']), transfers=transfer_weights, transfer_idx='1')
#         self.conv3= SuperCgcnn(activation= 'softplus', kernel_initializer=initializers.constant(transfer_weights['fc_w_2']), bias_initializer=initializers.constant(transfer_weights['fc_b_2']), transfers=transfer_weights, transfer_idx='2')
#

#
# class HNetConcatJanossy(Model):
#     def __init__(self, task, num_classes, embedding_size=52, d1=0.578, el=427, cl=265, fc_num=1, fc_size=23, regularizer='l2', return_s=False,  random_seed=0, **kwargs):
#         super().__init__()
#         glorot_initializer= initializers.glorot_uniform(seed=random_seed)
#         he_initializer= initializers.he_uniform(seed=random_seed)
#
#         self.return_s=return_s
#         self.task=task
#         self.num_classes=num_classes
#
#         self.embedding= Dense(embedding_size, kernel_initializer=glorot_initializer, kernel_regularizer=regularizer)
#
#         self.conv1= ModifiedCrystalConv(activation= 'softplus', kernel_initializer=he_initializer)#does this l2 have a lambda
#         self.conv2= ModifiedCrystalConv(activation= 'softplus', kernel_initializer=he_initializer)
#         self.conv3= ModifiedCrystalConv(activation= 'softplus', kernel_initializer=he_initializer)
#
#         self.disjoint2batch= Disjoint2Batch()
#         self.dropout1= Dropout(d1)
#
#         self.pool= RegularizedDiffPool(k=2, kernel_initializer=he_initializer, column_lambda=cl, entr_lambda=el, return_selection=True, activation='relu')
#
#         self.fc= Dense(fc_size, activation='softplus', kernel_initializer=he_initializer)
#         self.meanpool= GlobalAvgPool()
#         #we should have a dropout after aggregating crystal features
#         if self.task=='c':
#             self.out_layer= Dense(self.num_classes, activation='softmax', kernel_initializer=glorot_initializer)
#         elif self.task=='r':
#             self.out_layer= Dense(1, kernel_initializer=glorot_initializer)
#
#
#     def call(self, inputs):
#         x, a, e, i = inputs
#
#         x= self.embedding(x)
#         x= self.conv1([x, a, e])
#         x= tf.nn.softplus(x)
#         x= self.conv2([x, a, e])
#         x= tf.nn.softplus(x)
#         x= self.conv3([x, a, e])
#         x= tf.nn.softplus(x)
#
#         x= self.dropout1(x)
#
#         batch_X, batch_A= self.disjoint2batch([x, a, i])
#
#         x_pool_1, a, i, s= self.pool([batch_X, batch_A, i])
#         x_pool_print= tf.identity(x_pool_1)
#
#         #x_pool_1= self.bn1(x_pool_1)
#         x_pool_2= tf.reverse(x_pool_1, [1])
#
#         x_1=tf.reshape(x_pool_1, [x_pool_1.shape[0],x_pool_1.shape[1]*x_pool_1.shape[2]])
#         x_2=tf.reshape(x_pool_2, [x_pool_2.shape[0],x_pool_2.shape[1]*x_pool_2.shape[2]])
#         #print(self.janossy_fc_list)
#         x_1= self.fc(x_1)
#         x_2= self.fc(x_2)
#
#         temp_concat=tf.stack([x_1,x_2],axis=-2)
#
#         x_mean= self.meanpool(temp_concat)
#         #print(x_mean.shape)
#         x=self.out_layer(x_mean)
#
#         if self.return_s:
#             return x, s, x_pool_print
#         else:
#             return x

class ModifiedReduceLROnPlateau(Callback):
    """Args:
        monitor: quantity to be monitored.
        factor: factor by which the learning rate will be reduced.
          `new_lr = lr * factor`.
        patience: number of epochs with no improvement after which learning rate
          will be reduced.
        verbose: int. 0: quiet, 1: update messages.
        mode: one of `{'auto', 'min', 'max'}`. In `'min'` mode,
          the learning rate will be reduced when the
          quantity monitored has stopped decreasing; in `'max'` mode it will be
          reduced when the quantity monitored has stopped increasing; in
          `'auto'` mode, the direction is automatically inferred from the name
          of the monitored quantity.
        min_delta: threshold for measuring the new optimum, to only focus on
          significant changes.
        cooldown: number of epochs to wait before resuming normal operation
          after lr has been reduced.
        min_lr: lower bound on the learning rate.
    """

    def __init__(
        self,
        monitor="val_loss",
        factor=0.1,
        patience=10,
        verbose=0,
        mode="auto",
        min_delta=1e-4,
        cooldown=0,
        optim = None,
        min_lr=0,
        **kwargs,
    ):
        super().__init__()

        self.monitor = monitor
        if factor >= 1.0:
            raise ValueError(
                "ReduceLROnPlateau does not support "
                f"a factor >= 1.0. Got {factor}"
            )
        if "epsilon" in kwargs:
            min_delta = kwargs.pop("epsilon")
            logging.warning(
                "`epsilon` argument is deprecated and "
                "will be removed, use `min_delta` instead."
            )
        self.factor = factor
        self.min_lr = min_lr
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0  # Cooldown counter.
        self.wait = 0
        self.best = 0
        self.mode = mode
        self.optim= optim
        self.monitor_op = None
        self._reset()

    def _reset(self):
        """Resets wait counter and cooldown counter."""
        if self.mode not in ["auto", "min", "max"]:
            logging.warning(
                "Learning rate reduction mode %s is unknown, "
                "fallback to auto mode.",
                self.mode,
            )
            self.mode = "auto"
        if self.mode == "min" or (
            self.mode == "auto" and "acc" not in self.monitor
        ):
            self.monitor_op = lambda a, b: np.less(a, b - self.min_delta)
            self.best = np.Inf
        else:
            self.monitor_op = lambda a, b: np.greater(a, b + self.min_delta)
            self.best = -np.Inf
        self.cooldown_counter = 0
        self.wait = 0

    def on_train_begin(self, logs=None):
        self._reset()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        #logs["lr"] = backend.get_value(self.model.optimizer.lr)
        logs['lr'] = self.optim._learning_rate
        current = logs.get(self.monitor)
        if current is None:
            logging.warning(
                "Learning rate reduction is conditioned on metric `%s` "
                "which is not available. Available metrics are: %s",
                self.monitor,
                ",".join(list(logs.keys())),
            )

        else:
            if self.in_cooldown():
                self.cooldown_counter -= 1
                self.wait = 0

            if self.monitor_op(current, self.best):
                self.best = current
                self.wait = 0
            elif not self.in_cooldown():
                self.wait += 1
                if self.wait >= self.patience:
                    old_lr = self.optim._learning_rate
                    if old_lr > np.float32(self.min_lr):
                        new_lr = old_lr * self.factor
                        new_lr = max(new_lr, self.min_lr)
                        self.optim._learning_rate= self.optim._build_learning_rate(new_lr)
                        #backend.set_value(, new_lr)
                        if self.verbose > 0:
                            print(f"\nEpoch {epoch +1}: "
                                "ReduceLROnPlateau reducing "
                                f"learning rate to {new_lr}.")
                        self.cooldown_counter = self.cooldown
                        self.wait = 0

    def in_cooldown(self):
        return self.cooldown_counter > 0



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

        filename= self.savepath+'/x_before_bn'+str(self.saveindex)
        #np.savez(filename, x=cgcnn.nbr_sumed)

        #x= self.dropout1(x)

        batch_X, batch_A= self.disjoint2batch([x, a, i])
        filename= self.savepath+'/x_after_cgcnn_no_dropout'+str(self.saveindex)
        #np.savez(filename, x=batch_X)

        x_pool_all, a, i, s= self.pool([batch_X, batch_A, i, element_idx])
        self.saveindex+=1
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
#
# class DoubleJanossyPretrained(Model):
#     def __init__(self, task, num_classes, embedding_size=64, cgcnn_num=3, d1=0.578, el=427, cl=265, fc_num=1, fc_size=23, fc_num2=1, fc_size2=23, regularizer='l2', return_s=False,  random_seed=0, **kwargs):
#         super().__init__()
#         glorot_initializer= initializers.glorot_uniform(seed=random_seed)
#         he_initializer= initializers.he_uniform(seed=random_seed)
#
#         self.return_s=return_s
#         self.task=task
#         self.num_classes=num_classes
#
#         self.embedding= Dense(embedding_size, kernel_initializer=glorot_initializer, kernel_regularizer=regularizer)
#
#         transfer_weights= np.load('/Users/nilamandal/Desktop/spk-crystal-hierarchy/final_energy_cgcnn_params.npz')
#         print('first')
#         print(transfer_weights['fc_w_0'].shape)
#         print(transfer_weights['fc_b_0'].shape)
#         print('----')
#         print('second')
#         print(transfer_weights['fc_w_1'].shape)
#         print(transfer_weights['fc_b_1'].shape)
#         print('----')
#         print('third')
#         print(transfer_weights['fc_w_2'].shape)
#         print(transfer_weights['fc_b_2'].shape)
#         print('----')
#         self.conv1= SuperCgcnn(activation= 'softplus', kernel_initializer=initializers.constant(transfer_weights['fc_w_0']), bias_initializer=initializers.constant(transfer_weights['fc_b_0']), transfers=transfer_weights, transfer_idx='0')
#         self.conv2= SuperCgcnn(activation= 'softplus', kernel_initializer=initializers.constant(transfer_weights['fc_w_1']), bias_initializer=initializers.constant(transfer_weights['fc_b_1']), transfers=transfer_weights, transfer_idx='1')
#         self.conv3= SuperCgcnn(activation= 'softplus', kernel_initializer=initializers.constant(transfer_weights['fc_w_2']), bias_initializer=initializers.constant(transfer_weights['fc_b_2']), transfers=transfer_weights, transfer_idx='2')
#         #self.conv_list=[conv1, conv2, conv3]
#
#         self.disjoint2batch= Disjoint2Batch()
#         self.dropout1= Dropout(d1)
#
#         self.pool= DoubleJanossyDiffPool(k=2, kernel_initializer=he_initializer, column_lambda=cl, entr_lambda=el, return_selection=True, activation='relu')
#         #self.bn1= BatchNormalization()
#         self.janossy_orange_list=[]
#         for i in range(fc_num):
#             fc= Dense(fc_size, activation='softplus', kernel_initializer=he_initializer)
#             self.janossy_orange_list.append(fc)
#
#         self.janossy_green_list=[]
#         for i in range(fc_num):
#             fc= Dense(fc_size, activation='softplus', kernel_initializer=he_initializer)
#             self.janossy_green_list.append(fc)
#
#         self.janossy_2_list=[]
#         for i in range(fc_num2):
#             fc= Dense(fc_size2, activation='softplus', kernel_initializer=he_initializer)
#             self.janossy_2_list.append(fc)
#
#         self.meanpool= GlobalAvgPool()
#         #we should have a dropout after aggregating crystal features
#         if self.task=='c':
#             self.out_layer= Dense(self.num_classes, activation='softmax', kernel_initializer=glorot_initializer)
#         elif self.task=='r':
#             self.out_layer= Dense(1, kernel_initializer=glorot_initializer)
#
#     def call(self, inputs):
#         x, a, e, i = inputs
#         element_idx=np.empty((len(x)))
#         for id in range(len(x)):
#             temp=np.nonzero(x[id])[0]
#             element_idx[id]=int(str(temp[0])+str(temp[1]))
#
#         x= self.embedding(x)
#         #embed_x, throwaway= self.disjoint2batch([x, a, i])
#         x= self.conv1([x, a, e])
#         x= tf.nn.softplus(x)
#         x= self.conv2([x, a, e])
#         x= tf.nn.softplus(x)
#         x= self.conv3([x, a, e])
#         x= tf.nn.softplus(x)
#
#         x= self.dropout1(x)
#
#         batch_X, batch_A= self.disjoint2batch([x, a, i])
#
#         x_pool_all, a, i, s= self.pool([batch_X, batch_A, i, element_idx])
#
#         x_pool_p0=x_pool_all[:,:,0]
#         x_pool_p1=x_pool_all[:,:,1]
#
#         x_2o= tf.stack([x_pool_p0[:,0],x_pool_p0[:,2],x_pool_p0[:,1]], axis=1)
#         x_3o= tf.stack([x_pool_p0[:,1],x_pool_p0[:,0],x_pool_p0[:,2]], axis=1)
#         x_4o= tf.stack([x_pool_p0[:,1],x_pool_p0[:,2],x_pool_p0[:,0]], axis=1)
#         x_5o= tf.stack([x_pool_p0[:,2],x_pool_p0[:,0],x_pool_p0[:,1]], axis=1)
#         x_6o= tf.stack([x_pool_p0[:,2],x_pool_p0[:,1],x_pool_p0[:,0]], axis=1)
#
#         x_1o=tf.reshape(x_pool_p0, [x_pool_p0.shape[0],x_pool_p0.shape[1]*x_pool_p0.shape[2]])
#         x_2o=tf.reshape(x_2o, [x_2o.shape[0],x_2o.shape[1]*x_2o.shape[2]])
#         x_3o=tf.reshape(x_3o, [x_3o.shape[0],x_3o.shape[1]*x_3o.shape[2]])
#         x_4o=tf.reshape(x_4o, [x_4o.shape[0],x_4o.shape[1]*x_4o.shape[2]])
#         x_5o=tf.reshape(x_5o, [x_5o.shape[0],x_5o.shape[1]*x_5o.shape[2]])
#         x_6o=tf.reshape(x_6o, [x_6o.shape[0],x_6o.shape[1]*x_6o.shape[2]])
#
#         x_2g= tf.stack([x_pool_p1[:,0],x_pool_p1[:,2],x_pool_p1[:,1]], axis=1)
#         x_3g= tf.stack([x_pool_p1[:,1],x_pool_p1[:,0],x_pool_p1[:,2]], axis=1)
#         x_4g= tf.stack([x_pool_p1[:,1],x_pool_p1[:,2],x_pool_p1[:,0]], axis=1)
#         x_5g= tf.stack([x_pool_p1[:,2],x_pool_p1[:,0],x_pool_p1[:,1]], axis=1)
#         x_6g= tf.stack([x_pool_p1[:,2],x_pool_p1[:,1],x_pool_p1[:,0]], axis=1)
#
#
#         x_1g=tf.reshape(x_pool_p1, [x_pool_p1.shape[0],x_pool_p1.shape[1]*x_pool_p1.shape[2]])
#         x_2g=tf.reshape(x_2g, [x_2g.shape[0],x_2g.shape[1]*x_2g.shape[2]])
#         x_3g=tf.reshape(x_3g, [x_3g.shape[0],x_3g.shape[1]*x_3g.shape[2]])
#         x_4g=tf.reshape(x_4g, [x_4g.shape[0],x_4g.shape[1]*x_4g.shape[2]])
#         x_5g=tf.reshape(x_5g, [x_5g.shape[0],x_5g.shape[1]*x_5g.shape[2]])
#         x_6g=tf.reshape(x_6g, [x_6g.shape[0],x_6g.shape[1]*x_6g.shape[2]])
#
#
#         for layer in self.janossy_orange_list:
#             x_1o= layer(x_1o)
#             x_2o= layer(x_2o)
#             x_3o= layer(x_3o)
#             x_4o= layer(x_4o)
#             x_5o= layer(x_5o)
#             x_6o= layer(x_6o)
#
#         for layer in self.janossy_green_list:
#             x_1g= layer(x_1g)
#             x_2g= layer(x_2g)
#             x_3g= layer(x_3g)
#             x_4g= layer(x_4g)
#             x_5g= layer(x_5g)
#             x_6g= layer(x_6g)
#
#         x_o=tf.stack([x_1o,x_2o,x_3o,x_4o,x_5o,x_6o],axis=-2)
#         x_g=tf.stack([x_1g,x_2g,x_3g,x_4g,x_5g,x_6g],axis=-2)
#
#
#         x_o= self.meanpool(x_o)
#         x_g= self.meanpool(x_g)
#
#         x_og= tf.concat([x_o, x_g], axis=1)
#         x_go= tf.concat([x_g, x_o], axis=1)
#
#         for layer in self.janossy_2_list:
#             x_og = layer(x_og)
#             x_go = layer(x_go)
#         #print(x_og.shape)
#         x_final= tf.stack([x_og, x_go], axis=-2)
#
#         x_final=self.meanpool(x_final)
#
#         x=self.out_layer(x_final)
#         # #print(x.shape)
#         # #print('----')
#         if self.return_s:
#             return x, s
#         else:
#             return x
