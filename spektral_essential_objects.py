from spektral.data import Graph, Dataset, DisjointLoader
from spektral.data.utils import to_batch
from spektral.utils import reorder
from spektral.layers import CrystalConv, DiffPool, ops, GlobalSumPool, GlobalAvgPool, Disjoint2Batch
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Multiply
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

        self.assignment_fc= Dense(1)

    def build(self, input_shape):
        in_channels = input_shape[0][-1]
        if self.channels is None:
            self.channels = in_channels
        super(DiffPool, self).build(input_shape)

    def select(self, x, a, i, fltr=None, mask=None):

        s = self.assignment_fc(x)
        #s = activations.softmax(s, axis=-1)
        s_1 = activations.sigmoid(s)
        s_2 = tf.ones(s_1.shape)
        s_2 = tf.subtract(s_2,s_1)
        s= tf.concat([s_1, s_2], axis=2)

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
        #z = ops.modal_dot(fltr, K.dot(x, self.kernel_emb))
        #z = self.activation(z)
        #z = self.bn_reduce(z)

        return ops.modal_dot(s, x, transpose_a=True)

    def column_entropy(self, s):

        column_sums=tf.math.reduce_sum(s, axis=1)#this should give shape(batch size, k)
        column_means=tf.math.divide(column_sums,s.shape[1])#this should give shape(batch size, k)
        #print(column_means)
        column_logs=tf.math.log(column_means+ K.epsilon())
        #we want to maximize the column entropy to encourage distributing nodes into different pools
        inv_entr = tf.reduce_sum(tf.multiply(column_means, column_logs),axis=-1) #this should be a positive scalar
        inv_entr_sum=tf.reduce_sum(inv_entr)

        return inv_entr_sum

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
        x_orig, a, s= self.pool([batch_X, batch_A])

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


class HNetElementProduct(Model):
    def __init__(self, task, num_classes, embedding_size=52, d1=0, d2=0, el=1, cl=1, regularizer='l2', return_s=False,  random_seed=0, **kwargs):
        super().__init__()
        glorot_initializer= initializers.glorot_uniform(seed=random_seed)
        he_initializer= initializers.he_uniform(seed=random_seed)

        self.return_s=return_s
        self.task=task
        self.num_classes=num_classes

        self.embedding= Dense(embedding_size, kernel_initializer=glorot_initializer, kernel_regularizer=regularizer)

        self.conv1= ModifiedCrystalConv(activation= 'softplus', kernel_initializer=he_initializer)
        self.conv2= ModifiedCrystalConv(activation= 'softplus', kernel_initializer=he_initializer)
        self.conv3= ModifiedCrystalConv(activation= 'softplus', kernel_initializer=he_initializer)

        self.disjoint2batch= Disjoint2Batch()
        self.dropout1= Dropout(d1)

        self.pool= RegularizedDiffPool(k=2, kernel_initializer=he_initializer, column_lambda=cl, entr_lambda=el, return_selection=True, activation='relu')
        self.elementwise_multiply= Multiply()
        self.finalpool= GlobalSumPool()

        self.dropout2= Dropout(d2)
        self.fc= Dense(23, activation='softplus', kernel_initializer=he_initializer)

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

        w= tf.constant([1.0,-1.0], dtype=tf.float32)
        x=tf.multiply(x_orig, w[:,tf.newaxis])
        x = self.finalpool([x])
        x= tf.abs(x)

        x_pre_product= tf.reshape(x_orig, (x_orig.shape[1], x_orig.shape[0], x_orig.shape[2]))
        x_product= self.elementwise_multiply([x_pre_product[0], x_pre_product[1]])
        x_product= tf.reshape(x_product, (1, x_product.shape[0], x_product.shape[1]))

        x_orig= tf.reshape(x_orig, (x.shape[0], x_orig.shape[0], x_orig.shape[1]*x_orig.shape[2]))

        x_new=tf.concat([x,x_orig, x_product], axis=2)
        #print(x.shape)
        #print(x_orig.shape)
        #print(x_product.shape)
        #print(x_new.shape)
        #print('---')

        x= self.dropout2(x_new)
        x= self.fc(x)
        x=self.out_layer(x)

        if self.return_s:
            return x, s
        else:
            return x
