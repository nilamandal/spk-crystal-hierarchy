from spektral.data import Graph, Dataset, DisjointLoader
from spektral.data.utils import to_batch
from spektral.utils import reorder, sp_matrix_to_sp_tensor
from spektral.layers import CrystalConv, DiffPool, ops, GlobalSumPool, GlobalAvgPool, Disjoint2Batch, GraphSageConv, GATConv
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Multiply, Masking, LayerNormalization
from tensorflow.keras.losses import MeanSquaredError, SparseCategoricalCrossentropy
from tensorflow.keras.metrics import sparse_categorical_accuracy
from tensorflow.keras.regularizers import L2
from tensorflow.keras.models import clone_model
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
import matplotlib.pyplot as plt
import copy
#from torch_compatible_objects import AtomInitializer, AtomCustomJSONInitializer, GaussianDistance

class AtomInitializer(object):
    """
    Base class for intializing the vector representation for atoms.

    !!! Use one AtomInitializer per dataset !!!
    """
    def __init__(self, atom_types):
        self.atom_types = set(atom_types)
        self._embedding = {}

    def get_atom_fea(self, atom_type):
        assert atom_type in self.atom_types
        return self._embedding[atom_type]

    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.atom_types = set(self._embedding.keys())
        self._decodedict = {idx: atom_type for atom_type, idx in
                            self._embedding.items()}

    def state_dict(self):
        return self._embedding

    def decode(self, idx):
        if not hasattr(self, '_decodedict'):
            self._decodedict = {idx: atom_type for atom_type, idx in
                                self._embedding.items()}
        return self._decodedict[idx]


class AtomCustomJSONInitializer(AtomInitializer):
    """
    Initialize atom feature vectors using a JSON file, which is a python
    dictionary mapping from element number to a list representing the
    feature vector of the element.

    Parameters
    ----------

    elem_embedding_file: str
        The path to the .json file
    """
    def __init__(self, elem_embedding_file):
        with open(elem_embedding_file) as f:
            elem_embedding = json.load(f)
        elem_embedding = {int(key): value for key, value
                          in elem_embedding.items()}
        atom_types = set(elem_embedding.keys())
        super(AtomCustomJSONInitializer, self).__init__(atom_types)
        for key, value in elem_embedding.items():
            self._embedding[key] = np.array(value, dtype=float)



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


class AtomFeaDataset(MyDataset):
    def __init__(self, df, datadir, r_a, num_nbrs, task):
        self.ari = AtomCustomJSONInitializer(datadir+'/atom_init.json')
        self.gdf = GaussianDistance(dmin=0, dmax=8, step=0.2)
        super().__init__(df, datadir, r_a, num_nbrs, task)
        #print('making dataset')

    def read(self):
        df = self.dataframe.sample(frac=1).reset_index(drop=True)
        allgraphs=[]
        #print(df)
        cifs=list(df['id'])
        self.cifs=cifs
        all_atomic_numbers=[]
        for c in cifs:
            c=str(c)
            df_c=df[df['id']==c]
            #print(df_c)

            try:
                crystal= Structure.from_file(os.path.join(self.datadir,c))
            except:
                crystal= Structure.from_file(os.path.join(self.datadir,c+'.cif'))
            num_atoms=len(crystal)

            atom_fea = np.vstack([self.ari.get_atom_fea(crystal[i].specie.number)
                                  for i in range(len(crystal))])
            atom_fea_new=[]
            atomic_numbers=[crystal[i].specie.number for i in range(len(crystal))]


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
                target_encoding= np.zeros(8)
                target_encoding[df_MG['target'].values[0]]= 1
                #print(target_encoding)
                MG=Graph(x=atom_fea, a=adj, e=edges, y=target_encoding)
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


class Graphsage_dataset(MyDataset):
    def __init__(self, df, datadir, r_a, num_nbrs, task):
        self.ari = AtomCustomJSONInitializer(datadir+'/atom_init.json')
        self.gdf = GaussianDistance(dmin=0, dmax=8, step=0.2)
        super().__init__(df, datadir, r_a, num_nbrs, task)
        #print('making dataset')

    def read(self):
        df = self.dataframe.sample(frac=1).reset_index(drop=True)
        allgraphs=[]
        #print(df)
        cifs=list(df['id'])
        self.cifs=cifs
        all_atomic_numbers=[]
        for c in cifs:
            c=str(c)
            df_c=df[df['id']==c]
            #print(df_c)

            try:
                crystal= Structure.from_file(os.path.join(self.datadir,c))
            except:
                crystal= Structure.from_file(os.path.join(self.datadir,c+'.cif'))
            num_atoms=len(crystal)

            atom_fea = np.vstack([self.ari.get_atom_fea(crystal[i].specie.number)
                                  for i in range(len(crystal))])
            atom_fea_new=[]
            atomic_numbers=[crystal[i].specie.number for i in range(len(crystal))]


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

            for i in range(len(nbr_fea_idx)):
                for j in range(len(nbr_fea_idx[i])):
                    k=nbr_fea_idx[i][j]
                    adj[i,k]=1

            adj=sp.csr_matrix(adj)

            if self.task=='c':
                target_encoding= np.zeros(8)
                target_encoding[df_MG['target'].values[0]]= 1
                #print(target_encoding)
                MG=Graph(x=atom_fea, a=adj, y=target_encoding)
                MG._atomlist=set(atomic_numbers)
                MG._cif=c
            elif self.task=='r':
                MG=Graph(x=atom_fea, a=adj, y=float(df_MG['target'].values[0]))
                MG._atomlist=set(atomic_numbers)
                MG._cif=c
            else:
                print(self.task, ' is not c or r.')
            allgraphs.append(MG)
        self.all_atomic_numbers= set(all_atomic_numbers)


        return allgraphs





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


class NoShrinkDiffPool(RegularizedDiffPool):
    def __init__(self, k, beta=1, channels=None, return_selection=False, activation='relu', kernel_initializer="glorot_uniform",
        kernel_regularizer=None, kernel_constraint=None,  path='./', **kwargs):

        super().__init__(k, beta=beta, channels=channels, return_selection=return_selection, activation=activation,
                kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer, kernel_constraint=kernel_constraint,
                **kwargs)

    def reduce(self, x, s, fltr=None):
        x = ops.modal_dot(fltr, x)
        x_new= ops.modal_dot(s, x, transpose_a=True)
        x_two= ops.modal_dot(s, x_new, transpose_a=False)

        return x_two

    def connect(self, a, s, **kwargs):
        return a


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

class MPCgcnn(CrystalConv):
    def __init__(self, activation= None, kernel_initializer= None, transfer_weights=[], idx=0, **kwargs):
        super().__init__(self, activation=activation, kernel_initializer=kernel_initializer, **kwargs)

        self.transfer_weights=transfer_weights
        self.transfer_idx=str(idx)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        layer_kwargs = dict(
            kernel_initializer=initializers.constant(self.transfer_weights['fc_full_weight_'+self.transfer_idx]),
            bias_initializer=initializers.constant(self.transfer_weights['fc_full_bias_'+self.transfer_idx]),
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint,
            dtype=self.dtype,
        )
        channels = input_shape[0][-1] * 2

        self.dense_fc = Dense(channels, activation=self.activation, **layer_kwargs)
        #self.dense_s = Dense(channels, activation=self.activation, **layer_kwargs)

        bn1_w= 'bn1_weight_'+self.transfer_idx
        bn1_b= 'bn1_bias_'+self.transfer_idx
        bn2_w= 'bn2_weight_'+self.transfer_idx
        bn2_b= 'bn2_bias_'+self.transfer_idx
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

       nbr_filter, nbr_core= tf.split(z, 2, axis=1)
       nbr_filter= tf.sigmoid(nbr_filter)
       nbr_core= tf.keras.activations.softplus(nbr_core)
       nbr_sumed=nbr_filter * nbr_core
       output= tf.keras.activations.softplus(tf.math.add(x_i, self.bn2(nbr_sumed)))
       return output
#





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

        disjoint_adj= np.zeros((total_nodes, total_nodes))
        begin=0
        step=len(temp_a[0])
        end=begin+step

        for j in temp_a:
            disjoint_adj[begin:end, begin:end]= j#np.ones((self.k,self.k))
            begin= begin+step
            end= begin+step

        disjoint_adj= tf.sparse.from_dense(disjoint_adj)

        #edge
        adj_indices=disjoint_adj.indices
        disjoint_e= np.zeros((total_nodes, total_nodes))

        #temp_e= tf.unstack(batch_edge)
        edge_idx= batch_edge.indices
        edge_vals= batch_edge.values

        edge_idx, edges= reorder(edge_index=np.array(adj_indices), edge_features=np.array(edge_vals))

        return disjoint_adj, np.reshape(edges, [edges.shape[0],1])

class CGCNNModel(Model):
    def __init__(self, embedding_size=64, hidden_fea_size=128, regularizer='l2', random_seed=0):
        super().__init__()
        glorot_initializer= initializers.glorot_uniform(seed=random_seed)
        he_initializer= initializers.he_uniform(seed=random_seed)

        self.embedding= Dense(embedding_size, kernel_initializer=glorot_initializer, kernel_regularizer=regularizer)

        self.conv_list=[]
        for i in range(3):
            conv= ModifiedCrystalConv(activation= 'softplus', kernel_initializer=he_initializer)#does this l2 have a lambda
            self.conv_list.append(conv)

        self.meanpool= GlobalAvgPool()

        self.conv_to_fc = Dense(hidden_fea_size, activation= 'softplus', kernel_initializer=glorot_initializer, kernel_regularizer=regularizer)

        self.out_layer= Dense(1, kernel_initializer=glorot_initializer)

    def call(self, inputs):
        x, a, e, i = inputs

        x= self.embedding(x)

        for cgcnn in self.conv_list:
            x= cgcnn([x, a, e])
            x= tf.nn.softplus(x)

        x_crys= self.meanpool([x, i])
        x_crys= self.conv_to_fc(x_crys)
        out= self.out_layer(x_crys)
        return out




class NotShrinking(SparseEdgepool):
    def __init__(self, task, num_classes, embedding_size=52, cgcnn_num=3, cgcnn_num2=3, regularizer='l2', return_s=True, random_seed=0, path='./', k= 2, **kwargs):

        super().__init__(task, num_classes, embedding_size, cgcnn_num, cgcnn_num2, regularizer, return_s,  random_seed=random_seed, path=path, k=k, **kwargs)
        glorot_initializer= initializers.glorot_uniform(seed=random_seed)
        he_initializer= initializers.he_uniform(seed=random_seed)

        self.pool= NoShrinkDiffPool(k=self.k, return_selection=True)
        self.disjoint_masker= Masking(mask_value=np.zeros(embedding_size))

    def call(self, inputs):
        x, a, e, i = inputs
        x= self.embedding(x)

        for cgcnn in self.conv_list:
            x= cgcnn([x, a, e])
            x= tf.nn.softplus(x)

        batch_X = ops.disjoint_signal_to_batch(x, i)
        batch_A= self.disjoint_adjacency_to_batch(a, i)

        x_pool, a_pool, i_pool, s= self.pool([batch_X, batch_A, i])
        x_pool= tf.reshape(x_pool, [x_pool.shape[0]*x_pool.shape[1], x_pool.shape[2]]) #reshape to disjoint form


        masked_x = self.disjoint_masker(x_pool)
        x_pool= tf.ragged.boolean_mask(masked_x, masked_x._keras_mask)

        for cgcnn2 in self.conv_list2:
            x_pool= cgcnn2([x_pool, a, e])
            x_pool= tf.nn.softplus(x_pool)

        x_pool= tf.math.segment_mean(x_pool, i)

        x=self.out_layer(x_pool)

        if self.return_s:
            return x, s
        else:
            return x


class NoShrink_GAT(Model):
    def __init__(self, task, embedding_size=32, hidden_size=32, num_layers=2, random_seed=100, **kwargs):
        super().__init__()
        self.task= task
        glorot_initializer= initializers.glorot_uniform(seed=random_seed)
        he_initializer= initializers.he_uniform(seed=random_seed)

        self.embedding= Dense(embedding_size, kernel_initializer=glorot_initializer)

        self.conv_list=[]
        for i in range(num_layers):
            conv= GATConv(hidden_size)
            self.conv_list.append(conv)

        self.pool= GlobalAvgPool()

        if self.task=='c':
            self.out_layer= Dense(self.num_classes, activation='softmax', kernel_initializer=glorot_initializer)
        elif self.task=='r':
            self.out_layer= Dense(1, kernel_initializer=glorot_initializer)


    def call(self, inputs):
        x, a, e = inputs
        x= self.embedding(x)

        for gat in self.conv_list:
            x= gat([x, a])

        x_pool= self.pool(x)

        x= self.out_layer(x_pool)

        return x, None
