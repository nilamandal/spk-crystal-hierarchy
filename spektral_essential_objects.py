from spektral.data import Graph, Dataset, DisjointLoader
from spektral.data.utils import to_batch
from spektral.layers import CrystalConv, DiffPool, ops, GlobalMaxPool, GlobalAvgPool#, Disjoint2Batch
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import MeanSquaredError, SparseCategoricalCrossentropy
from tensorflow.keras.metrics import sparse_categorical_accuracy
import numpy as np
import pandas as pd
import os
import sys
from pymatgen.core.structure import Structure
import json
import argparse
import time
from keras import backend as BK

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

class AtomCustomJSONInitializer(AtomInitializer):
    def __init__(self, elem_embedding_file):
        with open(elem_embedding_file) as f:
            elem_embedding = json.load(f)

        elem_embedding = {int(key): value for key, value
                          in elem_embedding.items()}
        atom_types = set(elem_embedding.keys())
        super(AtomCustomJSONInitializer, self).__init__(atom_types)
        for key, value in elem_embedding.items():
            self._embedding[key] = np.array(value, dtype=float)

class PartitionedData(Dataset):
    def __init__(self, datalist):
        self.datalist=datalist
        super().__init__()


    def read(self):
        return self.datalist


class MyDataset(Dataset):

    def __init__(self, datadir, filename, r_a, num_atoms, num_nbrs, task):
        self.datadir=datadir
        self.filename=filename
        self.radius_angstroms= r_a
        #self.num_atoms= num_atoms
        self.num_nbrs= num_nbrs
        self.task= task

        super().__init__()

    def read(self):
        df = pd.read_csv(os.path.join(self.datadir,self.filename), names=['id','target'], header=0)
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

            ari = AtomCustomJSONInitializer(os.path.join(self.datadir,'atom_init.json'))#check atom initializer
            atomic_numbers=[crystal[i].specie.number for i in range(len(crystal))]
            all_atomic_numbers= all_atomic_numbers + atomic_numbers
            #print('ATOMIC NUMBERS')
            #print(atomic_numbers)
            atom_fea = np.vstack([ari.get_atom_fea(crystal[i].specie.number) for i in range(len(crystal))]) #the features of each element in the atom, in no particular order
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

            for i in range(len(nbr_fea_idx)):
                for j in range(len(nbr_fea_idx[i])):
                    k=nbr_fea_idx[i][j]
                    adj[i,k]+=1

                    edges[i,k]= nbr_fea[i][j]

            if self.task=='c':
                MG=Graph(x=atom_fea, a=adj, e=edges, y=int(df_MG['target'].values[0]))
                MG._atomlist=set(atomic_numbers)
            elif self.task=='r':
                MG=Graph(x=atom_fea, a=adj, e=edges, y=float(df_MG['target'].values[0]))
                MG._atomlist=set(atomic_numbers)
            else:
                print(self.task, ' is not c or r.')

            allgraphs.append(MG)
        self.all_atomic_numbers= set(all_atomic_numbers)

        return allgraphs#, cifs

    def get_cifs(self):
        return np.asarray(self.cifs , dtype=object)

class HNet(Model):
    def __init__(self, task, num_classes, return_s=False):
        super().__init__()
        self.return_s=return_s
        self.task=task
        self.num_classes=num_classes
        self.embedding= Dense(64)
        self.conv1= CrystalConv()
        self.conv2= CrystalConv()
        self.conv3= CrystalConv()

        self.assign_embedding= Dense(64)
        self.assign_conv1= CrystalConv()
        self.assign_conv2= CrystalConv()
        self.assign_conv3= CrystalConv()

        #self.disjoint2batch= Disjoint2Batch()
        self.pool= DiffPool(k=3, return_selection=True)
        self.conv4= CrystalConv()
        self.maxpool= GlobalAvgPool()
        self.maxpool.data_mode='disjoint'
        if self.task=='c':
            self.out_layer= Dense(self.num_classes, activation='softmax')
        elif self.task=='r':
            #initializer = tf.keras.initializers.HeUniform()
            #reg= tf.keras.regularizers.L2(1)

            #self.out_layer= Dense(1, activation=self.scaled_sigmoid, kernel_initializer= initializer, kernel_regularizer=reg)
            self.out_layer= Dense(1)

    #def scaled_sigmoid(self, x):
    #    return 20/(1+np.e**(-.25*x)) -10
    #    return 10*BK.tanh(x)

    def call(self, inputs):
        x, a, e, i = inputs

        x_assign= self.assign_embedding(x)

        x_assign= self.assign_conv1([x_assign, a, e])
        x_assign= self.assign_conv2([x_assign, a, e])
        x_assign= self.assign_conv3([x_assign, a, e])

        x= self.embedding(x)
        x= self.conv1([x, a, e])
        x= self.conv2([x, a, e])
        x= self.conv3([x, a, e])

        batch_X = ops.disjoint_signal_to_batch(x, i)
        batch_assignfeats= ops.disjoint_signal_to_batch(x_assign, i)
        batch_A, batch_E = self.local_disjoint_adjacency_to_batch(e, a, i)#had to rewrite

        x_assign, a, s= self.pool([batch_assignfeats, batch_A])

        x_temp=tf.einsum('bij,bmn->bjn',s,batch_X)

        i=tf.convert_to_tensor([k for k in range(0,x_temp.shape[0]) for j in range(0,3)])
        x= tf.reshape(x_temp, (x_temp.shape[0]*x_temp.shape[1],x_temp.shape[2]))

        temp=tf.einsum('bijk,bil->bilk',batch_E,s)
        e_new=tf.einsum('bmn,bilk->bnlk',s,temp)

        temp_e=tf.math.reduce_max(e_new, axis=3)
        a_count=tf.math.count_nonzero(a)
        e_count=tf.math.count_nonzero(temp_e)
        zero = tf.constant(0, dtype=tf.float32)

        if a_count!=e_count:
            where2=tf.not_equal(a, zero)
            indices_a= tf.where(where2)
            e=tf.gather_nd(e_new, indices_a)

        else:
            e= tf.reshape(e_new, (e_new.shape[0]*e_new.shape[1]*e_new.shape[2],e_new.shape[3]))

            temp2=tf.math.reduce_max(e, axis=1)
            where = tf.not_equal(temp2, zero)
            indices = tf.where(where)

            temp3=tf.gather(e, indices, axis=0)
            e=tf.reshape(temp3, (temp3.shape[0],temp3.shape[2]))

        #need to turn adj. back to disjoing mode.
        adj_empty = np.zeros((x.shape[0], x.shape[0]))
        idx=0
        for j in a:
            adj_empty[idx:idx+3, idx:idx+3]=j
            idx+=3

        a_new= tf.sparse.from_dense(adj_empty)

        x=self.conv4([x, a_new, e])
        #print('look here')
        #print(x)
        #print(x.shape)


        #x=self.maxpool([x, i])
        #print('after maxpool')
        #print(x)
        x=self.out_layer(tf.reshape(x,(len(a),192)))
        #print(x)
        if self.return_s:
            return x, s
        else:
            return x

    #def my_tf_round(self, x, decimals = 0):
        #print('runding function')
        #print(x)
    #    multiplier = tf.constant(10**decimals, dtype=x.dtype)
        #print(x.dtype)
        #print(multiplier.dtype)
        #temp=x * multiplier
        #temp=tf.round(tf.math.multiply(x, multiplier))
        #print(temp.dtype)
    #    return tf.round(x * multiplier) / multiplier


    def batch_to_disjoint(self, X, A, E):
        pass

    def local_disjoint_adjacency_to_batch(self, E, A, I):
        I = tf.cast(I, tf.int64)
        E = tf.cast(E, tf.float32)
        A = tf.cast(A, tf.float32)
        indices = A.indices
        values = tf.cast(A.values, tf.int64)
        i_nodes, j_nodes = indices[:, 0], indices[:, 1]

        graph_sizes = tf.math.segment_sum(tf.ones_like(I), I)
        max_n_nodes = tf.reduce_max(graph_sizes)
        n_graphs = tf.shape(graph_sizes)[0]
        relative_j_nodes = j_nodes - self._vectorised_get_cum_graph_size(j_nodes, graph_sizes)
        #relative_i_nodes = i_nodes - self._vectorised_get_cum_graph_size(i_nodes, graph_sizes)

        #spaced_i_nodes = I * max_n_nodes + relative_i_nodes
        new_indices = tf.transpose(tf.stack([i_nodes, relative_j_nodes]))

        new_indices = tf.cast(new_indices, tf.int32)
        n_graphs = tf.cast(n_graphs, tf.int32)
        max_n_nodes = tf.cast(max_n_nodes, tf.int32)

        #adj
        dense_adjacency = tf.scatter_nd(
            new_indices, values, (n_graphs * max_n_nodes, max_n_nodes)
        )

        batch_adj = tf.reshape(dense_adjacency, (n_graphs, max_n_nodes, max_n_nodes))
        batch_adj = tf.cast(batch_adj, tf.float32)

        #edge
        dense_edge = tf.scatter_nd(
            new_indices, E, (n_graphs * max_n_nodes, max_n_nodes, 41)
        )

        batch_edge = tf.reshape(dense_edge, (n_graphs, max_n_nodes, max_n_nodes, 41))

        batch_edge = tf.cast(batch_edge, tf.float32)

        return batch_adj, batch_edge


    def _vectorised_get_cum_graph_size(self, nodes, graph_sizes):
        """
        Takes a list of node ids and graph sizes ordered by segment ID and returns the
        number of nodes contained in graphs with smaller segment ID.
        :param nodes: List of node ids of shape (nodes)
        :param graph_sizes: List of graph sizes (i.e. tf.math.segment_sum(tf.ones_like(I), I) where I are the
        segment IDs).
        :return: A list of shape (nodes) where each entry corresponds to the number of nodes contained in graphs
        with smaller segment ID for each node.
        """

        def get_cum_graph_size(node):
            cum_graph_sizes = tf.cumsum(graph_sizes, exclusive=True)
            indicator_if_smaller = tf.cast(node - cum_graph_sizes >= 0, tf.int32)
            graph_id = tf.reduce_sum(indicator_if_smaller) - 1
            return tf.cumsum(graph_sizes, exclusive=True)[graph_id]

        return tf.map_fn(get_cum_graph_size, nodes)
