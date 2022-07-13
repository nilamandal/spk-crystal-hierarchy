from spektral.data import Graph, Dataset, DisjointLoader
from spektral.data.utils import to_batch
from spektral.layers import CrystalConv, DiffPool, ops, GlobalMaxPool#, Disjoint2Batch
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import MeanSquaredError, SparseCategoricalCrossentropy
from tensorflow.keras.metrics import categorical_accuracy
import numpy as np
import pandas as pd
import os
import sys
from pymatgen.core.structure import Structure
import json
import argparse
import time
#from spektral.datasets import QM9
begin_time = time.time()
parser = argparse.ArgumentParser(description='crystal hierarchy arguments.')

parser.add_argument('--datadir', dest='datadir',
        help='Directory where dataset is located', default='../crystalhierarchydata/icsd-zintl-search')

parser.add_argument('--filename', dest='filename',
                    help='csv where data is located', default='sysout.txt')
parser.add_argument('--file-out', dest='file_out',
                    help='output txt file name', default='.csv')
parser.add_argument('--num-atoms', dest='num_atoms', type=int,
                    help='Maximum number of nodes', default=8)
parser.add_argument('--num-nbrs', dest='num_nbrs', type=int,
                    help='num neighbors per atom', default=12)

parser.add_argument('--num-classes', dest='num_classes', type=int,
                    help='Number of label classes', default=3)

parser.add_argument('--radius-angstroms', dest='radius_angstroms', type=int,
                    help='search radius for neighbors', default=8)
parser.add_argument('--random-seed', dest='random_seed', type=int,
                    help='random seed for numpy', default=0)
parser.add_argument('--batch-size', dest='batch_size', type=int,
                    help='Batch size.', default=256)

parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run (default: 30)')

parser.add_argument('--lr', dest='learning_rate', type=float,
                    help='Learning rate.', default=1e-3)
parser.add_argument('--task', choices=['r', 'c'],
                    default='c', help='complete a regression or '
                        'classification task (default: regression)')
parser.add_argument('--patience', dest='patience',default=10, type=int,
                    help='num epochs for early stopping')

#datadir=
#filename='corrected_sym.csv'
#filename='id_prop.csv'
#num_atoms=8
#num_nbrs=12
#num_classes=3
#radius_angstroms=8
#batch_size=1
#epochs=30
#learning_rate =
#es_patience = 10  # Patience for early stopping
#task= 'c' #'c' for classification, 'r' for regression

args = parser.parse_args(sys.argv[1:])

np.random.seed(args.random_seed)
path = './spektraltest_8atom/'
if not os.path.exists(path):
    os.makedirs(path)
sys.stdout = open(path+'/'+args.file_out, 'w')



print(args)

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

class MyDataset(Dataset):
    """
    A dataset of 8-atom crystals.
    """
    def __init__(self, **kwargs):
        #self.nodes = nodes
        #self.feats = feats

        super().__init__(**kwargs)

    def read(self):
        df = pd.read_csv(os.path.join(args.datadir,args.filename), names=['id','target'], header=0)
        allgraphs=[]
        cifs=list(df['id'])
        #printhelper=0
        for c in cifs:
            c=str(c)
            try:
                crystal= Structure.from_file(os.path.join(args.datadir,c,'.cif'))
            except:
                crystal= Structure.from_file(os.path.join(args.datadir,c))
            ari = AtomCustomJSONInitializer(os.path.join(args.datadir,'atom_init.json'))#check atom initializer
            atomic_numbers=[crystal[i].specie.number for i in range(len(crystal))]
            atom_fea = np.vstack([ari.get_atom_fea(crystal[i].specie.number) for i in range(len(crystal))]) #the features of each element in the atom, in no particular order
            #atom_fea = Tensor(atom_fea)
            all_nbrs = crystal.get_all_neighbors(args.radius_angstroms, include_index=True)
            all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
            nbr_fea_idx, nbr_fea = [], []
            for nbr in all_nbrs:
                if len(nbr) < args.num_nbrs:
                    warnings.warn('{} not find enough neighbors to build graph. '
                                  'If it happens frequently, consider increase '
                                  'radius.'.format(cif_id))
                    nbr_fea_idx.append(list(map(lambda x: x[2], nbr)) +
                                       [0] * (args.num_nbrs - len(nbr)))
                    nbr_fea.append(list(map(lambda x: x[1], nbr)) +
                                   [args.radius_angstroms + 1.] * (args.num_nbrs -
                                                         len(nbr)))
                else:
                    nbr_fea_idx.append(list(map(lambda x: x[2],
                                                nbr[:args.num_nbrs])))
                    nbr_fea.append(list(map(lambda x: x[1],
                                            nbr[:args.num_nbrs])))
            df_MG=df[df['id'].astype(str)==c]
            gdf = GaussianDistance(dmin=0, dmax=8, step=0.2)
            nbr_fea = gdf.expand(np.array(nbr_fea))
            #nbr_fea = Tensor(nbr_fea)
            adj = np.zeros((args.num_atoms, args.num_atoms))
            edges= np.zeros((args.num_atoms, args.num_atoms, 41))
            #if 'Cs1As3Ba4_a3b1c4_sg123' in c:
            for i in range(len(nbr_fea_idx)):
                for j in nbr_fea_idx[i]:
                    adj[i,j]+=1

                    edges[i,j]= nbr_fea[i][j]
            #if np.array_equal(adj, np.transpose(adj))==False:
            #    print(c, df_MG['target'].values[0])
            #    for i in range(len(nbr_fea_idx)):
            #        print(nbr_fea_idx[i])

            if args.task=='c':
                MG=Graph(x=atom_fea, a=adj, e=edges, y=int(df_MG['target'].values[0]))
            elif args.task=='r':
                MG=Graph(x=atom_fea, a=adj, e=edges, y=float(df_MG['target'].values[0]))
            else:
                print(args.task, ' is not c or r.')
            #if float(df_MG['target'].values[0])==printhelper:
            #    print(adj)
            #    printhelper+=1
            #print(MG)
            allgraphs.append(MG)
            #print(adj)
        return allgraphs

class HNet(Model):
    def __init__(self):
        super().__init__()
        self.embedding= Dense(64)
        self.conv1= CrystalConv()
        self.conv2= CrystalConv()
        self.conv3= CrystalConv()
        #self.disjoint2batch= Disjoint2Batch()
        self.pool= DiffPool(k=3, return_selection=True)
        self.conv4= CrystalConv()
        self.maxpool= GlobalMaxPool()
        self.maxpool.data_mode='disjoint'
        if args.task=='c':
            self.out_layer= Dense(args.num_classes, activation='softmax')
        elif args.task=='r':
            self.out_layer= Dense(1, activation='softmax')

    def call(self, inputs):
        x, a, e, i = inputs
        x= self.embedding(x)
        x= self.conv1([x, a, e])
        x= self.conv2([x, a, e])
        x= self.conv3([x, a, e])

        batch_X = ops.disjoint_signal_to_batch(x, i)
        batch_A, batch_E = self.local_disjoint_adjacency_to_batch(e, a, i)#had to rewrite

        x, a, s= self.pool([batch_X, batch_A])

        i=tf.convert_to_tensor([k for k in range(0,x.shape[0]) for j in range(0,3)])
        x= tf.reshape(x, (x.shape[0]*x.shape[1],x.shape[2]))

        temp=tf.einsum('bijk,bil->bilk',batch_E,s)
        e_new=tf.einsum('bmn,bilk->bnlk',s,temp)

        temp_e=tf.math.reduce_max(e_new, axis=3)
        a_count=tf.math.count_nonzero(a)
        e_count=tf.math.count_nonzero(temp_e)
        zero = tf.constant(0, dtype=tf.float32)

        if a_count!=e_count:

            print('rip')
            print(e_new.shape)
            print(e_new)
            print(a.shape)
            print(a)
            print(a_count, e_count)
            where2=tf.not_equal(a, zero)
            indices_a= tf.where(where2)
            print(indices_a)
            e=tf.gather_nd(e_new, indices_a)

        else:
            e= tf.reshape(e_new, (e_new.shape[0]*e_new.shape[1]*e_new.shape[2],e_new.shape[3]))

            temp2=tf.math.reduce_max(e, axis=1)
            where = tf.not_equal(temp2, zero)
            indices = tf.where(where)

            temp3=tf.gather(e, indices, axis=0)
            e=tf.reshape(temp3, (temp3.shape[0],temp3.shape[2]))
            #print('----')
            #print(e.shape)
            #print('----')
        #     where = tf.not_equal(temp_e, zero)
        #     indices_e = tf.where(where)
        #
        #     where2=tf.not_equal(a, zero)
        #     indices_a= tf.where(where2)
        #     #print('here---')
        #     #print('a:', a_count, indices_a.shape)
        #     #print('e:', e_count, indices_e.shape)
        #
        #     i_a=0
        #     i_e=0
        #     mismatch=[]
        #     while i_a<len(indices_a) and i_e<len(indices_e):
        #         if tf.reduce_all(tf.math.equal(indices_a[i_a],indices_e[i_e])):
        #            i_a+=1
        #            i_e+=1
        #         else: #actually should calculate who is "ahead" of who
        #             #print('mismatch')
        #             #print(indices_a[i_a],indices_e[i_e])
        #             mismatch.append(i_e)
        #             i_e+=1
        #
        #
        #     #print('diff:', len(mismatch))
        #     #print(mismatch)
        #
        #     index_hope=list(range(len(e)))
        #     index_hope=[k for k in index_hope if k not in mismatch]
        #     e=tf.gather(e,index_hope)
        #     #print(e)
        #     #print("end of for loop---")


        #need to turn adj. back to disjoing mode.
        adj_empty = np.zeros((x.shape[0], x.shape[0]))
        idx=0
        for j in a:
            adj_empty[idx:idx+3, idx:idx+3]=j
            idx+=3
        #print(adj_empty)
        a_new= tf.sparse.from_dense(adj_empty)

        x=self.conv4([x, a_new, e])
        x=self.maxpool([x, i])
        x=self.out_layer(x)
        #print('here is x')
        #print(x)
        return x

    def my_tf_round(self, x, decimals = 0):
        #print('runding function')
        #print(x)
        multiplier = tf.constant(10**decimals, dtype=x.dtype)
        #print(x.dtype)
        #print(multiplier.dtype)
        #temp=x * multiplier
        #temp=tf.round(tf.math.multiply(x, multiplier))
        #print(temp.dtype)
        return tf.round(x * multiplier) / multiplier


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


def evaluate(loader):
    output = []
    step = 0
    while step < loader.steps_per_epoch:
        step += 1
        inputs, target = loader.__next__()
        pred = model(inputs, training=False)
        outs = (
            loss_fn(target, pred),
            tf.reduce_mean(categorical_accuracy(target, pred)),
            len(target),  # Keep track of batch size
        )
        output.append(outs)
        if step == loader.steps_per_epoch:
            output = np.array(output)
            return np.average(output[:, :-1], 0, weights=output[:, -1])

data= MyDataset()
datasettime=time.time()-begin_time
print('datset generated: time=', str(datasettime))
#data = QM9(amount=1000)
#print(data)

idxs = np.random.permutation(len(data))
split_va, split_te = int(0.8 * len(data)), int(0.9 * len(data))
idx_tr, idx_va, idx_te = np.split(idxs, [split_va, split_te])
data_tr = data[idx_tr]
#print(len(data_tr))
data_va = data[idx_va]
data_te = data[idx_te]
print('train size, va size, test size:')
print(data_tr, data_va, data_te)

loader_tr = DisjointLoader(data_tr, batch_size=args.batch_size, epochs=args.epochs)
loader_va = DisjointLoader(data_va, batch_size=args.batch_size)
loader_te = DisjointLoader(data_te, batch_size=args.batch_size)
#print(loader_tr, loader_va, loader_te)

optimizer = Adam(learning_rate=args.learning_rate)
if args.task=='c':
    loss_fn= SparseCategoricalCrossentropy()
elif args.task=='r':
    loss_fn = MeanSquaredError()
else:
    print(args.task, ' is not c or r.')

model= HNet()
#model.compile(optimizer, loss_fn)
epoch = step = 0
best_val_loss = np.inf
best_weights = None
results = []

init_time=time.time()-datasettime
print('model initialized, time=', str(init_time))

def train_step(inputs, target):
    with tf.GradientTape() as tape:
        #print('dot')
        predictions = model(inputs, training=True)
        print('here is t, p, and train loss')
        print(target)
        print(predictions)
        print(loss_fn(target, predictions))
        loss = loss_fn(target, predictions) + sum(model.losses)
    #val_loss = model.evaluate(loader_va.load(), steps=loader_va.steps_per_epoch)
    #print('val loss')
    #print(val_loss)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    mse = tf.reduce_mean((target-predictions)**2)
    #print(mse)
    return loss, mse

for batch in loader_tr:
    step += 1
    loss, mse = train_step(*batch)
    if step == loader_tr.steps_per_epoch:
        step = 0
        print("Loss: {}".format(loss / loader_tr.steps_per_epoch))
        loss = 0
        val_loss, val_acc = evaluate(loader_va)
        print('val loss and acc')
        print(val_loss, val_acc)

print('training time=', time.time()-init_time)

print('it worked?')
test_loss, test_acc = evaluate(loader_te)
print("Done. Test loss: {}".format(test_loss))
print('test_acc=', test_acc)
