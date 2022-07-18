from spektral.data import Graph, Dataset, DisjointLoader
from spektral.data.utils import to_batch
from spektral.layers import CrystalConv, DiffPool, ops, GlobalMaxPool#, Disjoint2Batch
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
from spektral_essential_objects import AtomInitializer, GaussianDistance,AtomCustomJSONInitializer,MyDataset,HNet

begin_time = time.time()
parser = argparse.ArgumentParser(description='crystal hierarchy arguments.')

parser.add_argument('--datadir', dest='datadir',
        help='Directory where dataset is located', default='../crystalhierarchydata/icsd-zintl-search')

parser.add_argument('--filename', dest='filename',
                    help='csv where data is located', default='id_prop.csv')
parser.add_argument('--file-out', dest='file_out',
                    help='output txt file name', default='sysout.txt')
parser.add_argument('--path-out', dest='path',
                    help='output path', default='./spektraltest_8atom/')
parser.add_argument('--num-atoms', dest='num_atoms', type=int,
                    help='Maximum number of nodes', default=8)
parser.add_argument('--num-nbrs', dest='num_nbrs', type=int,
                    help='num neighbors per atom', default=12)

parser.add_argument('--num-classes', dest='num_classes', type=int,
                    help='Number of label classes', default=4)

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

def evaluate(loader, model):
    output = []
    step = 0
    all_s=[]
    while step < loader.steps_per_epoch:
        step += 1
        inputs, target = loader.__next__()
        print('hello???')
        #print(inputs, target)
        pred, s_tensor = model(inputs, training=False)
        all_s.append(s_tensor)

        outs = (tf.reduce_mean(sparse_categorical_accuracy(target, pred)),
            len(target),  # Keep track of batch size
        )

        output.append(outs)
        if step == loader.steps_per_epoch:
            output = np.array(output)
            #print(np.average(output[:, :-1], 0, weights=output[:, -1]))
            return np.average(output[:, :-1], 0, weights=output[:, -1]), all_s

checkpoint_path = "../results-18/spk8/lr1e-3/lr1e-3.cpkt"
checkpoint_dir = os.path.dirname(checkpoint_path)

args = parser.parse_args(sys.argv[1:])


data = MyDataset(args.datadir,args.filename, args.radius_angstroms, args.num_atoms, args.num_nbrs, args.task)
cifs=data.get_cifs()
#print(data, cifs)
datasettime=time.time()-begin_time
#print('datset generated: time=', str(datasettime))
#data = QM9(amount=1000)
#print(data)

idxs = np.random.permutation(len(data))
split_va, split_te = int(0.8 * len(data)), int(0.9 * len(data))
idx_tr, idx_va, idx_te = np.split(idxs, [split_va, split_te])
#print(idx_tr)
data_tr = data[idx_tr]
cifs_tr = cifs[list(idx_tr)]
data_va = data[idx_va]
cifs_va = cifs[idx_va]
data_te = data[idx_te]
cifs_te = cifs[idx_te]
print('train size, va size, test size:')
print(len(cifs_tr), len(cifs_va), len(cifs_te))

loader_tr = DisjointLoader(data_tr, batch_size=args.batch_size)
loader_va = DisjointLoader(data_va, batch_size=args.batch_size)
loader_te = DisjointLoader(data_te, batch_size=args.batch_size)

model= HNet(args.task, args.num_classes, return_s=True)

latest = tf.train.latest_checkpoint(checkpoint_dir)
model.load_weights(latest)

result, s_tensors=evaluate(loader_tr,model)
#print(s_tensors)
i=0
for j in s_tensors:
    #print(j)
    for k in j:
        print(cifs_tr[i])
        print(k)
        i+=1
        print('...')
print('ok')
