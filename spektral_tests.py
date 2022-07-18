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
#from spektral.datasets import QM9
begin_time = time.time()
parser = argparse.ArgumentParser(description='crystal hierarchy arguments.')

parser.add_argument('--datadir', dest='datadir',
        help='Directory where dataset is located', default='../crystalhierarchydata/icsd-zintl-search')

parser.add_argument('--filename', dest='filename',
                    help='csv where data is located', default='corrected_sym.csv')
parser.add_argument('--file-out', dest='file_out',
                    help='output file name', default='sysout')
parser.add_argument('--path-out', dest='path',
                    help='output path', default='./spektraltest_8atom/')
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

args = parser.parse_args(sys.argv[1:])

np.random.seed(args.random_seed)
#path = './spektraltest_8atom/'
if not os.path.exists(args.path+'/'+args.file_out):
    os.makedirs(args.path+'/'+args.file_out)
sys.stdout = open(args.path+'/'+args.file_out+'/'+args.file_out+'.txt', 'w')

print(args)

def evaluate(loader):
    output = []
    step = 0
    while step < loader.steps_per_epoch:
        step += 1
        inputs, target = loader.__next__()
        pred = model(inputs, training=False)

        outs = (
            loss_fn(target, pred),
            tf.reduce_mean(sparse_categorical_accuracy(target, pred)),
            len(target),  # Keep track of batch size
        )

        output.append(outs)
        if step == loader.steps_per_epoch:
            output = np.array(output)
            return np.average(output[:, :-1], 0, weights=output[:, -1])

data= MyDataset(args.datadir,args.filename, args.radius_angstroms, args.num_atoms, args.num_nbrs, args.task)
datasettime=time.time()-begin_time
print('datset generated: time=', str(datasettime))
#data = QM9(amount=1000)

idxs = np.random.permutation(len(data))
split_va, split_te = int(0.8 * len(data)), int(0.9 * len(data))
idx_tr, idx_va, idx_te = np.split(idxs, [split_va, split_te])
data_tr = data[idx_tr]
data_va = data[idx_va]
data_te = data[idx_te]
print('train size, va size, test size:')
print(data_tr, data_va, data_te)

loader_tr = DisjointLoader(data_tr, batch_size=args.batch_size, epochs=args.epochs)
loader_va = DisjointLoader(data_va, batch_size=args.batch_size)
loader_te = DisjointLoader(data_te, batch_size=args.batch_size)

optimizer = Adam(learning_rate=args.learning_rate)
if args.task=='c':
    loss_fn= SparseCategoricalCrossentropy()
elif args.task=='r':
    loss_fn = MeanSquaredError()
else:
    print(args.task, ' is not c or r.')

model= HNet(args.task, args.num_classes)
#model.compile(optimizer, loss_fn)
epoch = step = 0
best_val_loss = np.inf
best_weights = None
results = []

init_time=time.time()-datasettime
print('model initialized, time=', str(init_time))

def train_step(inputs, target):
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        print('here is t, p, and train loss')
        #print(target)
        #print(predictions)
        print(loss_fn(target, predictions))
        loss = loss_fn(target, predictions) + sum(model.losses)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    mse = tf.reduce_mean((target-predictions)**2)
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

checkpoint_path = args.path+"/"+args.file_out+"/"+args.file_out+".ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
model.save_weights(checkpoint_path.format(epoch=args.epochs))
