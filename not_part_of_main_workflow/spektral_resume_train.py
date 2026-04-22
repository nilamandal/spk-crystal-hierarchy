from spektral.data import Graph, Dataset, DisjointLoader
from spektral.data.utils import to_batch
from spektral.layers import CrystalConv, DiffPool, ops, GlobalMaxPool#, Disjoint2Batch
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import MeanSquaredError, SparseCategoricalCrossentropy
from tensorflow.keras.metrics import sparse_categorical_accuracy, mean_squared_error
import numpy as np
import pandas as pd
import os
import sys
from pymatgen.core.structure import Structure
import json
import argparse
import time
from spektral_essential_objects import GaussianDistance,MyDataset,HNetSimple, HNetConcat, PartitionedData, HNetMultifilter
import matplotlib.pyplot as plt


begin_time = time.time()
parser = argparse.ArgumentParser(description='crystal hierarchy arguments.')

parser.add_argument('--datadir', dest='datadir',
        help='Directory where dataset is located', default='../Main_fol_Zintl/')

parser.add_argument('--filename', dest='filename',
                    help='csv where data is located', default='id_prop.csv')
parser.add_argument('--file-out', dest='file_out',
                    help='output txt file name', default='out.txt')
parser.add_argument('--path-out', dest='path_out',
                   help='output path', default='./longmodel_b59393f5/')
parser.add_argument('--num-atoms', dest='num_atoms', type=int,
                    help='Maximum number of nodes', default=10)
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

parser.add_argument('--task', choices=['r', 'c'],
                    default='r', help='complete a regression or '
                        'classification task (default: regression)')

def evaluate(loader, model, loss_fn, test=False):
    output = []
    step = 0
    while step < loader.steps_per_epoch:
        step += 1
        inputs, target = loader.__next__()
        pred = model(inputs, training=False)
        if args.task=='c':
            outs = (
                loss_fn(target, pred),
                tf.reduce_mean(sparse_categorical_accuracy(target, pred)),
                len(target),  # Keep track of batch size
            )
        elif args.task=='r':
            outs = (
                loss_fn(target, pred),
                tf.reduce_mean(mean_squared_error(target, pred)),
                len(target),  # Keep track of batch size
            )
        output.append(outs)
        if step == loader.steps_per_epoch:
            if test==True and args.task=='c':
                print('TEST confusion_matrix')
                print(confusion_matrix(target,np.argmax(pred, axis=1)))
            output = np.array(output)
            return np.average(output[:, :-1], 0, weights=output[:, -1])

def train_step(inputs, target, model, loss_fn, optimizer):
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        loss = loss_fn(target, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    if args.task=='r':
        mse = tf.reduce_mean((target-predictions)**2)
        return loss, mse
    if args.task=='c':
        sca= tf.reduce_mean(sparse_categorical_accuracy(target, predictions))
        return loss, sca

def gen_plots(train_metric, val_metric):
    print('plots pls')
    plt.figure()

    epochs=list(range(len(train_metric)))
    min_train= 'train min='+str(np.round(np.min(train_metric), decimals=3))+','
    min_val= 'val min='+str(np.round(np.min(val_metric), decimals=3))
    plt.plot(epochs, np.log(train_metric), label='training loss')
    plt.plot(epochs, np.log(val_metric), label='val loss')

    figtitle=args.path_out+'result.png'

    #plt.title(titleline, wrap=True)
    plt.xlabel('epochs')
    plt.legend()
    plt.ylabel('log of mean square error ')
    plt.savefig(figtitle)

checkpoint_path = "../corrected_concat_results/train_model_b59393f5_best/goodmodel.ckpt.index"

checkpoint_dir = os.path.dirname(checkpoint_path)

args = parser.parse_args(sys.argv[1:])
if not os.path.exists(args.path_out):
    os.makedirs(args.path_out)
sys.stdout = open(args.path_out+args.file_out, 'w')

val_df = pd.read_csv(os.path.join(args.datadir,'val.csv'), names=['id','target'], header=0)
data= MyDataset(val_df, args.datadir, args.radius_angstroms, args.num_nbrs, args.task)
load_va= DisjointLoader(data, shuffle=False, batch_size=len(val_df))

df = pd.read_csv(os.path.join(args.datadir,'train.csv'), names=['id','target'], header=0)
load_tr= DisjointLoader(MyDataset(df, args.datadir, args.radius_angstroms, args.num_nbrs, args.task), batch_size=8, epochs=5000)

paramsdict= json.load(open(checkpoint_dir+'/params.json'))

model= HNetConcat('r', 1, el=paramsdict['entropy_lambda'], cl=paramsdict['column_lambda'])
latest = tf.train.latest_checkpoint(checkpoint_dir)
model.load_weights(latest)
print(model)

new_checkpoint_path=args.path_out+'/goodmodel.ckpt'
optim=Adam(paramsdict['lr'])
loss_fn= MeanSquaredError()

train_metric=[]
val_metric_list=[]
early_stop_counter= 0
epoch = step = 0
best_val_loss = np.inf
best_weights = None
results = []
for batch in load_tr:
        step += 1
        loss, metric = train_step(*batch, model, loss_fn, optim)
        #train_metric.append(metric)

        if step == load_tr.steps_per_epoch:
            step = 0
            loss_str="Loss: {}".format(loss / load_tr.steps_per_epoch)
            is_nan= np.isnan(loss)
            val_loss, val_metric = evaluate(load_va, model, loss_fn)
            val_metric_list.append(val_metric)
            train_metric.append(metric)
            print('train mse:', flush=True)
            print(loss_str, flush=True)
            print('val mse:', flush=True)
            print(val_loss, flush=True)
            if val_loss<best_val_loss:
                model.save_weights(new_checkpoint_path)
                best_val_loss= val_loss
                early_stop_counter=0
            else:
                early_stop_counter+=1

            epoch+=1
gen_plots(train_metric, val_metric_list)
