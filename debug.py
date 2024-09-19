import tensorflow as tf
import os
import sys
import argparse
from spektral_essential_objects import GaussianDistance, MyDataset, RegularizedDiffPool, HNetEdgepool, HNetDoubleJanossy, HNetDebugOnly
from spektral.data import DisjointLoader
#from CorrectedRepeater import BOHBRepeater
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.losses import MeanSquaredError
import numpy as np
from tensorflow.keras.metrics import sparse_categorical_accuracy #, mean_squared_error
#from ray import tune
#from ray.tune.search.bayesopt import BayesOptSearch
#from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
#from ray.tune.search.bohb import TuneBOHB
import pandas as pd
import ConfigSpace
#from hpbandster.optimizers.config_generators.bohb import BOHB
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import CallbackList, CSVLogger
from tensorflow.keras import backend as K
import json

parser = argparse.ArgumentParser(description='crystal hierarchy arguments.')
parser.add_argument('--datadir', dest='datadir',
        help='Directory where dataset is located', default='/Users/nilamandal/desktop/Main_fol_Zintl')
parser.add_argument('--num-nbrs', dest='num_nbrs', type=int,
                    help='num neighbors per atom', default=12)
parser.add_argument('--radius-angstroms', dest='radius_angstroms', type=int,
                    help='search radius for neighbors', default=10)
parser.add_argument('--task', choices=['r', 'c'],
                    default='r', help='complete a regression or classification task (default: regression)')
args = parser.parse_args(sys.argv[1:])

def train_step(inputs, target, model, loss_fn, optimizer):
    #print(inputs)
    #print(target)
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

def evaluate(loader, model, loss_fn, test=False):
    step = 0
    mse=[]
    rmse=[]
    mae=[]
    while step < loader.steps_per_epoch:
        step += 1
        inputs, target = loader.__next__()
        x, a, e, i = inputs
        pred= model(inputs, training=False)


        if args.task=='r':
            mse.append(tf.reduce_mean((target-pred)**2))
            rmse.append(np.sqrt(tf.reduce_mean((target-pred)**2)))
            mae.append(tf.reduce_mean(np.abs(target-pred)))

        if step == loader.steps_per_epoch:
            return np.average(mse), np.average(rmse), np.average(mae)

checkpoint_path='./debug/goodmodel.ckpt'
epochs = 10
embedding_size= 16
batch_size= 16
entropy_lambda= 16
softmax_beta= 10
lr= 0.001

train_df = pd.read_csv(os.path.join(args.datadir,'train_no_metals.csv'))
train_df = train_df.head(20)
val_df = pd.read_csv(os.path.join(args.datadir,'val_no_metals.csv'))
val_df = val_df.head(10)

train_data= MyDataset(train_df, args.datadir, args.radius_angstroms, args.num_nbrs, args.task)
load_tr= DisjointLoader(train_data, batch_size=batch_size, epochs=epochs)

load_tr_copy= DisjointLoader(train_data, batch_size=len(train_data))
val_data= MyDataset(val_df, args.datadir, args.radius_angstroms, args.num_nbrs, args.task)
load_va= DisjointLoader(val_data, batch_size=len(val_data))

model= HNetDebugOnly('r', 1, beta= softmax_beta, return_s=True)
csv_log = CSVLogger("./debug/callback_results.csv")
all_callbacks= CallbackList([csv_log], add_history=True, model=model)
#
optim=Adam(lr)
loss_fn= MeanSquaredError()

train_metric=[]
val_metric_list=[]
early_stop_counter= 0
patience= 100
epoch = step = 0

best_val_loss = np.inf
best_model_mse = np.inf
logs = {}
all_callbacks.on_train_begin(logs=logs)
for batch in load_tr:
        if step==0:
            all_callbacks.on_epoch_begin(epoch, logs=logs)
        step += 1
        print(epoch, step)
        all_callbacks.on_train_batch_begin(step)
        loss, metric = train_step(*batch, model, loss_fn, optim)
        all_callbacks.on_train_batch_end(step, logs)

        if step == load_tr.steps_per_epoch:
            step = 0
            loss_str="Loss: {}".format(loss / load_tr.steps_per_epoch)

            tr_mse, tr_rmse, tr_mae= evaluate(load_tr_copy, model, loss_fn)
            val_mse, val_rmse, val_mae = evaluate(load_va, model, loss_fn)
            val_metric_list.append(val_mse)
            train_metric.append(tr_mse)

            if epoch>0:
                if val_mse<best_val_loss:
                    early_stop_counter=0
                    model.save_weights(checkpoint_path)
                    best_val_loss= val_mse

                else:
                    early_stop_counter+=1
            if epoch==epochs:
                all_callbacks.on_train_end(logs)
            else:
                all_callbacks.on_epoch_end(epoch, {'train_mse':tr_mse, 'train_rmse':tr_rmse, 'train_mae':tr_mae, 'val_mse':val_mse, 'val_rmse:':val_rmse, 'val_mae':val_mae})
                epoch+=1
    #        if early_stop_counter==patience:
    #            all_callbacks.on_train_end(logs)

print('nothing has gone wrong yet')
