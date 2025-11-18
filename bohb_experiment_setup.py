import argparse
import sys

import tensorflow as tf
import os
from spektral_essential_objects import MyDataset, SparseEdgepool, AtomFeaDataset, NotShrinking, NoShrink_GraphSage, Graphsage_dataset
from spektral.data import DisjointLoader
from CorrectedRepeater import BOHBRepeater
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError, CategoricalCrossentropy
import numpy as np
from tensorflow.keras.metrics import categorical_accuracy #, mean_squared_error
from ray import tune
from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
from ray.tune.search.bohb import TuneBOHB
import pandas as pd
import ConfigSpace
from hpbandster.optimizers.config_generators.bohb import BOHB
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import CallbackList, CSVLogger
from tensorflow.keras import backend as K
import json
import resource

from utils import train_step, evaluate
parser = argparse.ArgumentParser(description='crystal hierarchy arguments.')
parser.add_argument('--datadir', dest='datadir',
        help='Directory where dataset is located', default='/Users/nilamandal/desktop/Main_fol_Zintl')#/home/nim18004/Main_fol_Zintl

parser.add_argument('--task', choices=['r', 'c'],
                    default='r', help='complete a regression or classification task (default: regression)')
args = parser.parse_args(sys.argv[1:])



def train_model(config):
    print('BEGUN INDIVIDUAL TRAINING')
    config['batch_size']= 32
    checkpoint_path='./goodmodel.ckpt'

    epochs = 1000
    if epochs<1000:
        print('WARNING: CURRENTLY RUNNING IN DEBUG MODE WITH '+str(epochs)+' EPOCHS')

    # Load data and train model code here...
    train = pd.read_csv(os.path.join(args.datadir,'train_by_fam_resplit.csv'))
    train= train.head(10)
    val= pd.read_csv(os.path.join(args.datadir,'val_by_fam_resplit.csv'))
    val= val.head(10)

    train_data= Graphsage_dataset(train, args.datadir, 8, 12, args.task)
    load_tr= DisjointLoader(train_data, batch_size=config['batch_size'], epochs=epochs)
    load_tr_eval= DisjointLoader(train_data, batch_size=len(train_data))
    val_data= Graphsage_dataset(val, args.datadir, 8, 12, args.task)
    load_va= DisjointLoader(val_data, batch_size=len(val_data))

    csv_log = CSVLogger("./callback_results.csv")
    model= NoShrink_GraphSage(args.task, config['embedding_size'], config['hidden_size'], config['cgcnn_num'])
    all_callbacks= CallbackList([csv_log], add_history=True, model=model)
        #
    optim=Adam(config['lr'])
    if args.task=='r':
        loss_fn= MeanSquaredError()
    else:
        loss_fn= CategoricalCrossentropy()
    train_metric=[]
    val_metric_list=[]
    early_stop_counter= 0
    patience= 50
    epoch = step = 0

    best_val_loss = np.inf
    best_model_mse = np.inf
    logs = {}
    all_callbacks.on_train_begin(logs=logs)

    for batch in load_tr:
            #print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            # peak memory usage (kilobytes on Linux, bytes on OS X)
            if step==0:
                all_callbacks.on_epoch_begin(epoch, logs=logs)
            step += 1

            all_callbacks.on_train_batch_begin(step)
            loss, metric = train_step(*batch, model, loss_fn, optim)
            all_callbacks.on_train_batch_end(step, logs)

            if tf.math.is_nan(loss):
                all_callbacks.on_train_end(logs)
                if epoch>1:
                    gen_plots(train_metric, val_metric_list)
                return np.inf

            if step == load_tr.steps_per_epoch:
                step = 0
                tr_loss, tr_rmse, tr_mae= evaluate(load_tr_eval, model, loss_fn)
                val_loss, val_rmse, val_mae= evaluate(load_va, model, loss_fn)
                print(tr_loss)
                val_metric_list.append(val_loss)
                train_metric.append(tr_loss)
                total_val_loss= val_loss

                if epoch>0:
                    if total_val_loss<best_val_loss:
                        early_stop_counter=0
                        model.save_weights(checkpoint_path)
                        best_val_loss= val_loss
                    else:
                        early_stop_counter+=1
                if args.task=='r':
                    all_callbacks.on_epoch_end(epoch, {'train_mse':tr_loss, 'train_mae':tr_mae, 'val_mse':val_loss, 'val_mae':val_mae})
                else:
                    all_callbacks.on_epoch_end(epoch, {'train_loss':tr_loss, 'val_loss:':total_val_loss})
                if early_stop_counter==patience:
                    all_callbacks.on_train_end(logs)
                    gen_plots(train_metric, val_metric_list)
                    return {"score": best_val_loss}
                elif np.isnan(tr_loss):
                    return {"score": np.inf}
                else:
                    epoch+=1
    all_callbacks.on_train_end(logs)
    gen_plots(train_metric, val_metric_list)


    return {"score": best_val_loss}

def gen_plots(train_metric, val_metric):
    plt.switch_backend('Agg')

    plt.figure()

    epochs=list(range(len(train_metric)))
    min_train= 'train min='+str(np.round(np.min(train_metric), decimals=3))+','
    min_val= 'val min='+str(np.round(np.min(val_metric), decimals=3))
    plt.plot(epochs, np.log(train_metric), label='training loss')
    plt.plot(epochs, np.log(val_metric), label='val loss')

    figtitle='./result.png'

    plt.xlabel('epochs')
    plt.legend()
    plt.ylabel('log of mean square error ')
    plt.savefig(figtitle)


if __name__ == "__main__":
      NUM_MODELS = 2

      trial_space = {
            'embedding_size': tune.choice([4,8,16,32,64]),
            'hidden_size': tune.choice([4,8,16,32,64]),
            'cgcnn_num': tune.choice([1,2,3]),
            #'softmax_beta': tune.loguniform(1, 1e8),
            'lr': tune.loguniform(1e-6, 1e-2)
        }

      bohb_hyperband = HyperBandForBOHB(
        time_attr="training_iteration",
        max_t=81,
        reduction_factor=3,
        stop_last_trials=False,
      )

      bohb = BOHBRepeater(metric='score', mode='min', repeat=1, max_concurrent=1)
      train_model_object = tune.with_resources(train_model, {"cpu": 1})
      tuner = tune.Tuner(train_model_object, tune_config=tune.TuneConfig(
        search_alg=bohb,
        scheduler=bohb_hyperband,
        metric='score',
        mode='min',
        num_samples=NUM_MODELS), param_space=trial_space)
      print('CREATED all TUNING OBJECTS')
      results = tuner.fit()
      print(results)
