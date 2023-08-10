import tensorflow as tf
import os
import sys
import argparse
from spektral_essential_objects import GaussianDistance, MyDataset, RegularizedDiffPool, HNetConcatJanossy
from spektral.data import DisjointLoader
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.losses import MeanSquaredError
import numpy as np
from tensorflow.keras.metrics import sparse_categorical_accuracy, mean_squared_error
from ray import tune
from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
from ray.tune.search.bohb import TuneBOHB
import pandas as pd
import ConfigSpace
from hpbandster.optimizers.config_generators.bohb import BOHB
import matplotlib.pyplot as plt



parser = argparse.ArgumentParser(description='crystal hierarchy arguments.')

parser.add_argument('--datadir', dest='datadir',
        help='Directory where dataset is located', default='/Users/nilamandal/desktop/Main_fol_Zintl')
parser.add_argument('--filename', dest='filename',
                    help='csv where data is located', default='id_prop.csv')
parser.add_argument('--num-nbrs', dest='num_nbrs', type=int,
                    help='num neighbors per atom', default=12)
parser.add_argument('--radius-angstroms', dest='radius_angstroms', type=int,
                    help='search radius for neighbors', default=10)
parser.add_argument('--task', choices=['r', 'c'],
                    default='r', help='complete a regression or classification task (default: regression)')
args = parser.parse_args(sys.argv[1:])

#df = pd.read_csv(args.datadir+'/'+args.filename, names=['id','target', 'prototype'], header=None)


def evaluate(loader, model, loss_fn, test=False):
    step = 0
    output=[]
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

def train_model(config):
    checkpoint_path='./goodmodel.ckpt'
    epochs = 1000

    embedding_size= config['embedding_size']
    batch_size= config['batch_size']
    dr1= config['dr1']
    fc_size= config['fc_size']
    fc_num= config['fc_num']
    entropy_lambda= config['entropy_lambda']
    column_lambda= config['column_lambda']
    lr= config['lr']

    # Load data and train model code here...
    train_df = pd.read_csv(os.path.join(args.datadir,'train.csv'), names=['id','target'], header=None)
    train_data= MyDataset(train_df, args.datadir, args.radius_angstroms, args.num_nbrs, args.task)
    load_tr= DisjointLoader(train_data, batch_size=batch_size, epochs=epochs)

    val_df = pd.read_csv(os.path.join(args.datadir,'val.csv'), names=['id','target'], header=None)
    val_data= MyDataset(val_df, args.datadir, args.radius_angstroms, args.num_nbrs, args.task)
    load_va= DisjointLoader(val_data, batch_size=len(val_data))

    model= HNetConcatJanossy('r', 1, embedding_size=embedding_size, d1=dr1, el=entropy_lambda, cl=column_lambda, fc_num=fc_num, fc_size=fc_size)

    optim=Adam(lr)
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


            if step == load_tr.steps_per_epoch:
                step = 0
                loss_str="Loss: {}".format(loss / load_tr.steps_per_epoch)
                is_nan= np.isnan(loss)
                val_loss, val_metric = evaluate(load_va, model, loss_fn)
                val_metric_list.append(val_metric)
                train_metric.append(metric)
                if val_loss<best_val_loss:
                    model.save_weights(checkpoint_path)
                    best_val_loss= val_loss
                    early_stop_counter=0
                else:
                    early_stop_counter+=1

                epoch+=1
    gen_plots(train_metric, val_metric_list)
    # Return final stats. You can also return intermediate progress
    # using ray.air.session.report() if needed.
    # To return your model, you could write it to storage and return its
    # URI in this dict, or return it as a Tune Checkpoint:
    # https://docs.ray.io/en/latest/tune/tutorials/tune-checkpoints.html
    return {"score": best_val_loss}


def gen_plots(train_metric, val_metric):
    plt.switch_backend('Agg')
    print('plots pls')
    plt.figure()

    epochs=list(range(len(train_metric)))
    min_train= 'train min='+str(np.round(np.min(train_metric), decimals=3))+','
    min_val= 'val min='+str(np.round(np.min(val_metric), decimals=3))
    plt.plot(epochs, np.log(train_metric), label='training loss')
    plt.plot(epochs, np.log(val_metric), label='val loss')

    figtitle='./result.png'

    #plt.title(titleline, wrap=True)
    plt.xlabel('epochs')
    plt.legend()
    plt.ylabel('log of mean square error ')
    plt.savefig(figtitle)


if __name__ == "__main__":
      NUM_MODELS = 200
      #sys.stdout = open('./janossy_te_morelayers.txt', 'w')

      trial_space = {
            'embedding_size': tune.choice([4,8,16,32,64,128]),
            'batch_size': tune.choice([1,2,4,8,16,32,64]),
            'dr1': tune.uniform(0, 1),
            #'dr2': tune.uniform(0, 1),
            'fc_size': tune.choice([4,8,16,32,64]),
            'fc_num': tune.choice([1,2,3,4]),
            'entropy_lambda': tune.loguniform(1e-1, 1e8),
            'column_lambda': tune.loguniform(1e-1, 1e8),
            'lr': tune.loguniform(1e-5, 1e-1)
        }

      bohb_hyperband = HyperBandForBOHB(
        time_attr="training_iteration",
        max_t=5,
        reduction_factor=4,
        stop_last_trials=False,
      )
      bohb = TuneBOHB(metric='score', mode='min')
      #print(bayesopt)
      train_model = tune.with_resources(train_model, {"cpu": 20})
      tuner = tune.Tuner(train_model, tune_config=tune.TuneConfig(
        search_alg=bohb, scheduler=bohb_hyperband, metric='score', mode='min', num_samples=NUM_MODELS), param_space=trial_space)
      results = tuner.fit()
      print(results)
