import tensorflow as tf
import os
import sys
import argparse
from spektral_essential_objects import GaussianDistance, MyDataset, PartitionedData, HNetRecurrent, RegularizedDiffPool
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
#import ConfigSpace

import ConfigSpace
from hpbandster.optimizers.config_generators.bohb import BOHB
#except ImportError:
#    BOHB = ConfigSpace = None


parser = argparse.ArgumentParser(description='crystal hierarchy arguments.')

parser.add_argument('--datadir', dest='datadir',
        help='Directory where dataset is located', default='/Users/nilamandal/Desktop/Main_fol_Zintl')
parser.add_argument('--filename', dest='filename',
                    help='csv where data is located', default='id_prop.csv')
parser.add_argument('--num-nbrs', dest='num_nbrs', type=int,
                    help='num neighbors per atom', default=12)
parser.add_argument('--radius-angstroms', dest='radius_angstroms', type=int,
                    help='search radius for neighbors', default=10)
parser.add_argument('--task', choices=['r', 'c'],
                    default='r', help='complete a regression or classification task (default: regression)')
args = parser.parse_args(sys.argv[1:])

df = pd.read_csv(args.datadir+'/'+args.filename, names=['id','target', 'prototype'], header=None)


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

def random_split(dataset,batch_size, epochs):
    data_tr=[]
    data_va=[]
    data_te=[]
    split= int(len(dataset)/5)
    split_2= split*2
    #print(split)
    data_te=dataset[:split]
    data_va= dataset[split:split_2]
    data_tr=dataset[split_2:len(dataset)]
    #print(len(data_te))
    #print(len(data_va))
    #print(len(data_tr))
    loader_tr = DisjointLoader(PartitionedData(data_tr), batch_size=batch_size, epochs=epochs)
    loader_va = DisjointLoader(PartitionedData(data_va), batch_size=len(data_va))
    loader_te = DisjointLoader(PartitionedData(data_te), batch_size=len(data_te))
    return loader_tr, loader_va, loader_te

def train_model(config):
    #model_id = config["model_id"]
    #checkpoint_path='./'+str(config["model_id"])
    checkpoint_path='./'
    batch_size = config['batch_size']
    epochs = 10
    #embedding_size= 64
    #dr1=config['dr1']
    #dr2=config['dr2']
    entropy_lambda= config['entropy_lambda']
    column_lambda= config['column_lambda']
    lr= config['lr']

    # Import model libraries, etc...
    # Load data and train model code here...
    train_df = pd.read_csv(os.path.join(args.datadir,'train.csv'), names=['id','target'], header=None)
    train_data= MyDataset(train_df, args.datadir, args.radius_angstroms, args.num_nbrs, args.task)
    load_tr= DisjointLoader(train_data, batch_size=batch_size, epochs=epochs)


    val_df = pd.read_csv(os.path.join(args.datadir,'val.csv'), names=['id','target'], header=None)
    val_data= MyDataset(val_df, args.datadir, args.radius_angstroms, args.num_nbrs, args.task)
    load_va= DisjointLoader(val_data, batch_size=len(val_data))

    test_df = pd.read_csv(os.path.join(args.datadir,'test.csv'), names=['id','target'], header=None)
    test_data= MyDataset(test_df, args.datadir, args.radius_angstroms, args.num_nbrs, args.task)
    load_te= DisjointLoader(test_data, batch_size=len(test_data))

    #load_tr, load_va, load_te=random_split(data, batch_size, epochs)


    model= HNetRecurrent('r', 1, el=entropy_lambda, cl=column_lambda)

    optim=Adam(lr)
    loss_fn= MeanSquaredError()

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
                #loss = 0
                val_loss, val_metric = evaluate(load_va, model, loss_fn)
                if val_loss<best_val_loss:
                    model.save_weights(checkpoint_path)
                    best_val_loss= val_loss
                    early_stop_counter=0
                else:
                    early_stop_counter+=1

                epoch+=1



    # Return final stats. You can also return intermediate progress
    # using ray.air.session.report() if needed.
    # To return your model, you could write it to storage and return its
    # URI in this dict, or return it as a Tune Checkpoint:
    # https://docs.ray.io/en/latest/tune/tutorials/tune-checkpoints.html
    return {"score": best_val_loss}

if __name__ == "__main__":
      NUM_MODELS = 2

      trial_space = {
            # This is an example parameter. You could replace it with filesystem paths,
            # model types, or even full nested Python dicts of model configurations, etc.,
            # that enumerate the set of trials to run.

            'batch_size': tune.lograndint(1, 128, 2),
            #'embedding_size': tune.lograndint(1, 128, 2),
            #'dr1': tune.uniform(0, 1),
            'entropy_lambda': tune.loguniform(1e-1, 1e8),
            'column_lambda': tune.loguniform(1e-1, 1e8),
            'lr': tune.loguniform(1e-9, 1e-1)

        }

      bohb_hyperband = HyperBandForBOHB(
        time_attr="training_iteration",
        max_t=100,
        reduction_factor=4,
        stop_last_trials=False,
      )
      bohb = TuneBOHB(metric='score', mode='min')
      #print(bayesopt)
      train_model = tune.with_resources(train_model, {"cpu": 2})
      tuner = tune.Tuner(train_model, tune_config=tune.TuneConfig(
        search_alg=bohb, num_samples=NUM_MODELS), param_space=trial_space)
      results = tuner.fit()
      print(results)
