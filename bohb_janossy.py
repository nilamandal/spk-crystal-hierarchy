import tensorflow as tf
import os
import sys
import argparse
from spektral_essential_objects import GaussianDistance, MyDataset, RegularizedDiffPool, HNetDoubleJanossy
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
from tensorflow.keras.callbacks import CallbackList, CSVLogger
from tensorflow.keras import backend as K

parser = argparse.ArgumentParser(description='crystal hierarchy arguments.')

parser.add_argument('--datadir', dest='datadir',
        help='Directory where dataset is located', default='/home/nim18004/Main_fol_Zintl')
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
def entropy_loss(s):
    entr = tf.negative(
        tf.reduce_sum(tf.multiply(s, K.log(s + K.epsilon())), axis=-1)
    )
    entr_loss = tf.reduce_sum(entr, axis=-1)
    return entr_loss


def both_entropy(s, i):
    batch_size= s.shape[0]
    c_stack=[]
    row_entropy_sum=0
    for g in range(batch_size):
        count= np.count_nonzero(i==g)
        s_g=s[g,:count]
        #---
        row= entropy_loss(s_g)
        row_entropy_sum+=row
        #---
        column_means=tf.divide(tf.reduce_sum(s_g, axis=0),s_g.shape[0])
        c_stack.append(column_means)


    c_stack= tf.stack(c_stack, axis=0) #this should give shape(num graphs, k)
    column_entropy = tf.reduce_sum(tf.reduce_sum(tf.multiply(c_stack, K.log(c_stack + K.epsilon())), axis=-1), axis=-1)
    #print(column_entropy, row_entropy_sum)
    #print('--')
    return column_entropy, row_entropy_sum

def evaluate(loader, model, loss_fn, test=False):
    step = 0
    output=[]
    while step < loader.steps_per_epoch:
        step += 1
        inputs, target = loader.__next__()
        x, a, e, i = inputs
        pred, s = model(inputs, training=False)
        #print(s)
        c_e, r_e= both_entropy(s, i)
        if args.task=='c':
            outs = (
                loss_fn(target, pred),
                tf.reduce_mean(sparse_categorical_accuracy(target, pred)),
                len(target),  # Keep track of batch size
            )
        elif args.task=='r':
            mse = tf.reduce_mean((target-pred)**2)
            rmse= np.sqrt(mse)
            mae= tf.reduce_mean(np.abs(target-pred))
            outs = (
                loss_fn(target, pred),
                mse,
                rmse,
                mae,
                c_e,
                r_e,
                len(target),  # Keep track of batch size
            )
            #print('LOOK AT ME'+str(len(target)))
        output.append(outs)
        if step == loader.steps_per_epoch:
            output = np.array(output)
            #print(output.shape)
            return np.average(output[:, :-1], 0, weights=output[:, -1])

def train_step(inputs, target, model, loss_fn, optimizer):
    with tf.GradientTape() as tape:
        predictions, s = model(inputs, training=True)
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
    epochs = 2

    embedding_size= config['embedding_size']
    batch_size= config['batch_size']
    dr1= config['dr1']
    fc_size= config['fc_size']
    fc_num= config['fc_num']
    fc_size2= config['fc_size2']
    fc_num2= config['fc_num2']
    entropy_lambda= config['entropy_lambda']
    column_lambda= config['column_lambda']
    lr= config['lr']

    # Load data and train model code here...
    train_df = pd.read_csv(os.path.join(args.datadir,'train_with_counts_complete.csv'))
    train_df = train_df[train_df['num_elements']<=3]
    #train_df= train_df.head(10)
    train_data= MyDataset(train_df, args.datadir, args.radius_angstroms, args.num_nbrs, args.task)
    load_tr= DisjointLoader(train_data, batch_size=batch_size, epochs=epochs)
    load_tr_eval= DisjointLoader(train_data, batch_size=batch_size, epochs=epochs)

    val_df = pd.read_csv(os.path.join(args.datadir,'val_with_counts_complete.csv'))
    val_df = val_df[val_df['num_elements']<=3]
    #val_df= val_df.head(10)
    val_data= MyDataset(val_df, args.datadir, args.radius_angstroms, args.num_nbrs, args.task)
    load_va= DisjointLoader(val_data, batch_size=len(val_data))

    csv_log = CSVLogger("./callback_results.csv")


    model= HNetDoubleJanossy('r', 1, embedding_size=embedding_size, d1=dr1, el=entropy_lambda, cl=column_lambda, fc_num=fc_num, fc_size=fc_size, fc_size2=fc_size2, fc_num2=fc_num2, return_s=True)
    all_callbacks= CallbackList([csv_log], add_history=True, model=model)


    optim=Adam(lr)
    loss_fn= MeanSquaredError()

    train_metric=[]
    val_metric_list=[]
    early_stop_counter= 0
    epoch = step = 0
    best_val_loss = np.inf
    best_weights = None
    results = []
    logs = {}
    all_callbacks.on_train_begin(logs=logs)
    for batch in load_tr:
            if step==0:
                all_callbacks.on_epoch_begin(epoch, logs=logs)
            step += 1

            all_callbacks.on_train_batch_begin(step)
            loss, metric = train_step(*batch, model, loss_fn, optim)
            all_callbacks.on_train_batch_end(step, logs)

            if step == load_tr.steps_per_epoch:
                step = 0
                loss_str="Loss: {}".format(loss / load_tr.steps_per_epoch)
                is_nan= np.isnan(loss)
                tr_loss, tr_mse, tr_rmse, tr_mae, tr_ce, tr_re= evaluate(load_tr_eval, model, loss_fn)
                val_loss, val_mse, val_rmse, val_mae, val_ce, val_re = evaluate(load_va, model, loss_fn)
                val_metric_list.append(val_loss)
                train_metric.append(tr_loss)
                if val_loss<best_val_loss:
                    model.save_weights(checkpoint_path)
                    best_val_loss= val_loss
                    early_stop_counter=0
                else:
                    early_stop_counter+=1

                all_callbacks.on_epoch_end(epoch, {'train_mse':tr_mse, 'train_rmse':tr_rmse, 'train_mae':tr_mae, 'val_mse':val_mse, 'val_rmse:':val_rmse, 'val_mae':val_mae, 'train_row_entropy':tr_re, 'train_n_column_entropy':tr_ce, 'val_row_entropy':val_re, 'val_n_column_entropy':val_ce})

                epoch+=1
    all_callbacks.on_train_end(logs)
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
      sys.stdout = open('./debug_additional_logs.txt', 'w')

      trial_space = {
            'embedding_size': tune.choice([4,8,16,32,64,128]),
            'batch_size': tune.choice([1,2,4,8,16,32,64]),
            'dr1': tune.uniform(0, 1),
            'entropy_lambda': tune.loguniform(1e-1, 1e8),
            'column_lambda': tune.loguniform(1e-1, 1e8),
            'fc_size': tune.choice([4,8,16,32,64]),
            'fc_num': tune.choice([1,2,3]),
            'fc_size2': tune.choice([4,8,16,32,64]),
            'fc_num2': tune.choice([1,2,3]),
            'lr': tune.loguniform(1e-5, 1e-1)
        }

      bohb_hyperband = HyperBandForBOHB(
        time_attr="training_iteration",
        max_t=100,
        reduction_factor=4,
        stop_last_trials=False,
      )
      bohb = TuneBOHB(metric='score', mode='min')
      bohb = tune.search.ConcurrencyLimiter(bohb, max_concurrent=15)

      train_model = tune.with_resources(train_model, {"cpu": 1})
      tuner = tune.Tuner(train_model, tune_config=tune.TuneConfig(
        search_alg=bohb, scheduler=bohb_hyperband, metric='score', mode='min', num_samples=NUM_MODELS), param_space=trial_space)
      results = tuner.fit()
      print(results)
