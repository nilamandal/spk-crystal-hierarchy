import tensorflow as tf
import os
import sys
import argparse
from spektral_essential_objects import GaussianDistance, MyDataset, RegularizedDiffPool, HNetDoubleJanossy
from spektral.data import DisjointLoader
from CorrectedRepeater import BOHBRepeater
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
import json

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



def entropy_loss(s):
    entr = tf.negative(
        tf.reduce_sum(tf.multiply(s, tf.math.log(s + 10**-30)), axis=-1)
    )
    entr_loss = tf.reduce_mean(entr, axis=-1)
    return entr_loss

def row_e_and_column_p(s, i):
    batch_size= s.shape[0]
    column_prod_sum=0
    row_entropy_sum=0

    for g in range(batch_size):
        count= np.count_nonzero(i==g)
        s_g=s[g,:count]
        #---
        row= entropy_loss(s_g)
        row_entropy_sum+=row

        column_product= tf.math.reduce_prod(tf.divide(tf.reduce_sum(s_g, axis=0),s_g.shape[0]))
        column_prod_sum+= column_product

    return -1*column_prod_sum, row_entropy_sum


def evaluate(loader, model, loss_fn, test=False):
    step = 0
    output=[]
    while step < loader.steps_per_epoch:
        step += 1
        inputs, target = loader.__next__()
        x, a, e, i = inputs
        pred, s = model(inputs, training=False)

        c_p, r_e= row_e_and_column_p(s, i)
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
                c_p,
                r_e,
                len(target),  # Keep track of batch size
            )
        output.append(outs)
        if step == loader.steps_per_epoch:
            output = np.array(output)
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
    if os.path.isfile('./result.png'):
        resultdict= json.load(open('./result.json'))
        return {"score": resultdict["score"]}
    checkpoint_path='./goodmodel.ckpt'

    epochs = 10

    embedding_size= config['embedding_size']
    batch_size= config['batch_size']
    #dr1= config['dr1']
    entropy_lambda= config['entropy_lambda']
    column_lambda= config['column_lambda']

    fc_size= config['fc_size']
    fc_num= config['fc_num']
    fc_size2= config['fc_size2']
    fc_num2= config['fc_num2']

    lr= config['lr']

    # Load data and train model code here...
    train_df = pd.read_csv(os.path.join(args.datadir,'train_no_metals.csv'))
    train_df = train_df[train_df['num_elements']<=3]
    #train_df= train_df.head(10)
    train_data= MyDataset(train_df, args.datadir, args.radius_angstroms, args.num_nbrs, args.task)
    load_tr= DisjointLoader(train_data, batch_size=batch_size, epochs=epochs)
    load_tr_eval= DisjointLoader(train_data, batch_size=len(train_data))

    val_df = pd.read_csv(os.path.join(args.datadir,'val_no_metals.csv'))
    val_df = val_df[val_df['num_elements']<=3]
    #val_df= val_df.head(10)
    val_data= MyDataset(val_df, args.datadir, args.radius_angstroms, args.num_nbrs, args.task)
    load_va= DisjointLoader(val_data, batch_size=len(val_data))

    csv_log = CSVLogger("./callback_results.csv")


    model= HNetDoubleJanossy('r', 1, embedding_size=embedding_size, el=entropy_lambda, cl=column_lambda, fc_num=fc_num, fc_size=fc_size, fc_size2=fc_size2, fc_num2=fc_num2, return_s=True)
    all_callbacks= CallbackList([csv_log], add_history=True, model=model)

    optim=Adam(lr)
    loss_fn= MeanSquaredError()

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
            if step==0:
                all_callbacks.on_epoch_begin(epoch, logs=logs)
            step += 1

            all_callbacks.on_train_batch_begin(step)
            loss, metric = train_step(*batch, model, loss_fn, optim)
            all_callbacks.on_train_batch_end(step, logs)

            if step == load_tr.steps_per_epoch:
                step = 0
                loss_str="Loss: {}".format(loss / load_tr.steps_per_epoch)

                tr_loss, tr_mse, tr_rmse, tr_mae, tr_ce, tr_re= evaluate(load_tr_eval, model, loss_fn)
                val_loss, val_mse, val_rmse, val_mae, val_ce, val_re = evaluate(load_va, model, loss_fn)
                val_metric_list.append(val_loss)
                train_metric.append(tr_loss)
                total_val_loss= val_mse + (entropy_lambda*val_re) + (column_lambda*val_ce)

                if epoch>0:
                    if total_val_loss<best_val_loss:
                        early_stop_counter=0
                        model.save_weights(checkpoint_path)
                        best_val_loss= total_val_loss
                        best_model_mse= val_mse
                    else:
                        early_stop_counter+=1

                all_callbacks.on_epoch_end(epoch, {'train_mse':tr_mse, 'train_rmse':tr_rmse, 'train_mae':tr_mae, 'val_mse':val_mse, 'val_rmse:':val_rmse, 'val_mae':val_mae, 'train_row_penalty':tr_re, 'train_column_penalty':tr_ce, 'val_row_penalty':val_re, 'val_column_penalty':val_ce, 'val_total':total_val_loss})

                if early_stop_counter==patience:
                    all_callbacks.on_train_end(logs)
                    gen_plots(train_metric, val_metric_list)
                    return {"score": best_model_mse}
                else:
                    epoch+=1
    all_callbacks.on_train_end(logs)
    gen_plots(train_metric, val_metric_list)
    # Return final stats.
    return {"score": best_model_mse}


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
      NUM_MODELS = 500
     # sys.stdout = open('./debug.txt', 'w')

      trial_space = {
            'embedding_size': tune.choice([4,8,16,32,64]),
            'batch_size': tune.choice([4,8,16,32,64]),
            #'dr1': tune.uniform(0, 1),
            'entropy_lambda': tune.loguniform(1, 1e8),
            'column_lambda': tune.loguniform(1, 1e8),
            #'min_delta': tune.loguniform(1e-3, 1e8),
            'fc_size': tune.choice([4,8,16,32,64]),
            'fc_num': tune.choice([1,2,3]),
            'fc_size2': tune.choice([4,8,16,32,64]),
            'fc_num2': tune.choice([1,2,3]),
            'lr': tune.loguniform(1e-8, 1e-1)
        }

      bohb_hyperband = HyperBandForBOHB(
        time_attr="training_iteration",
        max_t=10,
        reduction_factor=4,
        stop_last_trials=False,
      )


      bohb = BOHBRepeater(metric='score', mode='min', repeat=4, max_concurrent=12)
      #bohb = tune.search.ConcurrencyLimiter(bohb, max_concurrent=4)

      train_model_object = tune.with_resources(train_model, {"cpu": 1})
      tuner = tune.Tuner(train_model_object, tune_config=tune.TuneConfig(
        search_alg=bohb,
        scheduler=bohb_hyperband,
        metric='score',
        mode='min',
        num_samples=NUM_MODELS), param_space=trial_space)
      results = tuner.fit()
      print(results)
