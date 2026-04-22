import argparse
import sys
import os
from spektral_essential_objects import MyDataset, CGCNNModel, AtomFeaDataset, gen_plots
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
import numpy as np
from tensorflow.keras.metrics import sparse_categorical_accuracy #, mean_squared_error
from tensorflow.keras.callbacks import CallbackList, CSVLogger
from tensorflow.keras import backend as K
from spektral.data import DisjointLoader
import pandas as pd

from ray import tune
from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
from ray.tune.search.bohb import TuneBOHB
import ConfigSpace
from hpbandster.optimizers.config_generators.bohb import BOHB
from CorrectedRepeater import BOHBRepeater


parser = argparse.ArgumentParser(description='crystal hierarchy arguments.')
parser.add_argument('--datadir', dest='datadir',
        help='Directory where dataset is located', default='/Users/nilamandal/desktop/Main_fol_Zintl')

parser.add_argument('--task', choices=['r', 'c'],
                    default='r', help='complete a regression or classification task (default: regression)')
args = parser.parse_args(sys.argv[1:])

def evaluate(loader, model, loss_fn, test=False):
    step = 0
    output=[]
    while step < loader.steps_per_epoch:
        step += 1
        inputs, target = loader.__next__()
        x, a, e, i = inputs
        pred = model(inputs, training=False)

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
                len(target),  # Keep track of batch size
            )
        output.append(outs)
        if step == loader.steps_per_epoch:
            output = np.array(output)
            return np.average(output[:, :-1], 0, weights=output[:, -1])

def train_step(inputs, target, model, loss_fn, optimizer):
    with tf.GradientTape() as tape:
        predictions= model(inputs, training=True)
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
    print('BEGUN INDIVIDUAL TRAINING')

    checkpoint_path='./goodmodel.ckpt'

    epochs = 1000
    if epochs<1000:
        print('WARNING: CURRENTLY RUNNING IN DEBUG MODE WITH '+str(epochs)+' EPOCHS')

    train_df = pd.read_csv(os.path.join(args.datadir,'train_no_metals.csv'))
    #train_df = train_df.head(20)
    train_data= AtomFeaDataset(train_df, args.datadir, 8, config['num_nbrs'], args.task)
    #print(train_data)
    load_tr= DisjointLoader(train_data, batch_size=config['batch_size'], epochs=epochs)
    load_tr_eval= DisjointLoader(train_data, batch_size=len(train_data))

    val_df = pd.read_csv(os.path.join(args.datadir,'val_no_metals.csv'))
    #val_df = val_df.head(20)
    val_data= AtomFeaDataset(val_df, args.datadir, 8, config['num_nbrs'], args.task)
    load_va= DisjointLoader(val_data, batch_size=len(val_data))
    print('loaded data')
    csv_log = CSVLogger("./callback_results.csv")

    model=CGCNNModel(embedding_size=config['embedding_size'], hidden_fea_size=config['hidden_fea_size'])

    print(model)
    print('ok')

    all_callbacks= CallbackList([csv_log], add_history=True, model=model)
    #
    optim=Adam(config['lr'])
    loss_fn= MeanSquaredError()

    train_metric=[]
    val_metric_list=[]
    early_stop_counter= 0
    patience= 25
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
                return {"score": np.inf}

            if step == load_tr.steps_per_epoch:
                step = 0
                loss_str="Loss: {}".format(loss / load_tr.steps_per_epoch)

                tr_loss, tr_mse, tr_rmse, tr_mae= evaluate(load_tr_eval, model, loss_fn)
                val_loss, val_mse, val_rmse, val_mae = evaluate(load_va, model, loss_fn)
                val_metric_list.append(val_loss)
                train_metric.append(tr_loss)
                total_val_loss= val_mse

                if epoch>0:
                    if total_val_loss<best_val_loss:
                        early_stop_counter=0
                        model.save_weights(checkpoint_path)
                        best_val_loss= total_val_loss
                        best_model_mse= val_mse
                    else:
                        early_stop_counter+=1

                all_callbacks.on_epoch_end(epoch, {'train_mse':tr_mse, 'train_rmse':tr_rmse, 'train_mae':tr_mae, 'val_mse':val_mse, 'val_rmse:':val_rmse, 'val_mae':val_mae, 'val_total':total_val_loss})

                if early_stop_counter==patience:
                    all_callbacks.on_train_end(logs)
                    gen_plots(train_metric, val_metric_list)
                    return {"score": best_model_mse}
                else:
                    epoch+=1
    all_callbacks.on_train_end(logs)
    gen_plots(train_metric, val_metric_list)

    return {"score": best_model_mse}


if __name__ == "__main__":
      NUM_MODELS = 100

      trial_space = {
            'embedding_size': tune.choice([4,8,16,32,64, 128]),
            'hidden_fea_size': tune.choice([4,8,16,32,64, 128]),
            'num_nbrs': tune.choice([1,2,3,4,5,6,7,8,9,10,11,12]),
            'batch_size': tune.choice([4,8,16,32,64]),
            'lr': tune.loguniform(1e-8, 1e-1)
        }

      bohb_hyperband = HyperBandForBOHB(
        time_attr="training_iteration",
        max_t=81,
        reduction_factor=3,
        stop_last_trials=False,
      )

      bohb = BOHBRepeater(metric='score', mode='min', repeat=1, max_concurrent=20)
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
