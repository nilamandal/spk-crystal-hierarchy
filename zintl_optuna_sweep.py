import optuna
import json
import resource
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import logging
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import pymysql
pymysql.install_as_MySQLdb()
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

import sklearn.datasets
import sklearn.linear_model
import sklearn.model_selection

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError, CategoricalCrossentropy
from tensorflow.keras.callbacks import CallbackList, CSVLogger

from spektral.data import DisjointLoader, BatchLoader
from spektral_essential_objects import MyDataset, SparseEdgepool, AtomFeaDataset, NotShrinking, CGCNNModel
from utils import train_step, evaluate

parser = argparse.ArgumentParser(description='crystal hierarchy arguments.')
parser.add_argument('--datadir', dest='datadir',
        help='Directory where dataset is located', default='/home/nim18004/Main_fol_Zintl')

parser.add_argument('--task', choices=['r', 'c'],
                    default='r', help='complete a regression or classification task (default: regression)')
parser.add_argument('--out_dir', default='./debug_save/', help='directory where output is saved')
args = parser.parse_args(sys.argv[1:])
if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)


def objective(trial):
    trial_name= args.out_dir+str(trial.number)+'/'
    if not os.path.exists(trial_name):
        os.makedirs(trial_name)
    print(trial_name)

    train_csv= pd.read_csv(os.path.join(args.datadir,'train_100_ternary.csv'))
    val_csv= pd.read_csv(os.path.join(args.datadir,'val_by_fam_ternary.csv'))
    cv_scores=[]

    epochs = 10
    if epochs<1000:
        print('WARNING: CURRENTLY RUNNING IN DEBUG MODE WITH '+str(epochs)+' EPOCHS')

    config= {}
    config['trial']= trial.number
    config['embedding_size']=trial.suggest_int('embedding_size', 4, 128, step=4, log=False)
    config['cgcnn_num']= trial.suggest_int('cgcnn_num', 1, 5, step=1, log=False)
    config['cgcnn_num2']= trial.suggest_int('cgcnn_num2', 1, 5, step=1, log=False)
    #config['hidden_size']=trial.suggest_int('hidden_size', 4, 128, step=4, log=False)
    #config['num_layers']= trial.suggest_int('num_layers', 1, 3, step=1, log=False)
    #config['hidden_size_2']=trial.suggest_int('hidden_size_2', 4, 64, step=4, log=False)
    #config['num_layers_2']= trial.suggest_int('num_layers_2', 1, 3, step=1, log=False)
    config['batch_size']= trial.suggest_int('batch_size', 4, 64, step=4, log=False)
    config['softmax_beta']= trial.suggest_float('softmax_beta', 1, 1e8, log=True)
    config['lr']= trial.suggest_float('lr', 1e-8, 1e-2, log=True)
    config['clipnorm']= trial.suggest_float('clipnorm', 1.0, 1e4, log=True)
    with open(trial_name+'params.json', 'w') as to_file:
        json.dump(config, to_file)



    #for i in range(5):
        #train_df= all_data[all_data['bin']!=i]
        #train_df= train_df.head(4)
        #val_df= all_data[all_data['bin']==i]
        #val_df= val_df.head(4)

        #batch mode:
        #train_data= AtomFeaDataset(train_df, args.datadir, 8, 12, args.task)
        #val_data= AtomFeaDataset(val_df, args.datadir, 8, 12, args.task)
        #load_tr= BatchLoader(train_data, mask=True, batch_size=config['batch_size'], epochs=epochs)
        #load_tr_eval= BatchLoader(train_data, mask=True, batch_size=len(train_data))
        #load_va= BatchLoader(val_data, mask=True, batch_size=len(val_data))

    #disjoint mode:
    train_data= AtomFeaDataset(train_csv, args.datadir, 8, 12, args.task)
    val_data= AtomFeaDataset(val_csv, args.datadir, 8, 12, args.task)
    load_tr= DisjointLoader(train_data, batch_size=config['batch_size'], epochs=epochs)
    load_tr_eval= DisjointLoader(train_data, batch_size=len(train_data))
    load_va= DisjointLoader(val_data, batch_size=len(val_data))


    model= NotShrinking(args.task, 1, config['embedding_size'], config['cgcnn_num'], config['cgcnn_num2'], softmax_beta=config['softmax_beta'])
    #model= CGCNNModel(config['embedding_size'], config['hidden_size'], config['num_layers'])
    optim=Adam(config['lr'], clipnorm=config['clipnorm'])
    try:
       trial_score= train_one_loop(model, load_tr, load_tr_eval, load_va, optim, trial, trial_name)
    except:
       trial_score= np.inf

    #if trial.should_prune():
    print(trial_name+'done')
        #raise optuna.TrialPruned()
    return trial_score





def train_one_loop(model, load_tr, load_tr_eval, load_va, optim, trial, path_i):
    checkpoint_path=path_i+'goodmodel.keras'
    csv_log = CSVLogger(path_i+"callback_results.csv")
    #print('loop entered')
    all_callbacks= CallbackList([csv_log], add_history=True, model=model)

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
    logs = {}
    all_callbacks.on_train_begin(logs=logs)

    for batch in load_tr:
        #print(epoch, step)
        if step==0:
            all_callbacks.on_epoch_begin(epoch, logs=logs)
        step += 1
        #print(epoch, step)

        all_callbacks.on_train_batch_begin(step)
        loss, metric = train_step(*batch, model, loss_fn, optim)
        all_callbacks.on_train_batch_end(step, logs)
        #print('---')
        if tf.math.is_nan(loss):
            print('nan occurred')
            all_callbacks.on_train_end(logs)
            if epoch>1:
                gen_plots(train_metric, val_metric_list, path_i)
            return np.inf

        if step == load_tr.steps_per_epoch:
            step = 0
            loss_str="Loss: {}".format(loss / load_tr.steps_per_epoch)
            tr_loss, tr_rmse, tr_mae= evaluate(load_tr_eval, model, loss_fn)
            val_loss, val_rmse, val_mae= evaluate(load_va, model, loss_fn)
            val_metric_list.append(val_loss)
            train_metric.append(tr_loss)
            #total_val_loss= val_loss
            print(val_loss)
            if epoch>0:
                if val_loss<best_val_loss:
                    early_stop_counter=0
                    model.save(checkpoint_path)
                    best_val_loss= val_loss

                else:
                    early_stop_counter+=1
                if epoch%50==0:
                   trial.report(val_loss,epoch)
                   print(trial)
                   if trial.should_prune():
                       print(trial)
                       raise optuna.TrialPruned()
            if args.task=='r':
                all_callbacks.on_epoch_end(epoch, {'train_mse':tr_loss, 'train_rmse':tr_rmse, 'train_mae':tr_mae, 'val_mse':val_loss, 'val_rmse:':val_rmse, 'val_mae':val_mae})
            else:
                all_callbacks.on_epoch_end(epoch, {'train_loss':tr_loss, 'val_loss:':val_loss})
            if early_stop_counter==patience:
                all_callbacks.on_train_end(logs)
                gen_plots(train_metric, val_metric_list, path_i)
                return best_val_loss
            else:
                epoch+=1

    all_callbacks.on_train_end(logs)
    gen_plots(train_metric, val_metric_list, path_i)

    return best_val_loss





def gen_plots(train_metric, val_metric, path='./'):
    plt.switch_backend('Agg')

    plt.figure()

    epochs=list(range(len(train_metric)))
    min_train= 'train min='+str(np.round(np.min(train_metric), decimals=3))+','
    min_val= 'val min='+str(np.round(np.min(val_metric), decimals=3))
    plt.plot(epochs, np.log(train_metric), label='training loss')
    plt.plot(epochs, np.log(val_metric), label='val loss')

    figtitle=path+'_result.png'

    plt.xlabel('epochs')
    plt.legend()
    plt.ylabel('log of mean square error ')
    plt.savefig(figtitle)
    plt.close('all')

if __name__ == "__main__":

    study = optuna.create_study()
    print(f"Sampler is {study.sampler.__class__.__name__}")

    # Add stream handler of stdout to show the messages
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
      # Unique identifier of the study.

    #storage_name = "sqlite:///{}out.db".format(args.out_dir)
    #storage_name="mysql://nila:scimlab2@localhost/example"
    storage_name = JournalStorage(JournalFileBackend("debug.log"))
    study = optuna.create_study(pruner=optuna.pruners.HyperbandPruner(), study_name='debug', storage=storage_name, load_if_exists=True)
    #study = optuna.create_study(study_name='out', storage=storage_name)
    study.optimize(objective, n_trials=2, n_jobs=2)
    df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    print(df)
    df.to_csv(args.out_dir+'results.csv')
    #print(study.best_trial.value, study.best_trial.params)
