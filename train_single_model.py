import tensorflow as tf
import os
import sys
import argparse
from spektral_essential_objects import MyDataset, SparseEdgepool, CGCNNModel, AtomFeaDataset, NotShrinking
#from edgepool_w_error_objects import SparseEdgepool
from spektral.data import DisjointLoader
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.losses import MeanSquaredError
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import CallbackList, CSVLogger
from tensorflow.keras import backend as K
import json
import random
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from utils import train_step, evaluate

parser = argparse.ArgumentParser(description='crystal hierarchy arguments.')
parser.add_argument('--datadir', dest='datadir',
        help='Directory where dataset is located', default='./ExampleContcars/')
parser.add_argument('--num-nbrs', dest='num_nbrs', type=int,
                    help='num neighbors per atom', default=12)
parser.add_argument('--radius-angstroms', dest='radius_angstroms', type=int,
                    help='search radius for neighbors', default=10)
parser.add_argument('--task', choices=['r', 'c'],
                    default='r', help='complete a regression or classification task (default: regression)')
args = parser.parse_args(sys.argv[1:])


def train_model(config):
    print('BEGUN INDIVIDUAL TRAINING')

    checkpoint_path='./goodmodel.ckpt'

    epochs = 1000
    if epochs<1000:
        print('WARNING: CURRENTLY RUNNING IN DEBUG MODE WITH '+str(epochs)+' EPOCHS')
    #
    # embedding_size= config['embedding_size']
    # batch_size= config['batch_size']
    # entropy_lambda= config['entropy_lambda']
    # softmax_beta= config['softmax_beta']
    # lr= config['lr']


    # Load data and train model code here...
    train_df = pd.read_csv(os.path.join(args.datadir,'example_data.csv'))
    #train_df = train_df.head(1000)
    train_data= MyDataset(train_df, args.datadir, 8, int(config['num_nbrs']), args.task)
    load_tr= DisjointLoader(train_data, batch_size=int(config['batch_size']), epochs=epochs)
    load_tr_eval= DisjointLoader(train_data, batch_size=len(train_data))

    val_df = pd.read_csv(os.path.join(args.datadir,'example_data.csv'))

    val_data= MyDataset(val_df, args.datadir, 8, int(config['num_nbrs']), args.task)
    load_va= DisjointLoader(val_data, batch_size=len(val_data))
    print('loaded data')
    csv_log = CSVLogger("./callback_results.csv")

    #model= SparseEdgepool('r', 1, embedding_size=int(config['embedding_size']), cgcnn_num=int(config['cgcnn_num']), cgcnn_num2=int(config['cgcnn_num2']), softmax_beta=config['softmax_beta'], return_s=True)
    model= NotShrinking('r', 1)
    all_callbacks= CallbackList([csv_log], add_history=True, model=model)
    #
    optim=Adam(config['lr'])
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
            #print(epoch)
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

                tr_loss, tr_mse, tr_rmse, tr_mae, tr_ce, tr_re= evaluate(load_tr_eval, model, loss_fn)
                val_loss, val_mse, val_rmse, val_mae, val_ce, val_re = evaluate(load_va, model, loss_fn)
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

                all_callbacks.on_epoch_end(epoch, {'train_mse':tr_mse, 'train_rmse':tr_rmse, 'train_mae':tr_mae, 'val_mse':val_mse, 'val_rmse:':val_rmse, 'val_mae':val_mae, 'train_row_penalty':tr_re, 'train_column_penalty':tr_ce, 'val_row_penalty':val_re, 'val_column_penalty':val_ce, 'val_total':total_val_loss})

                if early_stop_counter==patience:
                    all_callbacks.on_train_end(logs)
                    gen_plots(train_metric, val_metric_list)
                    return {"score": best_model_mse}
                else:
                    epoch+=1
    all_callbacks.on_train_end(logs)
    gen_plots(train_metric, val_metric_list)

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

def callback_plots():
    crazylist=['../train_model_2023-12-05_10-16-27/']
    for path in crazylist:
        df_getpaths= pd.read_csv(path+'evaluated_results.csv')
        getpaths= list(df_getpaths['path'])
#'train_model_03ec7b5e_41_trial_index=0,batch_size=4,column_lambda=84.4158,dr1=0.7048,embedding_size=128,entropy_lambda=4971905.8384_2023-11-07_21-09-21']
        for subpath in getpaths:
            fullpath=path+str(subpath)+'/'
            df=pd.read_csv(fullpath+'callback_results.csv')
            paramsdict= json.load(open(fullpath+'params.json'))

            el=paramsdict['entropy_lambda']
            cl=paramsdict['column_lambda']
            plt.figure(figsize=(10,10))
            plt.plot(np.log10(-1*df['train_column_penalty']), label='train column product')
            plt.plot(np.log10(df['train_mse']), label='train_mse')
            #plt.plot(np.log10(df['train_mse']), label='train_mse')
            plt.plot(np.log10(df['train_row_penalty']), label='train_row_entropy')
            plt.plot(np.log10(-1*df['val_column_penalty']), label='val column product')
            plt.plot(np.log10(df['val_mse']), label='val_mse')
            plt.plot(np.log10(df['val_row_penalty']), label='val_row_entropy')
            plt.legend()
            plt.xlabel('training epochs')
            plt.ylabel('log10 values of loss function components')
            plt.title(subpath)
            plt.savefig(fullpath+'/allcomponents.png')
            plt.figure(figsize=(10,10))
            df['train_total']=df['train_mse']+(el*df['train_row_penalty'])+(cl*df['train_column_penalty'])
            df['val_total']=df['val_mse']+(el*df['val_row_penalty'])+(cl*df['val_column_penalty'])
            plt.plot(df['train_total'], label='train loss')
            plt.plot(df['val_total'], label='val loss')
            plt.legend()
            plt.xlabel('training epochs')
            plt.ylabel('sum of validation components')
            plt.savefig(fullpath+'/sum_losscomponents.png')



def get_available(filename):
    try:
        crystal= Structure.from_file('../Main_fol_Zintl/'+filename)
        ana= SpacegroupAnalyzer(crystal)
        #print(ana)
        sym_crystal= ana.get_symmetrized_structure()
        #print(len(sym_crystal.equivalent_indices))
        return len(sym_crystal.equivalent_indices)

    except:
        print(filename)
        #return False

def scale_by_pred_vol(structure, site_bias, dls_vol_predictor):
    #global count
    # first predict the volume using the average volume per element (from ICSD)
    site_counts = pd.Series(Counter(
        str(site.specie) for site in structure.sites)).fillna(0)
    curr_site_bias = site_bias[site_bias.index.isin(site_counts.index)]

    try:
        linear_pred = site_counts @ curr_site_bias
        structure.scale_lattice(linear_pred)
    except:
        pass
        #count+=1
    # then apply Pymatgen's DLS predictor
    pred_volume = dls_vol_predictor.predict(structure)
    structure.scale_lattice(pred_volume)
    #
    return structure

def scale_dls_only(c):
    c=str(c)
    try:
        from pymatgen.core.structure import Structure
    except:
        crystal= Structure.from_file(os.path.join(data_path,c))
    structure= dls_vol_predictor.get_predicted_structure(crystal)
    newpath='./sc24_scaled/'+c.split('/')[-1][:-7]+'.cif'
    structure.to(filename=newpath)
    return newpath


if __name__ == "__main__":
    config={'batch_size': 256,
            'lr': 1e-4,
            'num_nbrs': 8
    }
    train_model(config)
