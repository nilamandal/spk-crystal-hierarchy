import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import sys
import argparse
from spektral.data import DisjointLoader
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import sparse_categorical_accuracy
import pandas as pd
from tensorflow.keras.callbacks import CallbackList, CSVLogger
from tensorflow.keras import backend as K
import json
from spektral_essential_objects import MyDataset, HNetSingleJanossy, Edgepool
from multiprocessing import Process, Lock, Value, Manager, Semaphore
import resource

parser = argparse.ArgumentParser(description='crystal hierarchy arguments.')
parser.add_argument('--datadir', dest='datadir',
        help='Directory where dataset is located', default='../Main_fol_Zintl')
parser.add_argument('--num-nbrs', dest='num_nbrs', type=int,
                    help='num neighbors per atom', default=12)
parser.add_argument('--radius-angstroms', dest='radius_angstroms', type=int,
                    help='search radius for neighbors', default=10)
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
        pred, s = model(inputs, training=False)

#        c_p, r_e= row_e_and_column_p(s, i)
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
                rmse,
                mae
            )
        #output.append(outs)
        del pred
        del s
        del inputs
        del x
        del a
        del e
        del i
        if step == loader.steps_per_epoch:
            output = np.array(outs)
            #print(np.average(output[:, :-1], 0))
            return output

def train_step(inputs, target, model, loss_fn, optimizer):
    with tf.GradientTape() as tape:
        predictions, s = model(inputs, training=True)
        #print(target)
        #print(predictions)
        loss = loss_fn(target, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    if args.task=='r':
        mse = tf.reduce_mean((target-predictions)**2)

        return loss, mse
    if args.task=='c':
        sca= tf.reduce_mean(sparse_categorical_accuracy(target, predictions))
        return loss, sca


def gen_plots(train_metric, val_metric, out_path):
    plt.switch_backend('Agg')

    plt.figure()

    epochs=list(range(len(train_metric)))
    min_train= 'train min='+str(np.round(np.min(train_metric), decimals=3))+','
    min_val= 'val min='+str(np.round(np.min(val_metric), decimals=3))
    plt.plot(epochs, np.log(train_metric), label='training loss')
    plt.plot(epochs, np.log(val_metric), label='val loss')

    figtitle=out_path+'result.png'

    plt.xlabel('epochs')
    plt.legend()
    plt.ylabel('log of mean square error ')
    plt.savefig(figtitle)

def train_model(idx, lr, beta, output_path):
    print('BEGUN INDIVIDUAL TRAINING')
    checkpoint_path= output_path+'goodmodel.ckpt'
    batch_size= 32
    epochs = 1000
    if epochs<1000:
        print('WARNING: CURRENTLY RUNNING IN DEBUG MODE WITH '+str(epochs)+' EPOCHS')

    train_df = pd.read_csv(os.path.join(args.datadir,'train_no_metals.csv'))
    #train_df= train_df.head(100)
    train_data= MyDataset(train_df, args.datadir, args.radius_angstroms, args.num_nbrs, args.task)
    load_tr= DisjointLoader(train_data, batch_size=batch_size, epochs=epochs)
    load_tr_eval= DisjointLoader(train_data, batch_size=len(train_data))
    print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9)

    val_df = pd.read_csv(os.path.join(args.datadir,'val_no_metals.csv'))
    #val_df= val_df.head(100)
    val_data= MyDataset(val_df, args.datadir, args.radius_angstroms, args.num_nbrs, args.task)
    load_va= DisjointLoader(val_data, batch_size=len(val_data))
    print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9)

    csv_log = CSVLogger(output_path+"callback_results.csv")
    model= Edgepool('r', 1, beta=beta, return_s=True)

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
    print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9)

    for batch in load_tr:
       if step==0:
           all_callbacks.on_epoch_begin(epoch, logs=logs)
       step += 1
       all_callbacks.on_train_batch_begin(step)
       loss, metric = train_step(*batch, model, loss_fn, optim)
       all_callbacks.on_train_batch_end(step, logs)
       print(epoch, step)
       print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9)
       # if tf.math.is_nan(loss):
       #          all_callbacks.on_train_end(logs)
       #          if epoch>1:
       #              gen_plots(train_metric, val_metric_list, output_path)
       #          best_model_mse= np.float64(best_model_mse)
       #          result_dict={'idx':idx, 'lr':lr, 'beta':beta, 'result':best_model_mse}
       #          #print(result_dict)
       #          with open(output_path+'result.json', 'w') as fp:
       #              json.dump(result_dict, fp)
       #          return result_dict
       if step == load_tr.steps_per_epoch:
                step = 0
                loss_str="Loss: {}".format(loss / load_tr.steps_per_epoch)

                tr_mse, tr_rmse, tr_mae= evaluate(load_tr_eval, model, loss_fn)
                val_mse, val_rmse, val_mae= evaluate(load_va, model, loss_fn)
                val_metric_list.append(val_mse)
                train_metric.append(tr_mse)
                #total_val_loss= val_mse
                if epoch>0:
                    if val_mse<best_val_loss:
                        early_stop_counter=0
                        model.save_weights(checkpoint_path)
                        best_val_loss= val_mse
                        best_model_mse= val_mse
                    else:
                        early_stop_counter+=1
                all_callbacks.on_epoch_end(epoch, {'train_mse':tr_mse, 'train_rmse':tr_rmse, 'train_mae':tr_mae, 'val_mse':val_mse, 'val_rmse:':val_rmse, 'val_mae':val_mae})

                if early_stop_counter==patience:
                    all_callbacks.on_train_end(logs)
                    gen_plots(train_metric, val_metric_list, output_path)
                    best_model_mse= np.float64(best_model_mse)
                    result_dict={'idx':idx, 'lr':lr, 'beta':beta, 'result':best_model_mse}
                    #print(result_dict)
                    with open(output_path+'result.json', 'w') as fp:
                        json.dump(result_dict, fp)
                    return result_dict
                else:
                    epoch+=1
                #del

    all_callbacks.on_train_end(logs)
    gen_plots(train_metric, val_metric_list, output_path)
    best_model_mse= np.float64(best_model_mse)
    result_dict={'idx':idx, 'lr':lr, 'beta':beta, 'result':best_model_mse}
    #print(result_dict)
    with open(output_path+'result.json', 'w') as fp:
       json.dump(result_dict, fp)
    return result_dict

if __name__ == "__main__":
      NUM_MODELS = 2
      main_output_path= './debug/beta4/'
      if not os.path.exists(main_output_path):
         os.makedirs(main_output_path)
      #main_out_file=open(main_output_path+'/results.csv','w+')
      #main_out_file.write('model_idx,beta,lr,val_mse \n')

      processlist=[]
      beta_set=[2]
      lr_set=[-1]
      for ex in beta_set:
         beta= 10**ex
         for ex2 in lr_set:
            lr= 10**ex2
            #print(NUM_MODELS, beta, lr)
            current_output_path=main_output_path+'/'+str(NUM_MODELS)+'/'
            if not os.path.exists(current_output_path):
               os.makedirs(current_output_path)

            out=train_model(NUM_MODELS, lr, beta, current_output_path)
            #p.start()
            #processlist.append(p)
            NUM_MODELS+=1
