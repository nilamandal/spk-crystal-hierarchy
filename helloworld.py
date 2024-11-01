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
from scipy.stats import qmc

parser = argparse.ArgumentParser(description='crystal hierarchy arguments.')
parser.add_argument('--datadir', dest='datadir',
        help='Directory where dataset is located', default='/home/nim18004/Main_fol_Zintl')
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

        if step == loader.steps_per_epoch:
            output = np.array(outs)
            #print(np.average(output[:, :-1], 0))
            return output

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

def train_model(idx, config, output_path):
    checkpoint_path= output_path+'goodmodel.ckpt'
    embedding_size= 2**int(np.ceil(config[0]))
    beta= 10**config[1]

    lr= 10**config[2]
    batch_size= 2**int(np.ceil(config[3]))

    jsonparams={'embedding_size':embedding_size,
                'batch_size':batch_size,
                'beta': beta,
                'lr':lr}
    json_object = json.dumps(jsonparams, indent=4)
    with open(output_path+"params.json", "w") as outfile:
        outfile.write(json_object)

    epochs = 1000
    if epochs<1000:
        print('WARNING: CURRENTLY RUNNING IN DEBUG MODE WITH '+str(epochs)+' EPOCHS')

    train_df = pd.read_csv(os.path.join(args.datadir,'train_no_metals.csv'))
    train_data= MyDataset(train_df, args.datadir, args.radius_angstroms, args.num_nbrs, args.task)
    load_tr= DisjointLoader(train_data, batch_size=batch_size, epochs=epochs)
    load_tr_eval= DisjointLoader(train_data, batch_size=len(train_data))

    val_df = pd.read_csv(os.path.join(args.datadir,'val_no_metals.csv'))
    val_data= MyDataset(val_df, args.datadir, args.radius_angstroms, args.num_nbrs, args.task)
    load_va= DisjointLoader(val_data, batch_size=len(val_data))

    csv_log = CSVLogger(output_path+"callback_results.csv")
    model= Edgepool('r', 1, beta=beta, embedding_size=embedding_size, return_s=True)

    optim=Adam(lr)
    loss_fn= MeanSquaredError()

    train_metric=[]
    val_metric_list=[]
    early_stop_counter= 0
    patience= 20
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

       if tf.math.is_nan(loss):
                all_callbacks.on_train_end(logs)
                if epoch>1:
                    gen_plots(train_metric, val_metric_list, output_path)
                best_model_mse= np.float64(best_model_mse)
                result_dict={'idx':idx, 'lr':lr, 'beta':beta, 'result':best_model_mse}
                print(result_dict)
                with open(output_path+'result.json', 'w') as fp:
                    json.dump(result_dict, fp)
                #raise Exception('found a nan')
                return result_dict

       if step == load_tr.steps_per_epoch:
                print('epoch', epoch)
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
                #    print(result_dict)
                    with open(output_path+'result.json', 'w') as fp:
                        json.dump(result_dict, fp)
                    return result_dict
                else:
                    epoch+=1

    all_callbacks.on_train_end(logs)
    gen_plots(train_metric, val_metric_list, output_path)
    best_model_mse= np.float64(best_model_mse)
    result_dict={'idx':idx, 'lr':lr, 'beta':beta, 'result':best_model_mse}
    #print(result_dict)
    del model
    with open(output_path+'result.json', 'w') as fp:
       json.dump(result_dict, fp)
    return result_dict


if __name__ == "__main__":
      NUM_MODELS = 0
      main_output_path= './serious_lhc_2/'
      if not os.path.exists(main_output_path):
         os.makedirs(main_output_path)

      sampler = qmc.LatinHypercube(d=4)
      quantity=30
      #embeddingsize, beta,  lr, batch size
      l_bounds= [1, 1, -6, 1]
      u_bounds= [8, 4, -1, 8]
      sample = sampler.random(n=quantity)
      scaled_sample= qmc.scale(sample, l_bounds, u_bounds)

      for s in scaled_sample:
         print(s)
         print(type(s))
         current_output_path=main_output_path+'/'+str(NUM_MODELS)+'/'
         if not os.path.exists(current_output_path):
            os.makedirs(current_output_path)
         out=train_model(NUM_MODELS, s, current_output_path)
         NUM_MODELS+=1
         print(out)
