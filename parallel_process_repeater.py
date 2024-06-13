from spektral.data import Graph, Dataset, DisjointLoader
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import MeanSquaredError, SparseCategoricalCrossentropy
from tensorflow.keras.metrics import sparse_categorical_accuracy, mean_squared_error
from tensorflow.keras.callbacks import CallbackList, CSVLogger
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import os
import sys
import random
from pymatgen.core.structure import Structure
import json
import argparse
import time
from spektral_essential_objects import GaussianDistance, MyDataset, HNetDoubleJanossy
from multiprocessing import Process, Lock, Value, Manager, Semaphore
from scipy.stats import qmc
import matplotlib.pyplot as plt


begin_time = time.time()
parser = argparse.ArgumentParser(description='crystal hierarchy arguments.')

parser.add_argument('--datadir', dest='datadir',
        help='Directory where dataset is located', default='../Main_fol_Zintl/')
#parser.add_argument('--filename', dest='filename',
#                    help='csv where data is located', default='id_prop.csv')
parser.add_argument('--file-out', dest='file_out',
                    help='output file name', default='doublejanossy')
parser.add_argument('--path-out', dest='path',
                    help='output path', default='./debug_only')
parser.add_argument('--num-atoms', dest='num_atoms', type=int,
                    help='Maximum number of nodes', default=200)
parser.add_argument('--num-nbrs', dest='num_nbrs', type=int,
                    help='num neighbors per atom', default=12)
parser.add_argument('--num-classes', dest='num_classes', type=int,
                    help='Number of label classes', default=3)
parser.add_argument('--radius-angstroms', dest='radius_angstroms', type=int,
                    help='search radius for neighbors', default=10)
parser.add_argument('--random-seed', dest='random_seed', type=int,
                    help='random seed for numpy', default=0)
parser.add_argument('--optim', default='Adam', type=str, metavar='SGD',
                        help='choose an optimizer, SGD or Adam, (default: SGD)')
parser.add_argument('--epochs', default=500, type=int, metavar='N',
                    help='number of total epochs to run (default: 30)')
parser.add_argument('--task', choices=['r', 'c'],
                    default='r', help='complete a regression or '
                        'classification task (default: regression)')
parser.add_argument('--dataset', choices=['prashun', 'mp'],
                    default='prashun')

def test_eval(loader_te,model,loss_fn,textlist):
    test_loss, test_metric = evaluate(loader_te, model, loss_fn, True)
    textlist.append("Done. Test loss: {}".format(test_loss))
    if args.task=='r':
        textlist.append('test mse=')
    elif args.task=='c':
        textlist.append('test_acc=')
    textlist.append(test_metric)
    return textlist

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
    #print('The function is happening')
    for g in range(batch_size):
        count= np.count_nonzero(i==g)
        s_g=s[g,:count]
        #---
        row= entropy_loss(s_g)
        row_entropy_sum+=row

        #column_means=tf.divide(tf.reduce_sum(s_g, axis=0),s_g.shape[0])
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
        #print(s)
        c_e, r_e= row_e_and_column_p(s, i)
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
    #outputtxt=[]
    with tf.GradientTape() as tape:
        predictions, s = model(inputs, training=True)
        loss = loss_fn(target, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    if args.task=='r':
        mse = tf.reduce_mean((target-predictions)**2)
        return loss, mse#, outputtxt
    if args.task=='c':
        sca= tf.reduce_mean(sparse_categorical_accuracy(target, predictions))
        outputtxt.append(confusion_matrix(target,np.argmax(predictions, axis=1)))

        return loss, sca#, outputtxt

def full_training_loop(printlock, load_tr, load_va, lr, specialindex, model_list={}, performance_list={}, testing=False):

        init_time= time.time()
        fullpath=args.path+'/'+args.file_out+'/'+str(specialindex)
        print(fullpath)
        if not os.path.exists(fullpath):
            os.makedirs(fullpath)

        checkpoint_path=fullpath+'/goodmodel.ckpt'

        if args.optim=='Adam':
            optim = Adam(learning_rate=lr)
        elif args.optim=='SGD':
            optim = SGD(learning_rate=lr)
        if args.task=='c':
            loss_fn= SparseCategoricalCrossentropy()
        elif args.task=='r':
            loss_fn = MeanSquaredError()
        else:
            print(args.task, ' is not c or r.')

        csv_log = CSVLogger(fullpath+"/callback_results.csv")

        paramsdict={
          "batch_size": 32,
          "column_lambda": 41903766.3588113,
          "dr1": 0.2513546780919437,
          "embedding_size": 4,
          "entropy_lambda": 108506.39801000628,
          "fc_num": 1,
          "fc_num2": 1,
          "fc_size": 4,
          "fc_size2": 4,
          "lr": 0.0631443016175717
        }
        entropy_lambda= paramsdict['entropy_lambda']
        column_lambda= paramsdict['column_lambda']

        model= HNetDoubleJanossy('r', 1, embedding_size=paramsdict['embedding_size'], d1=paramsdict['dr1'], el=paramsdict['entropy_lambda'], cl=paramsdict['column_lambda'], fc_num=paramsdict['fc_num'], fc_size=paramsdict['fc_size'], fc_size2=paramsdict['fc_size2'], fc_num2=paramsdict['fc_num2'], return_s=True, random_seed=specialindex, path=fullpath)
        all_callbacks= CallbackList([csv_log], add_history=True, model=model)


        df = pd.read_csv(os.path.join(args.datadir,'train_with_counts_complete.csv'))
        df = df[df['num_elements']<=3]
        #df = df.head(3)
        load_tr_eval= DisjointLoader(MyDataset(df, args.datadir, args.radius_angstroms, args.num_nbrs, args.task), batch_size=len(df), epochs=args.epochs)



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

                    all_callbacks.on_epoch_end(epoch, {'train_mse':tr_mse, 'train_rmse':tr_rmse, 'train_mae':tr_mae, 'val_mse':val_mse, 'val_rmse:':val_rmse, 'val_mae':val_mae, 'train_row_penalty':tr_re, 'train_column_penalty':tr_ce, 'val_row_penalty':val_re, 'val_column_penalty':val_ce})

                    if early_stop_counter==patience:
                        all_callbacks.on_train_end(logs)
                        gen_plots(train_metric, val_metric_list, fullpath)
                        return {"score": best_model_mse}
                    else:
                        epoch+=1
        all_callbacks.on_train_end(logs)
        gen_plots(train_metric, val_metric_list, fullpath)


def train_model(config, id_num):
    write_output_path='./'+str(id_num)+'/'
    if not os.path.exists(write_output_path):
        os.makedirs(write_output_path)
    #embeddingsize, bs, fc_size, fc_size2, fc_num, fc_num2, entropy_lambda, column_lambda, lr
    checkpoint_path=write_output_path+'goodmodel.ckpt'

    epochs = 1000

    embedding_size= int(config[0])
    batch_size= int(config[1])
    fc_size= int(config[2])
    fc_size2= int(config[3])
    fc_num= int(config[4])
    fc_num2= int(config[5])
    entropy_lambda= config[6]
    column_lambda= config[7]
    lr= config[8]

    jsonparams={'embedsize':embedding_size,
                'bs':batch_size,
                'fc_size':fc_size,
                'fc_size2':fc_size2,
                'fc_num':fc_num,
                'fc_num2':fc_num2,
                'entropy lambda':entropy_lambda,
                'column lambda':column_lambda,
                'lr':lr}
    json_object = json.dumps(jsonparams, indent=4)
    with open(write_output_path+"params.json", "w") as outfile:
        outfile.write(json_object)

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

    csv_log = CSVLogger(write_output_path+"callback_results.csv")


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
                    gen_plots(train_metric, val_metric_list, write_output_path)
                    return {"score": best_model_mse}
                else:
                    epoch+=1
    all_callbacks.on_train_end(logs)
    gen_plots(train_metric, val_metric_list, write_output_path)
    # Return final stats.
    return {"score": best_model_mse}


def gen_plots(train_metric, val_metric,output_path):
    plt.switch_backend('Agg')

    plt.figure()

    epochs=list(range(len(train_metric)))
    min_train= 'train min='+str(np.round(np.min(train_metric), decimals=3))+','
    min_val= 'val min='+str(np.round(np.min(val_metric), decimals=3))
    plt.plot(epochs, np.log(train_metric), label='training loss')
    plt.plot(epochs, np.log(val_metric), label='val loss')

    figtitle='result.png'

    plt.xlabel('epochs')
    plt.legend()
    plt.ylabel('log of mean square error ')
    plt.savefig(output_path+figtitle)



if __name__ == '__main__':
    args = parser.parse_args(sys.argv[1:])

    num_complete_models= 0
    sampler = qmc.LatinHypercube(d=9)
    quantity=5
    #embeddingsize, bs, fc_size, fc_size2, fc_num, fc_num2, entropy_lambda, column_lambda, lr
    l_bounds= [2, 2, 2, 2, 1, 1, 1, 1, -6]#, 0, 0, 0]
    u_bounds= [6, 6, 6, 6, 3, 3, 8, 8, -1]#, 4, 4, 4]
    while num_complete_models<=200:
        processlist=[]
        sample = sampler.random(n=quantity)
        scaled_sample= qmc.scale(sample, l_bounds, u_bounds)
        for onesample in scaled_sample:
            onesample[0]= np.ceil(2**onesample[0])
            onesample[1]= np.ceil(2**onesample[1])
            onesample[2]= np.ceil(2**onesample[2])
            onesample[3]= np.ceil(2**onesample[3])
            onesample[4]= np.ceil(onesample[4])
            onesample[5]= np.ceil(onesample[5])
            onesample[6]= 2**onesample[6]
            onesample[7]= 2**onesample[7]
            onesample[8]= 10**onesample[8]
            #train_model(onesample, num_complete_models)

            p= Process(target=train_model, args=(onesample, num_complete_models))
            p.start()
            processlist.append(p)
            print(p, ' started', flush=True)
            num_complete_models+=1

        for pr in processlist:
            pr.join()
            print(pr)
            print('process batch complete')


    # #sys.stdout = open(args.path+'/'+args.file_out+'/'+args.file_out+'.txt', 'w')
    #
    # for i in range(5):
    #     df = pd.read_csv(os.path.join(args.datadir,'train_with_counts_complete.csv'))
    #     df = df[df['num_elements']<=3]
    #     #df= df.head(3)
    #     train_data= DisjointLoader(MyDataset(df, args.datadir, args.radius_angstroms, args.num_nbrs, args.task), batch_size=paramdict['batch_size'], epochs=args.epochs)
    #
    #     val_df = pd.read_csv(os.path.join(args.datadir,'val_with_counts_complete.csv'))
    #     val_df = val_df[val_df['num_elements']<=3]
    #     #val_df= val_df.head(3)
    #     val_data= DisjointLoader(MyDataset(val_df, args.datadir, args.radius_angstroms, args.num_nbrs, args.task), batch_size=len(val_df))
    #     full_training_loop(printlock, train_data, val_data, 0.0631443016175717, i)
    #     #p= Process(target=full_training_loop, args=(printlock, train_data, val_data, 0.0631443016175717, i))
    #     #processlist.append(p)
