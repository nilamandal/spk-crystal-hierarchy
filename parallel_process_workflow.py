from spektral.data import Graph, Dataset, DisjointLoader
from spektral.data.utils import to_batch
from spektral.layers import CrystalConv, DiffPool, ops, GlobalMaxPool#, Disjoint2Batch
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay, PiecewiseConstantDecay
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import MeanSquaredError, SparseCategoricalCrossentropy
from tensorflow.keras.metrics import sparse_categorical_accuracy, mean_squared_error
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
from spektral_essential_objects import GaussianDistance, MyDataset, HNetConcat, HNetConcatJanossy, ModifiedReduceLROnPlateau, HNetDoubleJanossy
from multiprocessing import Process, Lock, Value, Manager, Semaphore
from scipy.stats import qmc
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import CallbackList, CSVLogger

begin_time = time.time()
parser = argparse.ArgumentParser(description='crystal hierarchy arguments.')

parser.add_argument('--datadir', dest='datadir',
        help='Directory where dataset is located', default='../Main_fol_Zintl/')
#parser.add_argument('--filename', dest='filename',
#                    help='csv where data is located', default='id_prop.csv')
parser.add_argument('--file-out', dest='file_out',
                    help='output file name', default='doublejanossy')
parser.add_argument('--path-out', dest='path',
                    help='output path', default='./debug_entropy')
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
parser.add_argument('--epochs', default=1000, type=int, metavar='N',
                    help='number of total epochs to run (default: 30)')
parser.add_argument('--task', choices=['r', 'c'],
                    default='r', help='complete a regression or '
                        'classification task (default: regression)')
parser.add_argument('--dataset', choices=['prashun', 'mp'],
                    default='prashun')
#parser.add_argument('--patience', dest='patience',default=30, type=int,
#                    help='num epochs for early stopping')

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

def full_training_loop(printlock, load_tr, load_va, load_te, textlist, testelement, valelement, lr, specialindex, el, cl, model_list, performance_list, testing=False):

        textlist.append('----NEW EXP----')
        init_time= time.time()
        if testing:
            fullpath=args.path+'/'+args.file_out+'_testing'+'/'+str(specialindex)
        else:
            fullpath=args.path+'/'+args.file_out+'/'+str(specialindex)
        print(fullpath)
        if not os.path.exists(fullpath):
            os.makedirs(fullpath)

        checkpoint_path=fullpath+'/goodmodel.ckpt'
        #
        #lr_schedule = ExponentialDecay(
        #    initial_learning_rate=lr,
        #    decay_steps=decay_steps,
        #    decay_rate=decay_rate)


        if args.optim=='Adam':
            optimizer = Adam(learning_rate=lr)
        elif args.optim=='SGD':
            optimizer = SGD(learning_rate=lr)
        if args.task=='c':
            loss_fn= SparseCategoricalCrossentropy()
        elif args.task=='r':
            loss_fn = MeanSquaredError()
        else:
            print(args.task, ' is not c or r.')

        csv_log = CSVLogger(fullpath+"/callback_results.csv")
        # reduce_lr = ModifiedReduceLROnPlateau(
        #     monitor='val_loss',
        #     factor=0.2,
        #     patience=2,
        #     min_lr=0.00001,
        #     optim= optimizer,
        #     verbose=2
        # )
        paramdict={
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
        #embedding_size=52,  d1=0.578, el=427, cl=265, fc_num=1, fc_size=23, fc_num2=1, fc_size2=23, regularizer='l2', return_s=False,  random_seed=0
        model= HNetDoubleJanossy(args.task, args.num_classes, embedding_size=paramdict['embedding_size'], d1=paramdict['dr1'], el=el, cl=cl, return_s=True, random_seed=0)

        all_callbacks= CallbackList([csv_log], add_history=True, model=model)


        textlist.append('evaluation on train set before training:')
        #print(testelement, valelement)
        temp_tr=evaluate(load_tr, model, loss_fn)
        textlist.append(str(temp_tr))

        textlist.append('evaluation on val set before training:')
        temp_va=evaluate(load_va, model, loss_fn)
        textlist.append(str(temp_va))

        train_metric=[]
        val_metric_list=[]
        early_stop_counter= 0
        patience= 50
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
                    total_val_loss= val_mse + (entropy_lambda*val_re) + (column_lambda*val_ce)
                    if total_val_loss<best_val_loss:
                        model.save_weights(checkpoint_path)
                        best_val_loss= total_val_loss
                        early_stop_counter=0
                    else:
                        early_stop_counter+=1

                    all_callbacks.on_epoch_end(epoch, {'train_mse':tr_mse, 'train_rmse':tr_rmse, 'train_mae':tr_mae, 'val_mse':val_mse, 'val_rmse:':val_rmse, 'val_mae':val_mae, 'train_row_penalty':tr_re, 'train_column_penalty':tr_ce, 'val_row_penalty':val_re, 'val_column_penalty':val_ce})

                    if early_stop_counter==patience:
                        all_callbacks.on_train_end(logs)
                        gen_plots(train_metric, val_metric_list)
                        return {"score": best_val_loss}
                    else:
                        epoch+=1
        all_callbacks.on_train_end(logs)
        gen_plots(train_metric, val_metric_list)

        textlist.append('----EXP OVER----')
        printlock.acquire()
        try:
            print(textlist, flush=True)
            gen_plots(textlist, specialindex, fullpath)
            newfile= open(fullpath+'/out.txt', 'w')
            for x in textlist:
                newfile.write(x)

        finally:
            printlock.release()


def gen_plots(textlist, specialindex, fullpath):
    print('plots pls')
    plt.figure()
    train_metric=[]
    val_metric=[]
    test_el=''
    val_el=''
    lr=''
    bs=''
    dr_rates=''
    for line_num in range(len(textlist)):
        if 'train mse' in textlist[line_num]:
            value=float(textlist[line_num].split('(')[1].split(',')[0])
            train_metric.append(value)
        elif 'val loss and mse' in textlist[line_num]:
            val_metric.append(float(textlist[line_num+2]))
        elif 'dropouts' in  textlist[line_num]:
            dr_rates=textlist[line_num]
        elif 'lr=' in textlist[line_num]:
            lr=textlist[line_num]
        elif 'bs=' in textlist[line_num]:
            bs=textlist[line_num]
        elif 'test element' in textlist[line_num]:
            test_el= textlist[line_num]
        elif 'val element' in textlist[line_num]:
            val_el= textlist[line_num]
        elif 'evaluation on train set before training:' in textlist[line_num]:
            value=float(textlist[line_num+1][1:-1].split(' ')[0])
            train_metric.append(value)
        elif 'evaluation on val set before training:' in textlist[line_num]:
            value=float(textlist[line_num+1][1:-1].split(' ')[0])
            val_metric.append(value)
    epochs=list(range(len(train_metric)))
    min_train= 'train min='+str(np.round(np.min(train_metric), decimals=3))+','
    min_val= 'val min='+str(np.round(np.min(val_metric), decimals=3))
    plt.plot(epochs, np.log(train_metric), label='training loss')
    plt.plot(epochs, np.log(val_metric), label='val loss')

    titleline= test_el+', '+val_el+', '+lr+', '+bs+', '+dr_rates
    figtitle=fullpath+'/result.png'

    plt.title(titleline, wrap=True)
    plt.xlabel('epochs')
    plt.legend()
    plt.ylabel('log of mean square error ')
    plt.savefig(figtitle)


def split_for_mp(data):
    elements_in_formation_energy_set={9: 5086, 90: 345, 57: 1393, 60: 893, 65: 592, 70: 581, 67: 684, 62: 808, 38: 1657, 56: 2332, 69: 522, 39: 1066, 68: 666, 19: 2468, 89: 76, 59: 757, 8: 26390, 71: 496, 37: 1417, 58: 705, 21: 701, 40: 938, 20: 1670, 11: 3185, 64: 330, 91: 77, 72: 584, 13: 1958, 17: 1849, 63: 274, 92: 686, 3: 10455, 12: 1428, 66: 670, 22: 2164, 94: 78, 82: 932, 14: 3238, 4: 398, 73: 757, 55: 1070, 5: 2448, 93: 112, 61: 120, 23: 3528, 34: 1668, 41: 1358, 74: 939, 24: 2185, 7: 2124, 49: 1242, 83: 1413, 31: 1156, 35: 827, 25: 4209, 6: 2146, 15: 6782, 75: 426, 52: 1493, 26: 4206, 30: 1354, 16: 3271, 81: 943, 79: 796, 50: 1810, 27: 2786, 28: 2630, 1: 3248, 29: 2754, 32: 1605, 48: 878, 51: 1823, 46: 917, 47: 1149, 42: 965, 45: 789, 33: 1337, 53: 987, 44: 701, 76: 315, 80: 698, 77: 593, 78: 728, 43: 169, 2: 2, 54: 46, 36: 8}
    data_tr = []
    data_va = []
    data_te = []
    data_ex= []
    test_element=[]
    val_element=[]
    sum=0
    while sum<7000:
        element, quant= elements_in_formation_energy_set.popitem()
        if quant+sum<9000:
            test_element.append(element)
            sum+=quant

    sum=0
    while sum<7000:
        element, quant= elements_in_formation_energy_set.popitem()
        if quant+sum<9000:
            val_element.append(element)
            sum+=quant

    test_element=set(test_element)
    val_element=set(val_element)
    for d in data:
        atomset= set(d._atomlist)
        if (atomset & test_element):
            if (atomset & val_element):
                data_ex.append(d._cif)
            else:
                data_te.append(d)
        elif (atomset & val_element):
            data_va.append(d)
        else:
            data_tr.append(d)

    return data_tr, data_va, data_te, data_ex, test_element, val_element

def split_for_prashuns_data(data, test_element, val_element):
    data_tr=[]
    data_va=[]
    data_te=[]
    data_ex=[]
    for d in data:
        atomset= set(d._atomlist)
        if test_element in atomset:
            if val_element in atomset:
                data_ex.append(d._cif)
            else:
                data_te.append(d)
        elif val_element in atomset:
            data_va.append(d)
        else:
            data_tr.append(d)
    #print(data_tr)
    return data_tr, data_va, data_te, data_ex


def lhs(data, printlock):
    atomic_num_list=[33, 83, 51]
    # sampler = qmc.LatinHypercube(d=7)
    # quantity=2
    # sample = sampler.random(n=quantity)
    #
    # #bs, lr, dr1a, dr2, el, cl, dr1b
    # l_bounds= [2, 0, 0, 0, 0, 0, 0]#, 0, 0, 0]
    # u_bounds= [8, 5, 1, 1, 4, 4, 1]#, 4, 4, 4]
    # scaled_sample= qmc.scale(sample, l_bounds, u_bounds)

    # parameter_sets= []
    #
    # for i in range(len(atomic_num_list)):
    #     te=atomic_num_list[i]
    #     for j in range(len(atomic_num_list)):
    #         va=atomic_num_list[j]
    #         if te!=va:
    #             for row in scaled_sample:
    #                     row=list(row)
    #                     row[0]=int(2**np.ceil(row[0]))
    #                     row[1]= 10**(-1*row[1])
    #                     row[4]= 10**row[4]
    #                     row[5]= 10**row[5]
    #
    #
    #                     param_set= [te, va]+ row
    #                     print(param_set)
    #                     parameter_sets.append(param_set)

    manager = Manager()
    performance_dict= manager.dict()
    model_dict = manager.dict()
    parameter_sets= [[33,83,0.001,32,52,23],
                    [33,51,0.001,32,52,23],
                    [83,33,0.001,32,52,23],
                    [83,51,0.001,32,52,23],
                    [51,83,0.001,32,52,23],
                    [51,33,0.001,32,52,23]]
    #training and validation
    for i in range(len(parameter_sets)):
        current_params=parameter_sets[i]
        test_element= current_params[0]
        val_element= current_params[1]
        lr= current_params[2]
        bs= current_params[3]

        if args.dataset=='prashun':
            data_tr, data_va, data_te, data_ex= split_for_prashuns_data(data, test_element, val_element)
        else:
            data_tr, data_va, data_te, data_ex= split_for_mp(data, test_element, val_element)
        textlist=[]
        textlist.append('model # '+str(i))
        textlist.append('test element='+str(test_element))
        textlist.append('test size='+str(len(data_te)))
        textlist.append('val element='+str(val_element))
        textlist.append('val size='+str(len(data_va)))
        textlist.append('train size='+str(len(data_tr)))
        textlist.append('excluded to prevent data leakage:')
        textlist.append(data_ex)
        textlist.append('lr='+str(lr))
        textlist.append('bs='+str(bs))
        # textlist.append('dropouts='+str(d1)+','+str(d2))
        # textlist.append('entropy lambda='+str(el))
        # textlist.append('column lambda='+str(cl))

        loader_tr = DisjointLoader(PartitionedData(data_tr), batch_size=bs, epochs=args.epochs)
        loader_va = DisjointLoader(PartitionedData(data_va), batch_size=len(data_va))
        loader_te = DisjointLoader(PartitionedData(data_te), batch_size=len(data_te))

        p= Process(target=full_training_loop, args=(printlock, loader_tr, loader_va, loader_te, textlist, test_element, val_element, lr, i, model_dict, performance_dict))#, r1, r2, r3))

        processlist.append(p)

    for pr in processlist:
        pr.start()
        print(pr, ' started', flush=True)
    for pr in processlist:
        pr.join()
        print(pr)
        print('complete')

    print(model_dict)
    print(performance_dict)
    print('---')

def random_split(dataset):
    data_tr=[]
    data_va=[]
    data_te=[]
    split= int(len(dataset)/5)
    split_2= split*2
    #print(split)
    data_te=dataset[:split]
    data_va= dataset[split:split_2]
    data_tr=dataset[split_2:len(dataset)]
    print(len(data_te))
    print(len(data_va))
    print(len(data_tr))
    loader_tr = DisjointLoader(PartitionedData(data_tr), batch_size=32, epochs=args.epochs)
    loader_va = DisjointLoader(PartitionedData(data_va), batch_size=len(data_va))
    loader_te = DisjointLoader(PartitionedData(data_te), batch_size=len(data_te))
    return loader_tr, loader_va, loader_te

if __name__ == '__main__':
    printlock= Lock()
    
    args = parser.parse_args(sys.argv[1:])

    np.random.seed(args.random_seed)
    if not os.path.exists(args.path+'/'+args.file_out):
        os.makedirs(args.path+'/'+args.file_out)
    #sys.stdout = open(args.path+'/'+args.file_out+'/'+args.file_out+'.txt', 'w')

    print(args)
    paramdict={
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
    df = pd.read_csv(os.path.join(args.datadir,'train_with_counts_complete.csv'))
    df = df[df['num_elements']<=3]

    train_data= DisjointLoader(MyDataset(df, args.datadir, args.radius_angstroms, args.num_nbrs, args.task), batch_size=paramdict['batch_size'], epochs=args.epochs)
    val_df = pd.read_csv(os.path.join(args.datadir,'val_with_counts_complete.csv'))
    val_df = val_df[val_df['num_elements']<=3]
    #val_df= val_df.head(10)
    val_data= DisjointLoader(MyDataset(val_df, args.datadir, args.radius_angstroms, args.num_nbrs, args.task), batch_size=len(val_df))

    test_df = pd.read_csv(os.path.join(args.datadir,'test_with_counts_complete.csv'))
    test_df = test_df[test_df['num_elements']<=3]
    #test_df= test_df.head(20)
    test_data= DisjointLoader(MyDataset(test_df, args.datadir, args.radius_angstroms, args.num_nbrs, args.task), batch_size=len(test_df))

    full_training_loop(printlock, train_data, val_data, test_data, [], '', '', lr=paramdict['lr'], 0, el=paramdict['entropy_lambda'], cl=paramdict['column_lambda'], {}, {})#, r1, r2, r3))
