from spektral.data import Graph, Dataset, DisjointLoader
from spektral.data.utils import to_batch
from spektral.layers import CrystalConv, DiffPool, ops, GlobalMaxPool#, Disjoint2Batch
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import MeanSquaredError, SparseCategoricalCrossentropy
from tensorflow.keras.metrics import sparse_categorical_accuracy, mean_squared_error
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import os
import sys
from pymatgen.core.structure import Structure
import json
import argparse
import time
from spektral_essential_objects import AtomInitializer, GaussianDistance,AtomCustomJSONInitializer,MyDataset,HNet, PartitionedData
import threading
import concurrent.futures


#from spektral.datasets import QM9
begin_time = time.time()
parser = argparse.ArgumentParser(description='crystal hierarchy arguments.')

parser.add_argument('--datadir', dest='datadir',
        help='Directory where dataset is located', default='../crystalhierarchydata/formationcifs')

parser.add_argument('--filename', dest='filename',
                    help='csv where data is located', default='id_prop_10.csv')
parser.add_argument('--file-out', dest='file_out',
                    help='output file name', default='debugging')
parser.add_argument('--path-out', dest='path',
                    help='output path', default='./debugging/')
parser.add_argument('--num-atoms', dest='num_atoms', type=int,
                    help='Maximum number of nodes', default=200)
parser.add_argument('--num-nbrs', dest='num_nbrs', type=int,
                    help='num neighbors per atom', default=12)

parser.add_argument('--num-classes', dest='num_classes', type=int,
                    help='Number of label classes', default=4)

parser.add_argument('--radius-angstroms', dest='radius_angstroms', type=int,
                    help='search radius for neighbors', default=8)
parser.add_argument('--random-seed', dest='random_seed', type=int,
                    help='random seed for numpy', default=0)
parser.add_argument('--batch-size', dest='batch_size', type=int,
                    help='Batch size.', default=256)

parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run (default: 30)')

parser.add_argument('--lr', dest='learning_rate', type=float,
                    help='Learning rate.', default=1e-3)
parser.add_argument('--task', choices=['r', 'c'],
                    default='r', help='complete a regression or '
                        'classification task (default: regression)')
parser.add_argument('--patience', dest='patience',default=5, type=int,
                    help='num epochs for early stopping')
#parser.add_argument('--lam', dest='lam',default=0, type=float,
#                    help='lambda param for s penalty')


def evaluate(loader, model, loss_fn):
    #print('WE HAVE ENTERED THE FUNCTION')
    #print(args.task)
    output = []
    step = 0
    while step < loader.steps_per_epoch:
        step += 1
        inputs, target = loader.__next__()
        pred, s = model(inputs, training=False)
        if args.task=='c':
            outs = (
                loss_fn(target, pred),
                tf.reduce_mean(sparse_categorical_accuracy(target, pred)),
                len(target),  # Keep track of batch size
            )
            #print(step)
            #print(outs)
        elif args.task=='r':
            outs = (
                loss_fn(target, pred),
                tf.reduce_mean(mean_squared_error(target, pred)),
                len(target),  # Keep track of batch size
            )
        output.append(outs)
        if step == loader.steps_per_epoch:
            output = np.array(output)
            #print(output)
            return np.average(output[:, :-1], 0, weights=output[:, -1])

def train_step(inputs, target, model, loss_fn, optimizer):
    outputtxt=[]
    with tf.GradientTape() as tape:
        predictions, s = model(inputs, training=True)
        #s_penalty= tf.norm(tf.linalg.diag_part(tf.einsum('bij,bnm->bjm', s, s)), ord=np.inf)
        loss = loss_fn(target, predictions)
        #+ sum(model.losses) # + args.lam * s_penalty

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    if args.task=='r':
        mse = tf.reduce_mean((target-predictions)**2)
        return loss, mse, outputtxt
    if args.task=='c':
        sca= tf.reduce_mean(sparse_categorical_accuracy(target, predictions))
        outputtxt.append(confusion_matrix(target,np.argmax(predictions, axis=1)))

        return loss, sca, outputtxt

def full_training_loop(load_tr, load_va, load_te, textlist, testelement, valelement):
    init_time= time.time()
    checkpoint_path = args.path+"/"+args.file_out+"/"+args.file_out+str(testelement)+'-'+str(valelement)+".ckpt"
    optimizer = Adam(learning_rate=args.learning_rate)
    if args.task=='c':
        loss_fn= SparseCategoricalCrossentropy()
    elif args.task=='r':
        loss_fn = MeanSquaredError()
    else:
        print(args.task, ' is not c or r.')

    model= HNet(args.task, args.num_classes, return_s=True)

    early_stop_counter= 0

    epoch = step = 0
    best_val_loss = np.inf
    best_weights = None
    results = []

    for batch in loader_tr:
        step += 1
        textlist.append('epoch, step num:'+str(epoch)+','+str(step))

        loss, metric, outputtxt = train_step(*batch, model, loss_fn, optimizer)
        textlist= textlist + outputtxt
        if step == loader_tr.steps_per_epoch:
            step = 0
            loss_str="Loss: {}".format(loss / loader_tr.steps_per_epoch)
            textlist.append(loss_str)

            loss = 0
            val_loss, val_metric = evaluate(loader_va, model, loss_fn)
            #print('WE ARE OUT OF THE FUNCTION')
            #print(val_loss)
            if val_loss<best_val_loss:
                best_val_loss= val_loss
                early_stop_counter=0
            else:
                early_stop_counter+=1
            if args.task=='r':
                textlist.append('train mse='+str(metric))
                textlist.append('val loss and mse')
            elif args.task=='c':
                textlist.append('train accuracy='+str(metric))
                textlist.append('val loss and acc')
            textlist.append(str(val_loss))
            textlist.append(str(val_metric))

            #checkpoint_path = args.path+"/"+args.file_out+"/"+args.file_out+str(testelement)+".ckpt"
            checkpoint_dir = os.path.dirname(checkpoint_path)
            #print(epoch)
            # Create a callback that saves the model's weights
            model.save_weights(checkpoint_path.format(epoch=epoch))
            epoch+=1

            if early_stop_counter>5:
                test_loss, test_metric = evaluate(loader_te, model, loss_fn)
                textlist.append("Done. Test loss: {}".format(test_loss))
                if args.task=='r':
                    textlist.append('test mse=')
                elif args.task=='c':
                    textlist.append('test_acc=')
                textlist.append(test_metric)

                return model, textlist


    textlist.append('training time=')
    textlist.append(str(time.time()-init_time))

    test_loss, test_metric = evaluate(loader_te, model, loss_fn)
    textlist.append("Done. Test loss: {}".format(test_loss))
    if args.task=='r':
        textlist.append('test mse=')
    elif args.task=='c':
        textlist.append('test_acc=')
    textlist.append(test_metric)


    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights
    model.save_weights(checkpoint_path.format(epoch=args.epochs))
    return model, textlist


args = parser.parse_args(sys.argv[1:])

np.random.seed(args.random_seed)
if not os.path.exists(args.path+'/'+args.file_out):
    os.makedirs(args.path+'/'+args.file_out)
sys.stdout = open(args.path+'/'+args.file_out+'/'+args.file_out+'.txt', 'w')

print(args)


data= MyDataset(args.datadir,args.filename, args.radius_angstroms, args.num_atoms, args.num_nbrs, args.task)
datasettime=time.time()-begin_time
print('datset generated: time=', str(datasettime))
atomic_num_list=list(data.all_atomic_numbers)
np.random.shuffle(atomic_num_list)
#print('HELLO')


for i in range(len(atomic_num_list)):
    test_element=atomic_num_list[i]
    try:
        val_element=atomic_num_list[i+1]
    except:
        val_element=atomic_num_list[0]
    #print(test_element, val_element)
    data_tr = []
    data_va = []
    data_te = []
    data_ex= []
    for d in data:
        if test_element in d._atomlist and val_element in d._atomlist:
            data_ex.append(d._cif)
        elif test_element in d._atomlist:
             data_te.append(d)
        elif val_element in d._atomlist:
            data_va.append(d)
        else:
            data_tr.append(d)


    textlist=[]
    textlist.append('test element='+str(test_element))
    textlist.append('test size='+str(len(data_te)))
    textlist.append('val element='+str(val_element))
    textlist.append('val size='+str(len(data_va)))
    textlist.append('train size='+str(len(data_tr)))
    textlist.append('excluded to prevent data leakage:')
    textlist.append(data_ex)
    print(textlist)
        #
    loader_tr = DisjointLoader(PartitionedData(data_tr), batch_size=args.batch_size, epochs=args.epochs)
    loader_va = DisjointLoader(PartitionedData(data_va), batch_size=args.batch_size)
    loader_te = DisjointLoader(PartitionedData(data_te), batch_size=args.batch_size)
    #print(textlist)
        #
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(full_training_loop, loader_tr, loader_va, loader_te, textlist, test_element, val_element)
        model_result = future.result()
    print(model_result)
