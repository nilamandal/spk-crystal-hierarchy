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
from pymatgen.core.structure import Structure
import json
import argparse
import time
from spektral_essential_objects import AtomInitializer, GaussianDistance,AtomCustomJSONInitializer,MyDataset,HNetSimple,PartitionedData#, PreloadDataset
from multiprocessing import Process, Lock, Value, Manager

begin_time = time.time()
parser = argparse.ArgumentParser(description='crystal hierarchy arguments.')

parser.add_argument('--datadir', dest='datadir',
        help='Directory where dataset is located', default='../crystalhierarchydata/formationcifs')
parser.add_argument('--filename', dest='filename',
                    help='csv where data is located', default='id_prop.csv')
parser.add_argument('--file-out', dest='file_out',
                    help='output file name', default='debugging')
parser.add_argument('--path-out', dest='path',
                    help='output path', default='./debugging/')
parser.add_argument('--num-atoms', dest='num_atoms', type=int,
                    help='Maximum number of nodes', default=200)
parser.add_argument('--num-nbrs', dest='num_nbrs', type=int,
                    help='num neighbors per atom', default=12)
parser.add_argument('--num-classes', dest='num_classes', type=int,
                    help='Number of label classes', default=3)
parser.add_argument('--radius-angstroms', dest='radius_angstroms', type=int,
                    help='search radius for neighbors', default=10)
parser.add_argument('--random-seed', dest='random_seed', type=int,
                    help='random seed for numpy', default=5)
#parser.add_argument('--batch-size', dest='batch_size', type=int,
#                    help='Batch size.', default=256)
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run (default: 30)')
#parser.add_argument('--lr', dest='learning_rate', type=float,
#                    help='Learning rate.', default=1e-3)
parser.add_argument('--task', choices=['r', 'c'],
                    default='r', help='complete a regression or '
                        'classification task (default: regression)')
#parser.add_argument('--patience', dest='patience',default=30, type=int,
#                    help='num epochs for early stopping')
#parser.add_argument('--lam', dest='lam',default=0, type=float,
#                    help='lambda param for s penalty')

def test_eval(loader_te,model,loss_fn,textlist):
    test_loss, test_metric = evaluate(loader_te, model, loss_fn, True)
    textlist.append("Done. Test loss: {}".format(test_loss))
    if args.task=='r':
        textlist.append('test mse=')
    elif args.task=='c':
        textlist.append('test_acc=')
    textlist.append(test_metric)
    return textlist

def evaluate(loader, model, loss_fn, test=False):
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
        elif args.task=='r':
            outs = (
                loss_fn(target, pred),
                tf.reduce_mean(mean_squared_error(target, pred)),
                len(target),  # Keep track of batch size
            )
        output.append(outs)
        if step == loader.steps_per_epoch:
            if test==True:
                print('TEST confusion_matrix')
                print(confusion_matrix(target,np.argmax(pred, axis=1)))
            output = np.array(output)
            return np.average(output[:, :-1], 0, weights=output[:, -1])

def train_step(inputs, target, model, loss_fn, optimizer):
    outputtxt=[]
    with tf.GradientTape() as tape:
        predictions, s = model(inputs, training=True)
        #s_penalty= tf.norm(tf.linalg.diag_part(tf.einsum('bij,bnm->bjm', s, s)), ord=np.inf)
        loss = loss_fn(target, predictions)
        #+ sum(model.losses) # + args.lam * s_penalty
    #print(model.trainable_variables)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    if args.task=='r':
        mse = tf.reduce_mean((target-predictions)**2)
        return loss, mse, outputtxt
    if args.task=='c':
        sca= tf.reduce_mean(sparse_categorical_accuracy(target, predictions))
        outputtxt.append(confusion_matrix(target,np.argmax(predictions, axis=1)))

        return loss, sca, outputtxt

def full_training_loop(printlock, load_tr, load_va, load_te, textlist, testelement, valelement, lr, specialindex, model_list, performance_list, d1, d2, d3):
    #global model_list
    #global performance_list
    init_time= time.time()
    fullpath=args.path+'/'+args.file_out+'/'+str(specialindex)
    if not os.path.exists(fullpath):
        os.makedirs(fullpath)
    checkpoint_path = fullpath+"/"+args.file_out+str(testelement)+'-'+str(valelement)+'idx'+str(specialindex)+".ckpt"

    #boundaries=[int(args.epochs/3), int(2*args.epochs/3)]
    #values=[lr, lr/2, lr/4]
    #scheduler=PiecewiseConstantDecay(boundaries, values)
    #scheduler=ExponentialDecay(initial_learning_rate=lr, decay_steps=int(args.epochs/3), decay_rate=decay_rate)
    optimizer = Adam(learning_rate=lr)
    if args.task=='c':
        loss_fn= SparseCategoricalCrossentropy()
    elif args.task=='r':
        loss_fn = MeanSquaredError()
    else:
        print(args.task, ' is not c or r.')

    model= HNetSimple(args.task, args.num_classes, d1, d2, d3, return_s=True)

    textlist.append('evaluation on train set before training:')
    temp=evaluate(load_tr, model, loss_fn)
    textlist.append(temp)

    textlist.append('evaluation on val set before training:')
    temp=evaluate(load_va, model, loss_fn)
    textlist.append(temp)

    early_stop_counter= 0

    epoch = step = 0
    best_val_loss = np.inf
    best_weights = None
    results = []

    for batch in loader_tr:
        step += 1
        #textlist.append('epoch, step num:'+str(epoch)+','+str(step))

        loss, metric, outputtxt = train_step(*batch, model, loss_fn, optimizer)
        textlist= textlist + outputtxt
        if step == loader_tr.steps_per_epoch:
            step = 0
            loss_str="Loss: {}".format(loss / loader_tr.steps_per_epoch)
            textlist.append(loss_str)

            loss = 0
            val_loss, val_metric = evaluate(loader_va, model, loss_fn)
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

            try:
            # Create a callback that saves the model's weights
                model.save_weights(checkpoint_path)
                epoch+=1
                textlist.append('epoch='+str(epoch))
            except:
                textlist.append('saving error at '+str(epoch))
            #if early_stop_counter>args.patience:
                #textlist= test_eval(loader_te,model,loss_fn,textlist)
            #    return model, textlist


    textlist.append('training time=')
    textlist.append(str(time.time()-init_time))

    #textlist=

    #checkpoint_dir = os.path.dirname(checkpoint_path)
    try:
    # Create a callback that saves the model's weights
        model.save_weights(checkpoint_path)
    #model_placeholder.value= model
    #val_loss_placeholder.value= best_val_loss
        model_list[specialindex]= str(checkpoint_path)
        performance_list[specialindex] = best_val_loss
    except:
        textlist.append('error saving model #'+str(specialindex))
    printlock.acquire()
    try:
        print(textlist, flush=True)
    finally:
        printlock.release()

def test_val_split(data):
    elements_in_formation_energy_set={9: 5086, 90: 345, 57: 1393, 60: 893, 65: 592, 70: 581, 67: 684, 62: 808, 38: 1657, 56: 2332, 69: 522, 39: 1066, 68: 666, 19: 2468, 89: 76, 59: 757, 8: 26390, 71: 496, 37: 1417, 58: 705, 21: 701, 40: 938, 20: 1670, 11: 3185, 64: 330, 91: 77, 72: 584, 13: 1958, 17: 1849, 63: 274, 92: 686, 3: 10455, 12: 1428, 66: 670, 22: 2164, 94: 78, 82: 932, 14: 3238, 4: 398, 73: 757, 55: 1070, 5: 2448, 93: 112, 61: 120, 23: 3528, 34: 1668, 41: 1358, 74: 939, 24: 2185, 7: 2124, 49: 1242, 83: 1413, 31: 1156, 35: 827, 25: 4209, 6: 2146, 15: 6782, 75: 426, 52: 1493, 26: 4206, 30: 1354, 16: 3271, 81: 943, 79: 796, 50: 1810, 27: 2786, 28: 2630, 1: 3248, 29: 2754, 32: 1605, 48: 878, 51: 1823, 46: 917, 47: 1149, 42: 965, 45: 789, 33: 1337, 53: 987, 44: 701, 76: 315, 80: 698, 77: 593, 78: 728, 43: 169, 2: 2, 54: 46, 36: 8}
    data_tr = []
    data_va = []
    data_te = []
    data_ex= []
    test_element=[]
    #print(len(elements_in_formation_energy_set))
    val_element=[]
    sum=0
    while sum<7000:
        #print(sum)
        element, quant= elements_in_formation_energy_set.popitem()
        if quant+sum<9000:
            test_element.append(element)
            sum+=quant
        #else:
        #    elements_in_formation_energy_set[element]=quant
    #print(sum)
    sum=0
    while sum<7000:
        #print(sum)
        element, quant= elements_in_formation_energy_set.popitem()
        if quant+sum<9000:
            val_element.append(element)
            sum+=quant
        #else:
        #    elements_in_formation_energy_set[element]=quant
    #print(sum)
    print(test_element, len(test_element))
    print(val_element, len(val_element))
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

    print(len(data_tr))
    print(len(data_va))
    print(len(data_te))
    print(len(data_ex))

    return data_tr, data_va, data_te, data_ex, test_element, val_element

    #     if test_element in d._atomlist and val_element in d._atomlist:
    #
    #     elif test_element in d._atomlist:
    #
    #     elif val_element in d._atomlist:
    #         data_va.append(d)
    #     else:
    #         data_tr.append(d)
    # return

if __name__ == '__main__':
    printlock= Lock()
    processlist=[]
    args = parser.parse_args(sys.argv[1:])

    np.random.seed(args.random_seed)
    if not os.path.exists(args.path+'/'+args.file_out):
        os.makedirs(args.path+'/'+args.file_out)
    sys.stdout = open(args.path+'/'+args.file_out+'/'+args.file_out+'.txt', 'w')

    print(args)
    # df = pd.read_csv(os.path.join(args.datadir,args.filename), names=['id','target'], header=0)
    # cifs=list(df['id'])
    # allgraphs=[]
    # for c in cifs:
    #     c=str(c)
    #     graphdata= np.load(os.path.join(args.datadir,c+'.npz'))
    #     MG=Graph(x=graphdata['x'], a=graphdata['a'], e=graphdata['e'], y=graphdata['y'])
    #     MG._atomlist=graphdata['s']
    #     MG._cif=graphdata['c']
    #     allgraphs.append(MG)
    data= MyDataset(args.datadir,args.filename, args.radius_angstroms, args.num_nbrs, args.task)
    datasettime=time.time()-begin_time
    print('datset generated: time=', str(datasettime))
    #atomic_num_list=list(data.all_atomic_numbers)
    #np.random.shuffle(atomic_num_list)
    #atomic_num_list=[33, 83, 51, 15]
    lr_candidates=[1e-3,1e-4]
    batch_size_candidates=[16,32,64]
    droprate1=[0]
    droprate2=[0]
    droprate3=[0.1,0.2,0.3,0.4,0.5,0.6]
    #decay_rate=[0.9,0.8,0.7,0.6,0.5]

    parameter_sets= []

    #
    # for i in range(len(atomic_num_list)):
    #     test_element=atomic_num_list[i]
    #     try:
    #         val_element=atomic_num_list[i+1]
    #     except:
    #         val_element=atomic_num_list[0]
    for lr in lr_candidates:
            for bs in batch_size_candidates:
                for d1 in droprate1:
                    for d2 in droprate2:
                        for d3 in droprate3:
                            parameter_sets.append([lr, bs, d1, d2, d3])

    manager = Manager()
    performance_dict= manager.dict()
    model_dict = manager.dict()
    data_tr, data_va, data_te, data_ex, test_element, val_element = test_val_split(data)

    for i in range(len(parameter_sets)):
            current_params=parameter_sets[i]
            #test_element= current_params[0]
            #val_element= current_params[1]
            lr= current_params[0]
            bs= current_params[1]
            d1= current_params[2]
            d2= current_params[3]
            d3= current_params[4]
            #dr= current_params[4]


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
            textlist.append('dropouts='+str(d1)+','+str(d2)+','+str(d3))

            loader_tr = DisjointLoader(PartitionedData(data_tr), batch_size=bs, epochs=args.epochs)
            loader_va = DisjointLoader(PartitionedData(data_va), batch_size=len(data_va))
            loader_te = DisjointLoader(PartitionedData(data_te), batch_size=len(data_te))

            p= Process(target=full_training_loop, args=(printlock, loader_tr, loader_va, loader_te, textlist, test_element, val_element, lr, i, model_dict, performance_dict, d1, d2, d3))

            p.start()
            print(i, ' started', flush=True)
            processlist.append(p)

    for pr in processlist:
        pr.join()
        print(pr)
        print('complete')

# lr_candidates=[1e-3,1e-4]
# batch_size_candidates=[16,32,64]
# droprate1=[0,0.25, 0.5]
# droprate2=[0,0.25, 0.5]
# droprate3=[0,0.25, 0.5]
    print(model_dict)
    print(performance_dict)
    total= len(lr_candidates)*len(batch_size_candidates)*len(droprate1)*len(droprate2)*len(droprate3)
    chunks=[0,int(total/4),int(total/2),int(total*3/4)]

    for i in chunks:
        i2= i+int(total/4)
        current_params=parameter_sets[i]
        test_element= current_params[0]
        val_element= current_params[1]
        textlist=[]
        textlist.append('for test element='+str(test_element))
        tempkeys= list(performance_dict.keys())
        tempkeys= [k for k in tempkeys if (k>i and k<i2)]
        temp_dict={k: performance_dict[k] for k in tempkeys}
        bestmodelindex= min(temp_dict, key=temp_dict.get)


        checkpoint_dir = os.path.dirname(model_dict[bestmodelindex])

        bestmodel= HNetSimple(args.task, args.num_classes, return_s=True)

        latest = tf.train.latest_checkpoint(checkpoint_dir)
        bestmodel.load_weights(latest)

        print('model #', bestmodelindex, ' is the best model')
        print('EVAL ON TEST SET')
        textlist=test_eval(loader_te,bestmodel,SparseCategoricalCrossentropy(),[])
        print(textlist)
