#from spektral.data import Graph, Dataset, DisjointLoader
#from spektral.data.utils import to_batch
#from spektral.layers import CrystalConv, DiffPool, ops, GlobalMaxPool#, Disjoint2Batch
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import MeanSquaredError, SparseCategoricalCrossentropy
from tensorflow.keras.metrics import sparse_categorical_accuracy, mean_squared_error
import numpy as np
import pandas as pd
import os
import sys
#from pymatgen.core.structure import Structure
#import json
import argparse
import time
#from spektral_essential_objects import GaussianDistance,MyDataset,HNetSimple,PartitionedData
#from sklearn.cluster import KMeans
#from sklearn.decomposition import PCA
#import matplotlib.pyplot as plt
#from scipy.spatial import distance
#from pymatgen.core.structure import Structure


# begin_time = time.time()
# parser = argparse.ArgumentParser(description='crystal hierarchy arguments.')
#
# parser.add_argument('--datadir', dest='datadir',
#         help='Directory where dataset is located', default='../cgcnn-pretrained-models/data/10atom_relaxed_cifs')
# parser.add_argument('--ckpt-path', dest='ckpt_path',
#                     help='checkpoint_path path', default='./latin4/debugging3/')
# parser.add_argument('--filename', dest='filename',
#                     help='csv where data is located', default='id_mini.csv')
# parser.add_argument('--file-out', dest='file_out',
#                     help='output txt file name', default='predscriptout.txt')
# parser.add_argument('--path-out', dest='path',
#                     help='output path', default='./debugging_/')
# parser.add_argument('--num-atoms', dest='num_atoms', type=int,
#                     help='Maximum number of nodes', default=10)
# parser.add_argument('--num-nbrs', dest='num_nbrs', type=int,
#                     help='num neighbors per atom', default=12)
#
# parser.add_argument('--num-classes', dest='num_classes', type=int,
#                     help='Number of label classes', default=3)
#
# parser.add_argument('--radius-angstroms', dest='radius_angstroms', type=int,
#                     help='search radius for neighbors', default=8)
# parser.add_argument('--random-seed', dest='random_seed', type=int,
#                     help='random seed for numpy', default=0)
# parser.add_argument('--batch-size', dest='batch_size', type=int,
#                     help='Batch size.', default=256)
#
# parser.add_argument('--task', choices=['r', 'c'],
#                     default='r', help='complete a regression or '
#                         'classification task (default: regression)')

# def performance_dict_extractor(filename):
#     f= open(filename)
#     f= f.readlines()
#     grand_performance_dict={}
#     path_dict={}
#     performance_dict={}
#     for i in range(len(f)):
#         line= f[i]
#         if '{' in line:
#             line = line.strip()
#             line = line[1:-1]
#             pairs = line.split(', ')
#             for pair in pairs:
#                 pair=pair.split(':')
#                 pair[0]=int(pair[0])
#                 pair[1]= pair[1].strip()
#                 try:
#                     performance_dict[pair[0]]=float(pair[1])
#                 except:
#                     path_dict[pair[0]]=pair[1][1:-1]
#         if path_dict and performance_dict:
#             new_key= path_dict[pair[0]].split('/')[0]
#             grand_performance_dict[new_key]=performance_dict
#             path_dict={}
#             performance_dict={}
#     return grand_performance_dict
#

# def long_file_extractor(filename):
#     f= open(filename)
#     f= f.readlines()
#     grand_model_dict={}
#     grand_params_dict={}
#     for i in range(len(f)):
#         line=f[i]
#         if 'lhs_relu' in line:
#             line=line.split('/')
#             if len(line)==3:
#                 temp=f[i+2].split("', '")
#                 print()
# def evaluate(loader, model, ciflist):
#     output = []
#     step = 0
#
#     while step < loader.steps_per_epoch:
#         print(ciflist[step])
#         step += 1
#         inputs, target = loader.__next__()
#
#         pred, s_tensor = model(inputs, training=False)
#
#         print(s_tensor)
#         if args.task=='c':
#             outs = tf.reduce_mean(sparse_categorical_accuracy(target, pred))
#
#         elif args.task=='r':
#             outs = tf.reduce_mean(mean_squared_error(target, pred))
#             #print(target, pred)
#         output.append(outs)
#         if step == loader.steps_per_epoch:
#             output = np.array(output)
#             return np.average(output)#, pred
#
# def split_for_prashuns_data(data, test_element, val_element):
#     data_tr=[]
#     data_va=[]
#     data_te=[]
#     data_ex=[]
#     for d in data:
#         atomset= set(d._atomlist)
#         if test_element in atomset:
#             if val_element in atomset:
#                 data_ex.append(d._cif)
#             else:
#                 data_te.append(d)
#         elif val_element in atomset:
#             data_va.append(d)
#         else:
#             data_tr.append(d)
#
#     return data_tr, data_va, data_te, data_ex
#
# args = parser.parse_args(sys.argv[1:])
def file_extractor(filename):
    f= open(filename)
    f= f.readlines()

    model_dict={}
    params_dict={}
    val_element_dict={}
    path_dict={}
    performance_dict={}

    for line in f:
        if 'model #' in line:
            temp=line.split("', '")
            model_num= None
            lr= None
            bs= None
            for item in temp:
                if 'model #' in item:
                    model_num= item.split('# ')[-1]
                    model_num= int(model_num)
                if 'val element' in item:
                    val_element_dict[model_num]=item.split('=')
                if 'lr=' in item:
                    lr=item.split('=')[-1]
                    lr=float(lr)
                if 'bs' in item:
                    bs= int(item.split('=')[-1])
            tuple_key= (lr, bs)
            if tuple_key not in model_dict:
                model_dict[tuple_key]= [model_num]
            else:
                model_dict[tuple_key].append(model_num)

            params_dict[model_num]=temp[:14]
        if '{' in line:
            line = line.strip()
            line = line[1:-1]
            pairs = line.split(', ')
            for pair in pairs:
                pair=pair.split(':')
                pair[0]=int(pair[0])
                pair[1]= pair[1].strip()
                try:
                    performance_dict[pair[0]]=float(pair[1])
                except:
                    path_dict[pair[0]]=pair[1][1:-1]

    keys=model_dict.keys()
    mses=[]
    main_perf=[]
    vals_all=[]
    lr=[]
    bs=[]
    dr1=[]
    dr2=[]
    dr3=[]
    el=[]
    cl=[]
    l2_1=[]
    l2_2=[]
    l2_3=[]
    decay_rate=[]
    decay_steps=[]
    for key in keys:
        model_nums=model_dict[key]
        performances=[]
        vals=[]
        for model_num in model_nums:
            performances.append(performance_dict[model_num])
            vals.append((model_num, val_element_dict[model_num]))
        relevant_params= params_dict[model_num]
        #print(relevant_params)
        #print('---')
        main_perf.append(np.mean(performances))
        vals_all.append(vals)
        lr.append(float(relevant_params[6].split('=')[-1]))
        bs.append(int(relevant_params[7].split('=')[-1]))
        drs= relevant_params[8].split('=')[-1].split(',')
        dr1.append(float(drs[0]))
        dr2.append(float(drs[1]))
        dr3.append(float(drs[2]))
        el.append(float(relevant_params[9].split('=')[-1]))
        cl.append(float(relevant_params[10].split('=')[-1]))
        l2s=relevant_params[11].split('=')[-1].split(',')
        l2_1.append(float(l2s[0]))
        l2_2.append(float(l2s[1]))
        l2_3.append(float(l2s[2]))
        decay_rate.append(float(relevant_params[12].split('=')[-1]))
        decay_steps.append(float(relevant_params[13].split('=')[-1]))
    # #     #print(l2s)
    # #     #print('---')
    # #
    df = pd.DataFrame({'val_mse_mean':main_perf})
    df['val_folds']= vals_all
    df['lr']=lr
    df['bs']=bs
    df['dr1']=dr1
    df['dr2']=dr2
    df['dr3']=dr3
    df['el']=el
    df['cl']=cl
    df['l2_1']=l2_1
    df['l2_2']=l2_2
    df['l2_3']=l2_3
    df['decay_rate']=decay_rate
    df['decay_steps']=decay_steps
    namelist=[filename]*len(lr)
    df['path']=namelist
    df.to_csv('val_var_siamese.csv', mode='a', header=True, index=False)
    #return model_dict, params_dict, performance_dict, val_element_dict
#
file_extractor('./var_lr_siamese/lhs_siamese0/debugging/debugging.txt')
file_extractor('./var_lr_siamese/lhs_siamese1/debugging/debugging.txt')
file_extractor('./var_lr_siamese/lhs_siamese2/debugging/debugging.txt')
file_extractor('./var_lr_siamese/lhs_siamese3/debugging/debugging.txt')
file_extractor('./var_lr_siamese/lhs_siamese4/debugging/debugging.txt')
file_extractor('./var_lr_siamese/lhs_siamese5/debugging/debugging.txt')
file_extractor('./var_lr_siamese/lhs_siamese6/debugging/debugging.txt')
file_extractor('./var_lr_siamese/lhs_siamese7/debugging/debugging.txt')
file_extractor('./var_lr_siamese/lhs_siamese8/debugging/debugging.txt')
file_extractor('./var_lr_siamese/lhs_siamese9/debugging/debugging.txt')
file_extractor('./var_lr_siamese/lhs_siamese10/debugging/debugging.txt')
file_extractor('./var_lr_siamese/lhs_siamese11/debugging/debugging.txt')
file_extractor('./var_lr_siamese/lhs_siamese12/debugging/debugging.txt')
file_extractor('./var_lr_siamese/lhs_siamese13/debugging/debugging.txt')
file_extractor('./var_lr_siamese/lhs_siamese14/debugging/debugging.txt')
file_extractor('./var_lr_siamese/lhs_siamese15/debugging/debugging.txt')
file_extractor('./var_lr_siamese/lhs_siamese16/debugging/debugging.txt')
file_extractor('./var_lr_siamese/lhs_siamese17/debugging/debugging.txt')
file_extractor('./var_lr_siamese/lhs_siamese18/debugging/debugging.txt')
file_extractor('./var_lr_siamese/lhs_siamese19/debugging/debugging.txt')
file_extractor('./var_lr_siamese/lhs_siamese20/debugging/debugging.txt')
file_extractor('./var_lr_siamese/lhs_siamese21/debugging/debugging.txt')
file_extractor('./var_lr_siamese/lhs_siamese22/debugging/debugging.txt')
file_extractor('./var_lr_siamese/lhs_siamese23/debugging/debugging.txt')
file_extractor('./var_lr_siamese/lhs_siamese24/debugging/debugging.txt')
file_extractor('./var_lr_siamese/lhs_siamese25/debugging/debugging.txt')
file_extractor('./var_lr_siamese/lhs_siamese26/debugging/debugging.txt')
file_extractor('./var_lr_siamese/lhs_siamese27/debugging/debugging.txt')
file_extractor('./var_lr_siamese/lhs_siamese28/debugging/debugging.txt')
file_extractor('./var_lr_siamese/lhs_siamese29/debugging/debugging.txt')
file_extractor('./var_lr_siamese/lhs_siamese30/debugging/debugging.txt')
file_extractor('./var_lr_siamese/lhs_siamese31/debugging/debugging.txt')
file_extractor('./var_lr_siamese/lhs_siamese32/debugging/debugging.txt')
file_extractor('./var_lr_siamese/lhs_siamese33/debugging/debugging.txt')
file_extractor('./var_lr_siamese/lhs_siamese34/debugging/debugging.txt')
file_extractor('./var_lr_siamese/lhs_siamese35/debugging/debugging.txt')
