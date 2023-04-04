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
import argparse
import time
import matplotlib.pyplot as plt

df= pd.DataFrame(columns=['te', 'va', 'lr', 'bs', 'dr1', 'dr2', 'el', 'cl', 'l2_1', 'l2_2', 'l2_3', 'model_num', 'path', 'val_error'])

def file_extractor(filename):
    f= open(filename)
    f= f.readlines()

    model_dict={}
    params_dict={}
    val_element_dict={}
    path_dict={}
    performance_dict={}
    for line in f:
        if 'testing time' in line:
                 break
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

    for line in f:
         if 'testing time' in line:
                 break
         if '----NEW EXP----' in line:
             temp=line.split("', '")
             expline=None
             model_num= None
             lr= None
             bs= None
             te= None
             va= None
             dr1= None
             dr2= None
             el= None
             cl= None
             l2_1= None
             l2_2= None
             l2_3= None
             for item in temp:
                 if 'model #' in item:
                     model_num= int(item.strip().split('#')[1])
                 if 'test element' in item:
                     te= int(item.strip().split('=')[1])
                     #print(te)
                 elif 'val element' in item:
                     va= int(item.strip().split('=')[1])
                 elif 'lr=' in item:
                     lr= float(item.strip().split('lr=')[1])
                     #print(lr)
                 elif 'bs=' in item:
                    bs= int(item.strip().split('=')[1])
                 elif 'dropouts' in item:
                    both=item.strip().split('=')[1].split(',')
                    dr1= float(both[0])
                    dr2= float(both[1])
                 elif 'entropy lambda' in item:
                    el= float(item.strip().split('=')[1])
                 elif 'column lambda' in item:
                     cl= float(item.strip().split('=')[1])
                 elif 'l2 feature reg hyperparams=' in item:
                     all= item.strip().split('=')[1].split(',')
                     l2_1= float(all[0])
                     l2_2= float(all[1])
                     l2_3= float(all[2])

             #print([te, va, lr, bs, dr1, dr2, el, cl, l2_1, l2_2, l2_3, 'model_num', 'path', 'val_error'])
             df.loc[len(df.index)] = [te, va, lr, bs, dr1, dr2, el, cl, l2_1, l2_2, l2_3, 'model_num', 'path', 'val_error']
             #print(model_num)
             df.loc[(df['te']==te) & (df['va']==va) & (np.round(df['lr'], 6)==np.round(lr,6)), 'model_num']= model_num
             df.loc[(df['te']==te) & (df['va']==va) & (np.round(df['lr'], 6)==np.round(lr,6)), 'path']= filename
             df.loc[(df['te']==te) & (df['va']==va) & (np.round(df['lr'], 6)==np.round(lr,6)), 'val_error']= performance_dict[model_num]

    #print(df)
def read_test(filename):
    f= open(filename)
    f= f.readlines()
    min= np.inf
    print(filename)
    for line in f:
        if 'model #' in line:
            min= np.inf
            temp=line.split("', '")
            for i in range(len(temp)):
                currentline= temp[i]
                if "val loss" in currentline:
                    val_mse=float(temp[i+1])
                    if val_mse<min:
                        min= val_mse
            print(min)

    #print(min)
            print('---')

def identify_best(csv_name, new_name):
    df= pd.read_csv(csv_name)
    te_keys=[33,51,83]
    lr_keys=np.unique(df['lr'].to_numpy())
    bs_keys=np.unique(df['bs'].to_numpy())
    df_new=pd.DataFrame(columns=['te', 'lr', 'bs', 'val_avg'])
    for te_key in te_keys:
         for lr_key in lr_keys:
             for bs_key in bs_keys:
                 df_temp=df[(df['te']==te_key) & (df['lr']==lr_key) &(df['bs']==bs_key)]
                 if len(df_temp)>0:
                      avg=np.mean(df[(df['te']==te_key) & (df['lr']==lr_key) &(df['bs']==bs_key)]['val_error'])
                      df_new.loc[len(df_new.index)]=[te_key, lr_key, bs_key, avg]
    df_newest=pd.merge(df, df_new, on=['te', 'lr', 'bs'], how='inner')
    print(df_newest)
    df_newest.to_csv(new_name)

def bin_histogram(filename, bins):
    fc_weights= np.load(filename)
    fc_norms=np.linalg.norm(fc_weights, 2, axis=1)
    print(fc_norms.shape)
    #abs_diff= np.abs(s_weights[:,0]-s_weights[:,1])
    #nodes_all= np.concatenate((s_weights[:,0],s_weights[:,1]))
    plt.figure()
    plt.hist(fc_norms, bins)
    plt.title('fc weight norms for noz4_1_4')
    plt.xlabel('fc weight norms ('+str(bins)+' bins)')
    plt.ylabel('count (out of 156)')
    plt.savefig('./noz4_1_4_fc_'+str(bins)+'bins.png')
    #
    # plt.figure()
    # plt.hist(nodes_all, bins)
    # plt.title('s-layer weights for noz4_1_4')
    # plt.xlabel('individual weights ('+str(bins)+' bins)')
    # plt.ylabel('count (out of 104)')
    # plt.savefig('./noz4_1_4_s_'+str(bins)+'bins.png')

bin_histogram('noz4_1_4_fc.npy', 5)
bin_histogram('noz4_1_4_fc.npy', 10)
bin_histogram('noz4_1_4_fc.npy', 15)
bin_histogram('noz4_1_4_fc.npy', 20)
#read_test('./noabsdiff_results/test_results/test83/debugging/debugging.txt')
# #read_test('./noabsdiff_results/test_results/test51/debugging/debugging.txt')
# file_list= ['','1','12','123','1234']
# for file in file_list:
#     try:
#            file_extractor('./noz_4th_batch/0adamredo'+str(file)+'/debugging/debugging.txt')
#            file_extractor('./noz_4th_batch/lhs_0adam'+str(file)+'/debugging/debugging.txt')
#     except:
#         print(file)
# df.to_csv('noz_sweep4.csv')
# identify_best('adam_scheduler.csv', 'adam_scheduler_valavgs.csv')
