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
    #print(performance_dict)
    for line in f:
         if 'testing time' in line:
                 break
         if 'model #' in line:
             temp=line.split("', '")
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


             df.loc[len(df.index)] = [te, va, lr, bs, dr1, dr2, el, cl, l2_1, l2_2, l2_3, 'model_num', 'path', 'val_error']

             df.loc[(df['te']==te) & (df['va']==va) & (np.round(df['lr'], 6)==np.round(lr,6)), 'model_num']= model_num
             df.loc[(df['te']==te) & (df['va']==va) & (np.round(df['lr'], 6)==np.round(lr,6)), 'path']= filename
             df.loc[(df['te']==te) & (df['va']==va) & (np.round(df['lr'], 6)==np.round(lr,6)), 'val_error']= performance_dict[model_num]

    #print(df)
def read_test(filename):
    f= open(filename)
    f= f.readlines()
    min= np.inf
    for line in f:
        if 'model #' in line:
            #min= np.inf
            temp=line.split("', '")
            for i in range(len(temp)):
                currentline= temp[i]
                if "val loss" in currentline:
                    val_mse=float(temp[i+1])
                    if val_mse<min:
                        min= val_mse
            #print(min)
    print(filename)
    print(min)
    print('---')


read_test('./noabsdiff_results/test_results/test15/debugging/debugging.txt')
read_test('./noabsdiff_results/test_results/test33/debugging/debugging.txt')
read_test('./noabsdiff_results/test_results/test83/debugging/debugging.txt')
read_test('./noabsdiff_results/test_results/test51/debugging/debugging.txt')
# file_list= list(range(21))
#
# for file in file_list:
#      try:
#          file_extractor('./noabsdiff_results/10_lhs_'+str(file)+'/debugging/debugging.txt')
#          file_extractor('./noabsdiff_results/10_lhs_'+str(file)+'adam/debugging/debugging.txt')
#          file_extractor('./noabsdiff_results/lhs_'+str(file)+'/debugging/debugging.txt')
#          file_extractor('./noabsdiff_results/lhs_'+str(file)+'adam/debugging/debugging.txt')
#
#      except:
#          print(file)
# df.to_csv('noabsdiff_sweep.csv')
# df= pd.read_csv('./noabsdiff_sweep.csv')
# te_keys=[15,33,51,83]
# lr_keys=np.unique(df['lr'].to_numpy())
# bs_keys=np.unique(df['bs'].to_numpy())
# gooddict={}
# df_new=pd.DataFrame(columns=['te', 'lr', 'bs', 'val_avg'])
#print(te_keys)
#print(lr_keys)
#print(bs_keys)
# for te_key in te_keys:
#      for lr_key in lr_keys:
#          for bs_key in bs_keys:
#             #print(te_key, lr_key, bs_key)
#              df_temp=df[(df['te']==te_key) & (df['lr']==lr_key) &(df['bs']==bs_key)]
#              if len(df_temp)>0:
#                   avg=np.mean(df[(df['te']==te_key) & (df['lr']==lr_key) &(df['bs']==bs_key)]['val_error'])
#                   df_new.loc[len(df_new.index)]=[te_key, lr_key, bs_key, avg]
#
# #
# #             print()
# #
# #             #gooddict[(te_key,lr_key)]= avg
#         #print('---')
#
#
# #meankeeper=df.groupby(['te', 'lr', 'bs'], as_index=False).mean()
# print(df)
# print(df_new)
#
# df_newest=pd.merge(df, df_new, on=['te', 'lr', 'bs'], how='inner')
# print(df_newest)
# df_newest.to_csv('./noabsdiff_avgs.csv')
