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
    s_weights= np.load(filename)
    #fc_norms=np.linalg.norm(fc_weights, 2, axis=1)
    #print(fc_norms.shape)
    abs_diff= np.abs(s_weights[:,0]-s_weights[:,1])
    nodes_all= np.concatenate((s_weights[:,0],s_weights[:,1]))
    plt.figure()
    plt.hist(abs_diff, bins)
    plt.title('s weight abs_diff for bayes51_211_33_sweights')
    plt.xlabel('s weight abs_diff ('+str(bins)+' bins)')
    plt.ylabel('count')
    plt.savefig('./bayes51_211_33_sweights_diff_'+str(bins)+'bins.png')

    plt.figure()
    plt.hist(nodes_all, bins)
    plt.title('s-layer weights for bayes51_211_33')
    plt.xlabel('individual weights ('+str(bins)+' bins)')
    plt.ylabel('count')
    plt.savefig('./bayes51_211_33_s_'+str(bins)+'bins.png')


def ratio():
    df_cgcnn=pd.read_csv('../cgcnn-pretrained-models/test_results_final/test_results_from_0_83.csv', names=['cif', 'real', 'pred'])
    df_ours= pd.read_csv('../oldspk/4000_results/noz4_4_out_og.csv')
    df_cgcnn['cgcnndiff']=np.abs(df_cgcnn['real']-df_cgcnn['pred'])
    df_ours['oursdiff']=np.abs(df_ours['real']-df_ours['pred'])
    df_joined= pd.merge(df_ours, df_cgcnn, on='cif', how='inner')
    print(df_joined)
    plt.scatter(df_joined['oursdiff'],df_joined['cgcnndiff'])
    x=np.linspace(0,1,100)
    y=x
    plt.plot(x,y)
    plt.xlabel('error on our model')
    plt.ylabel('error on cgcnn')
    plt.show()
    #df_joined.to_csv('comparison.csv')

def threshold1():
    df=pd.read_csv('./comparisons_with_decomp.csv')
    decomp_threshold= 2000
    our_mse=1
    cgcnn_mse=0.5
    threshold_list=[]
    ours_list=[]
    cgcnn_list=[]
    while our_mse>cgcnn_mse:
        decomp_threshold-=.5
        df_threshold=df[df['Edecomp']<=decomp_threshold]
        df_threshold['temp']=(df_threshold['real_y']-df_threshold['pred_y'])**2
        our_mse=np.mean(df_threshold['sq error'])
        cgcnn_mse=np.mean(df_threshold['temp'])
        print(decomp_threshold, our_mse, cgcnn_mse)
        threshold_list.append(decomp_threshold)
        ours_list.append(our_mse)
        cgcnn_list.append(cgcnn_mse)
    plt.plot(threshold_list,ours_list, label='our model')
    plt.plot(threshold_list,cgcnn_list, label='cgcnn')
    plt.xlabel('decomposition energy threshold')
    plt.ylabel('mean squared error on crystals below threshold')
    plt.title("Crystal decomposition energy vs prediction mse")
    plt.legend()
    plt.show()

def threshold2():
    df=pd.read_csv('./ext_with_correct_decomp.csv')
    #df=df[df['Edecomp (meV/atom)']>0]
    #df=df[df['Edecomp (meV/atom)']>=-500]
    #ax = plt.subplot()
    scatter= plt.scatter(df['diff1'],df['diff2'], c=df['Edecomp (meV/atom)'])
    plt.colorbar(scatter)

    x=np.linspace(0,5,100)
    y=x
    plt.plot(x,y)
    plt.title('Error for All Extrapolation Structures')
    plt.xlabel('absolute error on our model')
    plt.ylabel('absolute error on cgcnn')

    plt.show()
    # errorbar=0.1
    # while errorbar<10:
    #     df2=df[df['diff1']<=errorbar]
    #     df_ours= df2[df2['diff1']<df2['diff2']]
    #     df_theirs= df2[df2['diff1']>df2['diff2']]
    #     print(errorbar, len(df2),len(df_ours),len(df_theirs))
    #     errorbar+=.1

threshold2()

#
