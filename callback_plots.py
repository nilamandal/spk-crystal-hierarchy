import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
crazylist=['../train_model_2023-12-05_10-16-27/']


for path in crazylist:
    df_getpaths= pd.read_csv(path+'evaluated_results.csv')
    getpaths= list(df_getpaths['path'])
    #getpaths=['0', '1','2','3','4']

    #'train_model_03ec7b5e_41_trial_index=0,batch_size=4,column_lambda=84.4158,dr1=0.7048,embedding_size=128,entropy_lambda=4971905.8384_2023-11-07_21-09-21']
    for subpath in getpaths:
        fullpath=path+str(subpath)+'/'
        #print(fullpath)

        df=pd.read_csv(fullpath+'callback_results.csv')
        paramsdict= json.load(open(fullpath+'params.json'))
        # paramsdict={
        #   "batch_size": 32,
        #   "column_lambda": 41903766.3588113,
        #   "dr1": 0.2513546780919437,
        #   "embedding_size": 4,
        #   "entropy_lambda": 108506.39801000628,
        #   "fc_num": 1,
        #   "fc_num2": 1,
        #   "fc_size": 4,
        #   "fc_size2": 4,
        #   "lr": 0.0631443016175717
        # }
        el=paramsdict['entropy_lambda']
        cl=paramsdict['column_lambda']
        plt.figure(figsize=(10,10))
        plt.plot(np.log10(-1*df['train_column_penalty']), label='train column product')
        plt.plot(np.log10(df['train_mse']), label='train_mse')
        #plt.plot(np.log10(df['train_mse']), label='train_mse')
        plt.plot(np.log10(df['train_row_penalty']), label='train_row_entropy')
        plt.plot(np.log10(-1*df['val_column_penalty']), label='val column product')
        plt.plot(np.log10(df['val_mse']), label='val_mse')
        plt.plot(np.log10(df['val_row_penalty']), label='val_row_entropy')
        plt.legend()
        plt.xlabel('training epochs')
        plt.ylabel('log10 values of loss function components')
        plt.title(subpath)
        plt.savefig(fullpath+'/allcomponents.png')
        plt.figure(figsize=(10,10))
        df['train_total']=df['train_mse']+(el*df['train_row_penalty'])+(cl*df['train_column_penalty'])
        df['val_total']=df['val_mse']+(el*df['val_row_penalty'])+(cl*df['val_column_penalty'])
        plt.plot(df['train_total'], label='train loss')
        plt.plot(df['val_total'], label='val loss')
        plt.legend()
        plt.xlabel('training epochs')
        plt.ylabel('sum of validation components')
        #plt.show()
        plt.savefig(fullpath+'/sum_losscomponents.png')
        #plt.plot(np.log10(df['val_row_penalty']), label='val_row_entropy')

def get_len(path):
    df= pd.read_csv('./train_model_2023-10-27_15-40-05/'+path+'/callback_results.csv')
    return len(df)

#maindf['num_epochs']= maindf['path'].apply(get_len)
#maindf.to_csv('./train_model_2023-10-27_15-40-05/evaluated_results2.csv')
#
#
#
#
#
#
#     plt.plot(np.log10(df['val_row_penalty']), label='val_row_entropy')
#
# #plt.plot(df['validation sum'], label='val loss')
#     plt.legend()
#     plt.xlabel('training epochs')
#     plt.ylabel('log10 values of loss function components')
#
#
