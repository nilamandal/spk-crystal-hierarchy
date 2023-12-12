import pandas as pd
from pymatgen.core.structure import Structure
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

crazylist=['./group1/', './group2/']

for path in crazylist:
    df_getpaths= pd.read_csv(path+'evaluated_results.csv')
    getpaths= list(df_getpaths['path'])
    for subpath in getpaths:
        fullpath=path+subpath+'/'
        df=pd.read_csv(fullpath+'pooling_eval.csv')

#df_master_dict={}
#crazylist=crazylist[:2]
for path in crazylist:

    #print(path)
    fullpath='./train_model_2023-10-27_15-40-05/'+path+'/'


    print(path, len(df))
    # print(df)
    # plt.figure()
    # df_0=df[df['perfect']==0]
    # df_1=df[df['perfect']==1]
    # plt.scatter(df_0['row_entropy'], df_0['neg_col_entropy '], c=df_0['avg_acc'], label='other')
    # plt.colorbar()
    #
    # #plt.scatter(df_1['row_entropy'], df_1['neg_col_entropy '], c=df_1['avg_acc'], label='perfect pools')
    # plt.xlabel('row entropy (desired value is 0)')
    # plt.ylabel('negative column entropy (desired value is -0.69)')
    # plt.legend()
    # plt.savefig(fullpath+'assignment_re_cp.png')
    # #df_metrics.replace({'tf.tensor(': ''}, regex=True)
    # #print(df_metrics)
