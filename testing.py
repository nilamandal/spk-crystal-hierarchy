import numpy as np
import pandas as pd
import os
import sys
import argparse
import time
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns

val_df = pd.read_csv('../Main_fol_Zintl/val_with_counts_complete.csv', header=0)
val_df = val_df[val_df['num_elements']<=3]
#print(val_df)

contcars= list(val_df['id'])
#contcars= contcars[:2]

for contcar in contcars:
    filepath= '../Main_fol_Zintl/'+contcar+'_feats_after_cgcnn_model_f01f713f.csv'
    df_feats= pd.read_csv(filepath)
    df_feats= df_feats.drop('Unnamed: 0', axis=1)
    df_feats=df_feats.loc[(df_feats != 0).any(1)]
    #print(df_feats)
    df_pools= pd.read_csv('../Main_fol_Zintl/'+contcar+'_pool_model_f01f713f.dat')
    df_pools= df_pools.dropna()
    #print(df_pools)
    tsne = TSNE(n_components=2, random_state=0)
    tsne_result = tsne.fit_transform(df_feats)
    y= df_pools['ground_truth_P1']
    tsne_result_df = pd.DataFrame({'tsne_1': tsne_result[:,0], 'tsne_2': tsne_result[:,1], 'label': y})

    fig, ax = plt.subplots(1)
    sns.scatterplot(x='tsne_1', y='tsne_2', hue='label', data=tsne_result_df, ax=ax,s=120)
    lim = (tsne_result.min()-5, tsne_result.max()+5)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect('equal')
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    acc= np.mean(np.abs(1-df_pools['ground_truth_P1']-df_pools['SVM_pred_P1']))
    print(acc)
    plt.title(contcar+', acc='+str(acc))
    plt.savefig('../Main_fol_Zintl/'+contcar+'_tsne__model_f01f713f.png')
    #title should include structure name and pooling accuracy
