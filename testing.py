import numpy as np
import pandas as pd
import os
import sys
import argparse
#import time
import matplotlib.pyplot as plt
#from sklearn.manifold import TSNE
import seaborn as sns
val_set=[32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800, 832, 864, 896, 928, 960, 992, 1024, 1056, 1088, 1120, 1152, 1184, 1216, 1248, 1280, 1312, 1344, 1376, 1408, 1440, 1472, 1504, 1536, 1568, 1600, 1632, 1664, 1696, 1728, 1760, 1792, 1824, 1856, 1888, 1920, 1952, 1984, 2016, 2048, 2080, 2112, 2144, 2176, 2208, 2240, 2272, 2304, 2336, 2368, 2400, 2432, 2464, 2496, 2528, 2560, 2592, 2624, 2656, 2688, 2720, 2752, 2784, 2816, 2848, 2880, 2912, 2944, 2976, 3008, 3040]
v=[]
x = np.load('./dj_normalize_by_pool/doublejanossy/1/s_after_normalizer_3040.npz')
y = np.load('./dj_normalize_by_pool/doublejanossy/1/s_after_softmax_3040.npz')
mask= x['s']
s= y['s']
for i in range(len(s)):
    sub= s[i]
    sub_mask= mask[i]
    #print(sub)
    num= int(np.count_nonzero(sub_mask)/2)

    sub= sub[:num]
    print(sub)
    print(num)
    plt.figure()
    plt.scatter(sub[:,0],sub[:,1])
    plt.title('Crystal assignments (after softmax)')
    plt.xlabel('P1 assignment value; range [0,1]')
    plt.ylabel('P2 assignment value; range [0,1]')
    plt.savefig('./examples/'+str(i)+'.png')
# for i in range(1,3041):
#     x = np.load('./dj_normalize_by_pool/doublejanossy/1/s_after_normalizer_'+str(i)+'.npz')
#     x= x['s']
#     print(x.shape)
#     if x.shape[0]==312:
#             v.append(i)




# print(v)
#     x= x['x']
# for i in range(1,3361):
#     x = np.load('./batchnorm_npz/doublejanossy/1/x_after_dropout'+str(i)+'.npz')
#     x= x['x']
#     print(x.shape)
#     if x.shape[0]==311:
#         v.append(i)
# print(v)
# for j in val_set:
#     idx= str(j)
#     # x1= np.load('./tanhversion/debug_with_npz/doublejanossy/6/x_before_bn'+idx+'.npz')
#     # disjoint_x1= x1['x']
#     #
#     # x2= np.load('./tanhversion/debug_with_npz/doublejanossy/6/x_after_dropout'+idx+'.npz')
#     # batch_x2= x2['x']
#     #
#     # s1 = np.load('./tanhversion/debug_with_npz/doublejanossy/6/s_after_fc_'+idx+'.npz')
#     # batch_s1=s1['s']
#
#
#     for i in range(312):
#

    # s2 = np.load('./tanhversion/debug_with_npz/doublejanossy/6/s_after_softmax_30.npz')
    # batch_s2=s2['s']
    # print(np.shape(batch_s2))
    # s2 = np.load('./tanhversion/debug_with_npz/doublejanossy/6/s_after_softmax_31.npz')
    # batch_s2=s2['s']
    # print(np.shape(batch_s2))
    # s2 = np.load('./tanhversion/debug_with_npz/doublejanossy/6/s_after_softmax_40.npz')
    # batch_s2=s2['s']
    # print(np.shape(batch_s2))
    # s2 = np.load('./tanhversion/debug_with_npz/doublejanossy/6/s_after_softmax_41.npz')
    # batch_s2=s2['s']
    # print(np.shape(batch_s2))


#     for i in range(len(batch_s1)):
#         crystal= batch_s1[i]
#         count_rows= int(np.count_nonzero(crystal)/2)
#         print(count_rows)
#         print(crystal[0:count_rows])
# #val_df = pd.read_csv('../Main_fol_Zintl/val_with_counts_complete.csv', header=0)
#val_df = val_df[val_df['num_elements']<=3]
#print(val_df)

#contcars= list(val_df['id'])
#contcars= contcars[:2]

# for contcar in contcars:
#     filepath= '../Main_fol_Zintl/'+contcar+'_feats_after_cgcnn_model_f01f713f.csv'
#     df_feats= pd.read_csv(filepath)
#     df_feats= df_feats.drop('Unnamed: 0', axis=1)
#     df_feats=df_feats.loc[(df_feats != 0).any(1)]
#     #print(df_feats)
#     df_pools= pd.read_csv('../Main_fol_Zintl/'+contcar+'_pool_model_f01f713f.dat')
#     df_pools= df_pools.dropna()
#     #print(df_pools)
#     tsne = TSNE(n_components=2, random_state=0)
#     tsne_result = tsne.fit_transform(df_feats)
#     y= df_pools['ground_truth_P1']
#     tsne_result_df = pd.DataFrame({'tsne_1': tsne_result[:,0], 'tsne_2': tsne_result[:,1], 'label': y})
#
#     fig, ax = plt.subplots(1)
#     sns.scatterplot(x='tsne_1', y='tsne_2', hue='label', data=tsne_result_df, ax=ax,s=120)
#     lim = (tsne_result.min()-5, tsne_result.max()+5)
#     ax.set_xlim(lim)
#     ax.set_ylim(lim)
#     ax.set_aspect('equal')
#     ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
#     acc= np.mean(np.abs(1-df_pools['ground_truth_P1']-df_pools['SVM_pred_P1']))
#     print(acc)
#     plt.title(contcar+', acc='+str(acc))
#     plt.savefig('../Main_fol_Zintl/'+contcar+'_tsne__model_f01f713f.png')
#     #title should include structure name and pooling accuracy
