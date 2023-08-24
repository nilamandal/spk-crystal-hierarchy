import pandas as pd
import matplotlib.pyplot as plt
import umap
import seaborn as sns
import numpy as np

electroneg={'H':2.20, 'Li': 0.98, 'Na': 0.93, 'K': 0.82, 'Rb': 0.82, 'Cs': 0.79, 'Be': 1.57, 'Mg': 1.31, 'Ca': 1.0, 'Sr': 0.95, 'Ba': 0.89,
    'B': 2.04, 'C':2.55, 'N':3.04, 'O':3.44, 'F':3.98, 'Al':1.61, 'Si':1.90, 'P':2.19, 'S':2.58, 'Cl':3.16, 'Ga':1.81, 'Ge':2.01, 'As':2.18,
    'Se':2.55, 'Br':2.96, 'In':1.78, 'Sn':1.96, 'Sb':2.05, 'Te':2.1, 'I':2.66, 'Tl':1.62, 'Pb':1.87, 'Bi':2.02, 'Mn':1.55, 'Fe':1.83, 'Co': 1.88,
    'Ni':1.91, 'Cu':1.9, 'Zn':1.65, 'Yb':1.1, 'Cd':1.69, 'Hg':2.0}

df= pd.read_csv('./past_results/zintl_janossy_constant/train_model_a363460a_1_batch_size=8,column_lambda=2569099.8214,dr1=0.5332,embedding_size=12,entropy_lambda=6419201.9280,fc_size=14_2023-07-13_18-39-04/learned_reps_val_set.csv')
# #df=df[df['pool_num']==1]
df0=df[df['pool_num']==0]
#df1=df[df['pool_num']==1]
# for i in range(2,50):
plt.rcParams.update({'font.size': 20})
#     for j in (.1,.2,.3,.4,.5,.6,.7,.8,.9):
plt.figure()
reducer = umap.UMAP(random_state=42)
embedding=reducer.fit_transform(df0[['f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11']], df0['total_energy'])
df0['embedding0']=embedding[:, 0]
df0['embedding1']=embedding[:, 1]
df0.to_csv('umap_embeddings.csv')


# print(df0)
# print(df1)
# #     #print(embedding.shape)
plt.scatter(embedding[:, 0], embedding[:, 1], c=df0.total_energy, cmap='Spectral', s=50)
# # #     #plt.gca().set_aspect('equal', 'datalim')
plt.colorbar(boundaries=np.arange(-6,0), label= "total energy (eV/atom)")
# # plt.colorbar()
plt.title('UMAP projection of embeddings for both pools')
plt.xlabel('Component 1')
# plt.xlim(2,20)
plt.ylabel('Component 2')
# plt.ylim(-4,12)
plt.show()

# plt.figure()
# plt.scatter(df1['embedding0'], df1['embedding1'], c='#663fbe', s=10)
# # #     #plt.gca().set_aspect('equal', 'datalim')
# # plt.colorbar()
# plt.title('UMAP projection of embeddings for pool 1 only')
# plt.xlabel('Component 1')
# plt.xlim(2,20)
# plt.ylabel('Component 2')
# plt.ylim(-4,12)
# plt.show()
# #         plt.savefig('../learningtouseumap/batch7/'+str(i)+'_'+str(j)+'_chebyshev.png')
# #
# # def get_avg_electronegativity(cifname, pool_num):
# #     #print(cifname, pool_num)
# #     pool_df= pd.read_csv('../fromCW/bestmodel_poolings/'+cifname[:-7]+'pool.dat')
# #     pool_df= pool_df.dropna()
# #     elements= list(pool_df['species'])
# #     if pool_num==0:
# #         assign= list(pool_df['P1'])
# #     else:
# #         assign= list(pool_df['P2'])
# #     electro= [electroneg[i] for i in elements]
# #     mult = np.multiply(electro, assign)
# #     result= np.mean(mult)
# #
# #     return result
# def getlabelval(cifname, poolnum):
#     pool_df= pd.read_csv('../fromCW/bestmodel_poolings/'+cifname[:-7]+'pool.dat')
#     pool_df= pool_df.dropna()
#     print(pool_df.head(2))
#     assign= list(pool_df['P1'])
#     real= list(pool_df['ground_truth_P1'])
#     mult = np.multiply(real, assign)
#     result= np.mean(mult)
#     if poolnum==0:
#         return result
#     else:
#         return 1-result
# #
# df['labelval2']= df.apply(lambda x: getlabelval(x.cif, x.pool_num), axis=1)
# # df.to_csv('./zintl_janossy_constant/train_model_a363460a_1_batch_size=8,column_lambda=2569099.8214,dr1=0.5332,embedding_size=12,entropy_lambda=6419201.9280,fc_size=14_2023-07-13_18-39-04/learned_reps_val_set.csv')
