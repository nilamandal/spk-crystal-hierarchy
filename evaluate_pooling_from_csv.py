import pandas as pd
from pymatgen.core.structure import Structure
import numpy as np
import matplotlib.pyplot as plt

# def get_species(name):
#     crystal= Structure.from_file('../Main_fol_Zintl/'+name)
#     return crystal.species
#
def get_mean_acc(path):
    df2= pd.read_csv(path+'pooling_eval.csv')
    #print(df2)
    return np.mean(df2['avg_acc '])
# crazylist=['/train_model_a363460a_1_batch_size=8,column_lambda=2569099.8214,dr1=0.5332,embedding_size=12,entropy_lambda=6419201.9280,fc_size=14_2023-07-13_18-39-04/','/train_model_ae901487_186_batch_size=4,column_lambda=656.6624,dr1=0.4572,embedding_size=4,entropy_lambda=6264.4211,fc_size=25,lr=0._2023-07-20_23-15-47/']
# df_best= pd.read_csv('./zintl_janossy_constant'+crazylist[0]+'pooling_eval.csv')
# df_second= pd.read_csv('./zintl_janossy_constant'+crazylist[1]+'pooling_eval.csv')
#
# all_cifs= list(df_best['name'])
# #crystal= Structure.from_file(os.path.join(args.datadir,cifs[i]))
# all_species=[]
# for cif in all_cifs:
#     crystal= Structure.from_file('../Main_fol_Zintl/'+cif)
#     all_species= all_species+crystal.species
#
# all_species= np.unique(all_species)
#
# df_best['species']= df_best['name'].apply(get_species)
# print(df_best)
# print(all_species)
df= pd.read_csv('./decompresults_8_7_23/large_eval.csv')
df['avg_acc']= df['name'].apply(get_mean_acc)
df= df[df['MAE']<.7]
print(df)
plt.scatter(df['MAE'], df['avg_acc'])
plt.xlabel('Validation MAE')
plt.ylabel('Average crystal pooling accuracy')
plt.show()
