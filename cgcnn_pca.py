from spektral.data import Dataset, DisjointLoader
import tensorflow as tf
from tensorflow.keras import Model
import numpy as np
import pandas as pd
import os
import json
from spektral_essential_objects import AtomFeaDataset, CGCNNModel
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE, MDS
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sn
from pymatgen.core.structure import Structure

def main(fullpath_of_model, fullpath_of_data_file, write_output_path, parampath):
    #load hyperparameter files
    checkpoint_path = fullpath_of_model+"goodmodel.keras"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    data_dir = os.path.dirname(fullpath_of_data_file)
    config= json.load(open(parampath+'params.json'))

    #load data
    val_df = pd.read_csv(fullpath_of_data_file, header=0)
    mini= val_df.head(1)
    minidata= AtomFeaDataset(mini, data_dir, 8, 12, 'r')
    miniloader= DisjointLoader(minidata, shuffle=False, batch_size=1)

    data= AtomFeaDataset(val_df, data_dir, 8, 12, 'r')
    loader= DisjointLoader(data, shuffle=False, batch_size=len(data))
    cifs=data.get_cifs()
    singleloader= DisjointLoader(data, shuffle=False, batch_size=1)

    if not os.path.exists(write_output_path):
        os.makedirs(write_output_path)

    model= CGCNNModel(config['embedding_size'], config['hidden_size'], config['num_layers'])
    model.return_embedding= True
    #initialize model prior to loading weights
    mini_input, mini_target= miniloader.__next__()
    pred = model(mini_input, training=False)
    model.load_weights(fullpath_of_model+'goodmodel.weights.h5')

    inputs, target = loader.__next__()
    feats = model(inputs, training=False)
    scaler = StandardScaler()
    feats= scaler.fit_transform(feats)
    #PCA
    pca = PCA(n_components=9)
    pc_feats= pca.fit_transform(feats)
    total_explained_variance=np.sum(pca.explained_variance_ratio_)
    print(total_explained_variance)
    #MDS
    mds = MDS(n_components=2, n_init=10)
    feats_transformed= mds.fit_transform(pc_feats)
    if not os.path.exists(write_output_path):
         os.makedirs(write_output_path)
    for i in range(len(cifs)):

        crystal= Structure.from_file('../Main_fol_Zintl/'+cifs[i])
        ciftitle= cifs[i].replace('./','').replace('/','_')
        elements=[str(crystal[j].specie) for j in range(len(crystal))]
        num_atoms= len(elements)
        inputs, target = singleloader.__next__()
        feats = model(inputs, training=False)
        feats= scaler.transform(feats)


        pc_feats= pca.transform(feats)
        pca_df= pd.DataFrame(data = pc_feats, columns =("Comp_1", "Comp_2","Comp_3","Comp_4","Comp_5","Comp_6","Comp_7","Comp_8","Comp_9"))
        pca_df['species']= elements
        pca_df.to_csv(write_output_path+ciftitle+'_pca.csv')
        feats_transformed= mds.fit_transform(pc_feats)
        # tsne_obj = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=p)
        # tsne_out = tsne_obj.fit_transform(feats)
        mds_df = pd.DataFrame(data = feats_transformed, columns =("Dim_1", "Dim_2"))
        mds_df['species']= elements
        mds_df.to_csv(write_output_path+ciftitle+'_mds.csv')
        plt.figure()
        sn.scatterplot(data=mds_df, x='Dim_1', y='Dim_2',
                hue='species', palette="bright")


        plt.title('num_atoms='+str(num_atoms)+',total_explained_variance='+str(total_explained_variance))

        #print(ciftitle)
        plt.savefig(write_output_path+ciftitle+'.png')

        #plt.show()
    #print(np.mean(var_list))
    return 'temp'

if __name__ == '__main__':
    subpaths=['../full_cgcnn_all/83']

    #fullpath of data file is the path to the CSV FILE where the list of crystals and target values is stored.
    fullpath_of_data_file='../Main_fol_Zintl/test_by_fam_ternary.csv'
    #fullpath_of_data_file='../Main_fol_Zintl/Zintl_bonding_analysis_new_heuristic.csv'

    for pathstring in subpaths:
        pathstring= str(pathstring)
        #fullpath of model is the path to the DIRECTORY where the saved model is located.
        fullpath_of_model= './'+pathstring+'/'
        parampath_for_model= fullpath_of_model

        #write output path is the DIRECTORY where you want the output files to be saved.
        #Best practice is to use a new directory every time you run this script, to avoid past results being overwritten.
        write_output_path=fullpath_of_model+'pca9_mds_to_disk/'

        result= main(fullpath_of_model, fullpath_of_data_file, write_output_path, parampath_for_model)
