import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import DBSCAN
from sklearn import metrics
from pymatgen.core.structure import Structure

#This script generates clusterings based on DBSCAN, and computes various clustering performance metrics. 

def dbscan_metrics(feature_df, ground_truth_P1=None):
    print(feature_df)
    clustering= DBSCAN(eps=.5, min_samples=2).fit(feature_df)
    print(clustering.labels_)
    #rand= metrics.rand_score(ground_truth_P1, clustering.labels_)
    #homo, comp, vm= metrics.homogeneity_completeness_v_measure(ground_truth_P1, clustering.labels_)

    #print(rand, homo, comp, vm, sil)

    return clustering.labels_

def pooling_clustering_metrics(feature_df, p1, ground_truth_P1):
    rand= metrics.rand_score(ground_truth_P1, p1)
    homo, comp, vm= metrics.homogeneity_completeness_v_measure(ground_truth_P1, p1)
    sil= metrics.silhouette_score(feature_df, p1, metric='euclidean')
    print(rand, homo, comp, vm, sil)

df_main=pd.read_csv('../Main_fol_Zintl/test_by_fam_ternary.csv')
df_main= df_main.head(20)
#print(df_main)
all_crystals=df_main['id'].tolist()
sil_unit=[]
sil_super=[]
for crystal_path in all_crystals:
    #df= pd.read_csv('../full_tern_varied_patience/p50_id143/test_set/'+crystal)
    print(crystal_path)
    crystal= Structure.from_file('../Main_fol_Zintl/'+crystal_path)
    #crystal.make_supercell([2,2,2])
    cdf= crystal.as_dataframe()
    #cdf['Species']= cdf['Species'].astype(str)
    #temp= pd.get_dummies(cdf['Species'])
    df_sub= cdf[['a','b','c']]
    scaler = StandardScaler()
    feature_df= scaler.fit_transform(df_sub)
    try:
        df_sub['dbscan']= dbscan_metrics(feature_df)
        sil= metrics.silhouette_score(feature_df, df_sub['dbscan'], metric='euclidean')
        sil_unit.append(sil)
    except:
        sil_unit.append(np.nan)
    filepath='../dbscan_unitonly/coord_only/'+crystal_path+'.csv'
    if not os.path.exists('../dbscan_unitonly/coord_only/'+crystal_path):
            os.makedirs('../dbscan_unitonly/coord_only/'+crystal_path)
    df_sub.to_csv(filepath)

    crystal.make_supercell([2,2,2])
    cdf= crystal.as_dataframe()
    #cdf['Species']= cdf['Species'].astype(str)
    #temp= pd.get_dummies(cdf['Species'])
    df_sub= cdf[['a','b','c']]
    feature_df= scaler.fit_transform(df_sub)
    try:
        df_sub['dbscan']= dbscan_metrics(feature_df)
        sil= metrics.silhouette_score(feature_df, df_sub['dbscan'], metric='euclidean')
        sil_super.append(sil)
    except:
        sil_super.append(np.nan)
    filepath='../dbscan_supercell/coord_only/'+crystal_path+'.csv'
    if not os.path.exists('../dbscan_supercell/coord_only/'+crystal_path):
            os.makedirs('../dbscan_supercell/coord_only/'+crystal_path)
    df_sub.to_csv(filepath)
    #pooling_clustering_metrics(df_sub, df['P1'], df['ground_truth_P1'])
    #df_sub= df_sub[]
df_main['sil_unit']=sil_unit
df_main['sil_super']= sil_super

df_main.to_csv('../dbscan_stuff_coord_only.csv')
