import pandas as pd
from pymatgen.core.structure import Structure
import os
from pymatgen.analysis.structure_prediction.volume_predictor import DLSVolumePredictor
from collections import Counter

#count=0

def scale_by_pred_vol(structure, site_bias, dls_vol_predictor):
    #global count
    # first predict the volume using the average volume per element (from ICSD)
    site_counts = pd.Series(Counter(
        str(site.specie) for site in structure.sites)).fillna(0)
    curr_site_bias = site_bias[site_bias.index.isin(site_counts.index)]

    try:
        linear_pred = site_counts @ curr_site_bias
        structure.scale_lattice(linear_pred)
    except:
        pass
        #count+=1
    # then apply Pymatgen's DLS predictor
    pred_volume = dls_vol_predictor.predict(structure)
    structure.scale_lattice(pred_volume)
    #
    return structure


dls_vol_predictor = DLSVolumePredictor()

def scale_dls_only(c):
    c=str(c)
    try:
        from pymatgen.core.structure import Structure
    except:
        crystal= Structure.from_file(os.path.join(data_path,c))
    structure= dls_vol_predictor.get_predicted_structure(crystal)
    newpath='./sc24_scaled/'+c.split('/')[-1][:-7]+'.cif'
    structure.to(filename=newpath)
    return newpath


data_path= '../crystalhierarchydata/sc24/'

data_file='id_prop24.csv'


df= pd.read_csv(data_path+data_file, names=['id', 'class'])

cifs=list(df['id'])
#site_bias_file = "inputs/site_volumes_from_icsd.csv"
#site_bias = pd.read_csv(site_bias_file, index_col=0, squeeze=True)

df['newpath']=df['id'].apply(scale_dls_only)

df.to_csv('./sc24_scaled/id_prop.csv')
