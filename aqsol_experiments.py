import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Dropout, Input
from tensorflow.keras.models import Model
from spektral.data import Graph, Dataset
from spektral.utils import reorder, sp_matrix_to_sp_tensor

import ConfigSpace
from hpbandster.optimizers.config_generators.bohb import BOHB
from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
from ray.tune.search.bohb import TuneBOHB
from ray import tune

import json
import pandas as pd
from scipy.spatial import distance
import scipy.sparse as sp
import argparse
import sys
import numpy as np

from utils import train_single_model
from CorrectedRepeater import BOHBRepeater

parser = argparse.ArgumentParser(description='crystal hierarchy arguments.')
parser.add_argument('--datadir', dest='datadir',
                    help='Directory where dataset is located', default='/home/jaa21031/spk-crystal-hierarchy/')
#parser.add_argument('--num-nbrs', dest='num_nbrs', type=int,
#                    help='num neighbors per atom', default=12)
#parser.add_argument('--radius-angstroms', dest='radius_angstroms', type=int,
#                    help='search radius for neighbors', default=10)
parser.add_argument('--task', choices=['r', 'c'],
                    default='c', help='complete a regression or classification task (default: classification)')
args = parser.parse_args(sys.argv[1:])

class Dataset_from_json(Dataset):
    def __init__(self, df, task='r', target='Solubility'):
        #print(df)
        self.df= df.dropna(subset=[target])
        self.dataframe = df
        self.target_name= target
        ref= pd.read_csv(args.datadir+'PeriodicTableCSV.csv')[['symbol', 'period', 'group']]
        self.element_ref = {}
        for i, row in ref.iterrows():
            self.element_ref[row.symbol] = [row.period, row.group]

        super().__init__()

    def read(self):
        pre_graph= self.df['structure'].tolist()
        targets= self.df[self.target_name].tolist()
        graphs=[]
        for i in range(len(pre_graph)):
            g2= self.make_dataset(pre_graph[i], targets[i])
            graphs.append(g2)

        return graphs

    def make_dataset(self, g, target):
        try:
            json_graph= json.loads(g.replace("'", "\""))
        except json.decoder.JSONDecodeError as e:
            print(g)
            print(e)
        atoms= json_graph['atoms']
        bonds= json_graph['bonds']

        x=[] #atom features
        for i in range(len(atoms)):
            a= atoms[i]
            encoding= self.element_ref[a['element']]
            group_encoding= np.zeros(18)
            group_encoding[encoding[1]-1]=1
            row_encoding= np.zeros(9)
            row_encoding[encoding[0]-1]=1
            atom_hot=np.concatenate((group_encoding, row_encoding))
            x.append(atom_hot)
        atom_fea= np.vstack(x)
        #print(x)

        #adjacency matrix and edge feature(s)
        adj = np.zeros((len(atoms), len(atoms)))
        e= np.zeros((len(atoms), len(atoms), 1))
        for i in range(len(bonds)):
            b= bonds[i]
            idx_u= b['aid1']-1
            idx_v= b['aid2']-1
            adj[idx_u, idx_v]= b['order']
            adj[idx_v, idx_u]= b['order']
            u= atoms[idx_u]
            v= atoms[idx_v]
            dist=distance.euclidean([u['x'],u['y']], [v['x'],v['y']])
            e[idx_u, idx_v, 0]= dist
            e[idx_v, idx_u, 0]= dist
        #print(e)
        adj=sp.csr_matrix(adj)
        if target==0:
            onehot_target= [1,0]
        else:
            onehot_target= [0,1]
        MG=Graph(x=atom_fea, a=adj, e=e, y=onehot_target)
        return MG

    def get_cifs(self):       
        df = self.dataframe.sample(frac=1).reset_index(drop=True)
        cifs = list(df['structure'])
        print(len(cifs))
        for i in range(len(cifs)):
            cif = cifs[i]
            cif.strip()
            try:
                cifs[i] = json.loads(cif.replace("'", "\""))
            except Exception:
                pass
        print(type(cifs[0]))
        return cifs

def main_workflow(config):
    df= pd.read_csv(args.datadir+'aqsol_train.csv')
    #df= df.head(200)
    cv_scores=[]
    config['task']= 'r'
    for i in range(5):
        train= df[df['fold']!=i]
        train= Dataset_from_json(train)

        val= df[df['fold']==i]
        val= Dataset_from_json(val)
        save_path= './'+str(i)
        score= train_single_model(config, train, val, epochs=1000, save_path=save_path)
        cv_scores.append(score)


    return {"score": np.mean(cv_scores)}


if __name__ == "__main__":
     NUM_MODELS = 500

     trial_space = {
           'embedding_size': tune.choice([4,8,16,32,64]),
           'cgcnn_num': tune.choice([1,2,3]),
           'cgcnn_num2': tune.choice([1,2,3]),
           #'cgcnn_p': tune.choice([1,2,3]),
           'batch_size': tune.choice([4,8,16,32,64]),
           'softmax_beta': tune.loguniform(1, 1e8),
           'lr': tune.loguniform(1e-8, 1e-1)
       }

     bohb_hyperband = HyperBandForBOHB(
       time_attr="training_iteration",
       max_t=81,
       reduction_factor=3,
       stop_last_trials=False,
     )

     bohb = BOHBRepeater(metric='score', mode='min', repeat=1, max_concurrent=100)
     train_model_object = tune.with_resources(main_workflow, {"cpu": 1})
     tuner = tune.Tuner(train_model_object, tune_config=tune.TuneConfig(
       search_alg=bohb,
       scheduler=bohb_hyperband,
       metric='score',
       mode='min',
       num_samples=NUM_MODELS), param_space=trial_space)
     print('CREATED all TUNING OBJECTS')
     results = tuner.fit()
     print(results)
