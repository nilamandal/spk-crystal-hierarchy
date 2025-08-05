import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Dropout, Input
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from spektral.data import Graph, Dataset, DisjointLoader

import json
import pandas as pd
from scipy.spatial import distance

class MyDataset(Dataset):
    def __init__(self, df, task='c'):
        self.df= df
        ref= pd.read_csv('PeriodicTableCSV.csv')[['symbol', 'period', 'group']]
        self.element_ref = {}
        for i, row in ref.iterrows():
            self.element_ref[row.symbol] = [row.period, row.group]

        super().__init__()

    def read(self):
        pre_graph= self.df['structure'].tolist()
        graphs=[]
        for g in pre_graph:
            g2= self.make_dataset(g)
            graphs.append(g2)

        return graphs

    def make_dataset(self, g):
        json_graph= json.loads(g.replace("'", "\""))
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
        x= np.vstack(x)
        #print(x)

        #adjacency matrix and edge feature(s)
        adj = np.zeros((len(atoms), len(atoms)))
        e= np.zeros((len(atoms), len(atoms)))
        for i in range(len(bonds)):
            b= bonds[i]
            idx_u= b['aid1']-1
            idx_v= b['aid2']-1
            adj[idx_u, idx_v]= b['order']
            adj[idx_v, idx_u]= b['order']
            u= atoms[idx_u]
            v= atoms[idx_v]
            dist=distance.euclidean([u['x'],u['y']], [v['x'],v['y']])
            e[idx_u, idx_v]= dist
            e[idx_v, idx_u]= dist
        print(e)

        MG=Graph(x=x, a=adj, e=e)
        return MG


if __name__ == "__main__":
    df= pd.read_csv('./tox21_updated_2.csv')
    df= df.sample(n=10)
    graphs= MyDataset(df)
