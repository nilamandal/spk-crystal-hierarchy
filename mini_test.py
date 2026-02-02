import optuna
import json
import resource
import os
import logging
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

import sklearn.datasets
import sklearn.linear_model
import sklearn.model_selection

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError, CategoricalCrossentropy
from tensorflow.keras.callbacks import CallbackList, CSVLogger

from spektral.data import DisjointLoader, BatchLoader
from spektral_essential_objects import MyDataset, SparseEdgepool, AtomFeaDataset, NotShrinking, NoShrink_GAT, NoShrink_Refactor
from utils import train_step, evaluate

parser = argparse.ArgumentParser(description='crystal hierarchy arguments.')
parser.add_argument('--datadir', dest='datadir',
        help='Directory where dataset is located', default='/Users/nilamandal/desktop/Main_fol_Zintl')#/home/nim18004/Main_fol_Zintl

parser.add_argument('--task', choices=['r', 'c'],
                    default='r', help='complete a regression or classification task (default: regression)')
parser.add_argument('--out_dir', default='./optuna_fri1/', help='directory where output is saved')
args = parser.parse_args(sys.argv[1:])

all_data = pd.read_csv(os.path.join(args.datadir,'crossval.csv'))
cv_scores=[]

i=0
epochs=10

train_df= all_data[all_data['bin']!=i]
train_df= train_df.head(5)
val_df= all_data[all_data['bin']==i]
val_df= val_df.head(5)
train_data= AtomFeaDataset(train_df, args.datadir, 8, 12, args.task)
val_data= AtomFeaDataset(val_df, args.datadir, 8, 12, args.task)
load_tr= DisjointLoader(train_data, batch_size=5, epochs=epochs)
load_tr_eval= DisjointLoader(train_data, batch_size=len(train_data))
load_va= DisjointLoader(val_data, batch_size=len(val_data))

model= NotShrinking(args.task, 1, batch_size=5)
optim=Adam(.001)

loss_fn= MeanSquaredError()
train_metric=[]
val_metric_list=[]
early_stop_counter= 0
patience= 50
epoch = step = 0

best_val_loss = np.inf
logs = {}

for batch in load_tr:

    step += 1

    #all_callbacks.on_train_batch_begin(step)
    loss, metric = train_step(*batch, model, loss_fn, optim)
    #all_callbacks.on_train_batch_end(step, logs)
    tr_loss, tr_rmse, tr_mae= evaluate(load_tr_eval, model, loss_fn)
    #print(tr_loss)
