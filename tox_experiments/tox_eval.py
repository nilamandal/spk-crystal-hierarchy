import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import Accuracy
from spektral.data import Graph, Dataset, DisjointLoader
from spektral.utils import reorder, sp_matrix_to_sp_tensor
from sklearn.metrics import accuracy_score, f1_score
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
import os

from tox_experiments import Dataset_from_json
from spektral_essential_objects import NotShrinking

def eval_test_set(loader, model, val_df, fullpath_of_model, fullpath_of_data_file, write_output_path):
    step = 0
    pubchemid=val_df['pid'].tolist()




    while step < loader.steps_per_epoch:
        step += 1
        inputs, target = loader.__next__()


        pred, s_tensor = model(inputs, training=False)

        x, a, e, idx= inputs



        pred= pred.numpy()

        print(len(pred))
        print(len(target))
        print(len(x))
        print(len(idx))
        print(s_tensor.shape)
        print(idx)

        act_label = np.argmax(target, axis=1) # act_label = 1 (index)
        pred_label = np.argmax(pred, axis=1) # pred_label = 1 (index)
        for i in range(10):
            print(pubchemid[i])
            print(pred[i])
            print(s_tensor[i])
            print('---')
        #print(act_label)
        #print(pred_label)
        #print(f1_score(act_label, pred_label))

    return 'temp'

def main(fullpath_of_model, fullpath_of_data_file, write_output_path, parampath):

    checkpoint_path = fullpath_of_model+"0goodmodel.ckpt.index"

    checkpoint_dir = os.path.dirname(checkpoint_path)

    data_dir = os.path.dirname(fullpath_of_data_file)
    config= json.load(open(parampath+'params.json'))

    val_df = pd.read_csv(fullpath_of_data_file, header=0)
    val_df= val_df.head(10)
    data= Dataset_from_json(val_df)
    loader_va= DisjointLoader(data, shuffle=False, batch_size=len(val_df))
    #cifs=data.get_cifs()

    model= NotShrinking('c', 2, config['embedding_size'], config['cgcnn_num'], config['cgcnn_num2'], softmax_beta=config['softmax_beta'])
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    model.load_weights(latest)
    if not os.path.exists(write_output_path):
        os.makedirs(write_output_path)
    print(val_df)
    result_dict=eval_test_set(loader_va, model, val_df, fullpath_of_model, os.path.dirname(fullpath_of_data_file), write_output_path)

    return result_dict

if __name__ == "__main__":
    model_paths=['tox_target1_bestmodel']

    for pathstring in model_paths:
        pathstring= str(pathstring)
        #fullpath of model is the path to the DIRECTORY where the saved model is located.
        fullpath_of_model= './'+pathstring+'/'
        parampath_for_model= fullpath_of_model
        fullpath_of_data_file='./tox_test_set.csv'

        #write output path is the DIRECTORY where you want the output files to be saved.
        #Best practice is to use a new directory every time you run this script, to avoid past results being overwritten.
        write_output_path=fullpath_of_model+'tox_nr_ar_validation/'

        result_dict= main(fullpath_of_model, fullpath_of_data_file, write_output_path, parampath_for_model)
