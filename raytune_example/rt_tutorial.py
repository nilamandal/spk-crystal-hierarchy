import tensorflow as tf
import os
import sys
import argparse
from spektral_essential_objects import GaussianDistance, MyDataset, HNetRecurrent
from spektral.data import DisjointLoader
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.losses import MeanSquaredError
from ray.tune.integration.keras import TuneReportCallback
from ray.tune.schedulers import AsyncHyperBandScheduler
import ray
from ray import air, tune
import numpy as np

parser = argparse.ArgumentParser(description='crystal hierarchy arguments.')

parser.add_argument('--datadir', dest='datadir',
        help='Directory where dataset is located', default='/Users/nilamandal/Desktop/cgcnn-pretrained-models/data/10atom_relaxed_cifs')
parser.add_argument('--filename', dest='filename',
                    help='csv where data is located', default='id_mini.csv')
parser.add_argument('--num-nbrs', dest='num_nbrs', type=int,
                    help='num neighbors per atom', default=12)
parser.add_argument('--radius-angstroms', dest='radius_angstroms', type=int,
                    help='search radius for neighbors', default=10)
args = parser.parse_args(sys.argv[1:])
data= MyDataset(args.datadir,args.filename, args.radius_angstroms, args.num_nbrs, 'r')
load_tr = DisjointLoader(data, batch_size=32, epochs=1)
model= HNetRecurrent('r', 1)

#print(model.layers)

with tf.GradientTape() as tape:
     for inputs, target in load_tr:
         predictions = model(inputs, training=False)
#print(model.trainable_variables)
conv_temp=model.layers[1]


#print(conv_temp.dense_s.get_weights())
#print(conv_temp.dense_f.get_weights())
#print(conv_temp.bn1.get_weights())
#print(conv_temp.bn2)
# for l in model.layers:
#     print(l)
#     print(type(l))
    #t=l.get_weights()
    #for k in t:
        #print(k)
    #    print(k.shape)
