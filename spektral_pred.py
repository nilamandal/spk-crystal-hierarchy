from spektral.data import Graph, Dataset, DisjointLoader
from spektral.data.utils import to_batch
from spektral.layers import CrystalConv, DiffPool, ops, GlobalMaxPool#, Disjoint2Batch
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import MeanSquaredError, SparseCategoricalCrossentropy
from tensorflow.keras.metrics import sparse_categorical_accuracy, mean_squared_error
import numpy as np
import pandas as pd
import os
import sys
from pymatgen.core.structure import Structure
import json
import argparse
import time
from spektral_essential_objects import AtomInitializer, GaussianDistance,AtomCustomJSONInitializer,MyDataset,HNetSimple
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.spatial import distance
from pymatgen.core.structure import Structure

#from scipy.special import softmax

begin_time = time.time()
parser = argparse.ArgumentParser(description='crystal hierarchy arguments.')

parser.add_argument('--datadir', dest='datadir',
        help='Directory where dataset is located', default='../qy_requests')

parser.add_argument('--filename', dest='filename',
                    help='csv where data is located', default='id_prop.csv')
parser.add_argument('--file-out', dest='file_out',
                    help='output txt file name', default='predscriptout.txt')
#parser.add_argument('--path-out', dest='path',
#                    help='output path', default='./debugging_loeo/sc24-lr1e-3/sc24-lr1e-3/')
parser.add_argument('--num-atoms', dest='num_atoms', type=int,
                    help='Maximum number of nodes', default=10)
parser.add_argument('--num-nbrs', dest='num_nbrs', type=int,
                    help='num neighbors per atom', default=12)

parser.add_argument('--num-classes', dest='num_classes', type=int,
                    help='Number of label classes', default=3)

parser.add_argument('--radius-angstroms', dest='radius_angstroms', type=int,
                    help='search radius for neighbors', default=8)
parser.add_argument('--random-seed', dest='random_seed', type=int,
                    help='random seed for numpy', default=0)
parser.add_argument('--batch-size', dest='batch_size', type=int,
                    help='Batch size.', default=256)

#parser.add_argument('--epochs', default=30, type=int, metavar='N',
#                    help='number of total epochs to run (default: 30)')

#parser.add_argument('--lr', dest='learning_rate', type=float,
#                    help='Learning rate.', default=1e-3)
parser.add_argument('--task', choices=['r', 'c'],
                    default='r', help='complete a regression or '
                        'classification task (default: regression)')
#parser.add_argument('--patience', dest='patience',default=10, type=int,
#                    help='num epochs for early stopping')

def evaluate(loader, model, cifs):
    output = []
    step = 0
    all_s=[]
    all_pre_feats=[]
    while step < loader.steps_per_epoch:
        step += 1
        inputs, target = loader.__next__()
        order=list(inputs[3])
        order2=[order.count(0),order.count(1),order.count(2)]
        print(order2)
        pred, s_tensor = model(inputs, training=False)
        for i in range(3):
            assignment=s_tensor[i]
            if order2[i]==6:
                cif_id='mp-3486'
            elif order2[i]==24:
                cif_id='mp-29374'
            else:
                cif_id='mp-18300'
            crystal = Structure.from_file(os.path.join(args.datadir, cif_id+'.cif'))
            elements=list(crystal.species)
            print(cif_id)
            for j in range(len(elements)):

                print(str(elements[j]),assignment[j])
            print('--')
        #print(s_tensor)
        #print(pred)
        #print(target)
        #np.savez('./feats28', x=s_tensor)
        #all_s.append(s_tensor)
        #all_pre_feats.append(prepool_feats)

        if args.task=='c':
            outs = tf.reduce_mean(sparse_categorical_accuracy(target, pred))

        elif args.task=='r':
            outs = tf.reduce_mean(mean_squared_error(target, pred)),

        output.append(outs)
        if step == loader.steps_per_epoch:
            #b= tf.concat(all_pre_feats, axis=0)
            #s= tf.concat(all_s, axis=0)
            s= None
            output = np.array(output)
            return np.average(output), s, pred#, b

checkpoint_path = "../formation_3drops/debugging/108/debugging{33, 2, 36, 42, 43, 76, 77, 78, 44, 80, 45, 53, 54}-{32, 46, 47, 48, 50, 51}idx108.ckpt"

checkpoint_dir = os.path.dirname(checkpoint_path)

args = parser.parse_args(sys.argv[1:])

data = MyDataset(args.datadir,args.filename, args.radius_angstroms, args.num_nbrs, args.task)
cifs=data.get_cifs()

datasettime=time.time()-begin_time

loader = DisjointLoader(data, batch_size=args.batch_size)

model= HNetSimple(args.task, args.num_classes, return_s=True)

latest = tf.train.latest_checkpoint(checkpoint_dir)
model.load_weights(latest)

result, s_tensors, pred=evaluate(loader,model, cifs)

print(pred)
# print(s_tensors.shape)
# for i in range(len(prepool_feats)):
#     pools=np.argmax(s_tensors[i], axis=1)
#     # kmeans = KMeans(n_clusters=3, random_state=0).fit_predict(prepool_feats[i])
#     pca = PCA(n_components=2)
#     x = pca.fit_transform(prepool_feats[i])
#     id= cifs[i]
#     crystal= Structure.from_file(os.path.join(args.datadir,id))
#     # #print(id, kmeans, x)
#     #
#     plt.figure()
#     plt.xlabel('PCA dim 1')
#     plt.ylabel('PCA dim 2')
#     plt.scatter(x[:,0], x[:,1], c=pools)
#     for j, txt in enumerate(crystal.species):
#         name=str(txt)+' '+str(j)
#         plt.annotate(name, (x[j,0], x[j,1]))
#     plt.title(id)
#     filename='../hnet2pool/sc10_3class_2pl_dropboth/debugging/1/'+id[:-4]+'diffpools.png'
#     plt.savefig(filename)
#print(kmeans.labels_)
