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
from spektral_essential_objects import GaussianDistance,MyDataset,HNetSimple, HNetConcat, PartitionedData, HNetElementProduct
import matplotlib.pyplot as plt


begin_time = time.time()
parser = argparse.ArgumentParser(description='crystal hierarchy arguments.')

parser.add_argument('--datadir', dest='datadir',
        help='Directory where dataset is located', default='../cgcnn-pretrained-models/data/10atom_relaxed_cifs')

parser.add_argument('--filename', dest='filename',
                    help='csv where data is located', default='id_mini.csv')
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

parser.add_argument('--task', choices=['r', 'c'],
                    default='r', help='complete a regression or '
                        'classification task (default: regression)')

def split_for_prashuns_data(data, test_element, val_element):
    data_tr=[]
    data_va=[]
    data_te=[]
    data_ex=[]
    for d in data:
        atomset= set(d._atomlist)
        if test_element in atomset:
            if val_element in atomset:
                data_ex.append(d._cif)
            else:
                data_te.append(d)
        elif val_element in atomset:
            data_va.append(d)
        else:
            data_tr.append(d)
    return data_tr, data_va, data_te, data_ex

def evaluate(loader, model, cifs, color='#000000', label=''):
    output = []
    step = 0
    all_s=[]
    all_pre_feats=[]
    cif_idx=0
    i=0
    while step < loader.steps_per_epoch:
        step += 1
        inputs, target = loader.__next__()

        pred, s_tensor = model(inputs, training=False)

        for j in range(len(s_tensor)):
                assign= s_tensor[j]

                individual_error= (target[j]-pred[0][j])**2
                if individual_error<0.001:
                    print(cifs[i])
                    print(individual_error)
                    print(assign)
                    print('---')
                i+=1

        if args.task=='c':
            outs = tf.reduce_mean(sparse_categorical_accuracy(target, pred))

        elif args.task=='r':
            outs = tf.reduce_mean(mean_squared_error(target, pred)),

        output.append(outs)
        if step == loader.steps_per_epoch:

            output = np.array(output)
            return np.average(output), s_tensor, pred#, b

checkpoint_path = "./51bayes/trial_211/51_83_211/goodmodel.ckpt"

checkpoint_dir = os.path.dirname(checkpoint_path)

args = parser.parse_args(sys.argv[1:])

data = MyDataset(args.datadir,args.filename, args.radius_angstroms, args.num_nbrs, args.task)
cifs=data.get_cifs()

datasettime=time.time()-begin_time

loader = DisjointLoader(data, batch_size=args.batch_size, shuffle=False)

model= HNetConcat(args.task, args.num_classes, embedding_size=8, return_s=True)
#sys.stdout = open('./bayes51_211_33.txt', 'w')
#print(checkpoint_dir)
latest = tf.train.latest_checkpoint(checkpoint_dir)
#print(latest)
model.load_weights(latest)
result, s_tensors, pred=evaluate(loader,model, cifs)
for layer in model.layers:
     print(layer.name, layer)
     if layer.name == 'dense_2':
         np.save('bayes51_211_83_fc',layer.weights[0].numpy())
     if layer.name=='regularized_diff_pool':
        rdp_all=layer.weights
        for i in range(len(rdp_all)):
            print(rdp_all[i])
        s_weights=rdp_all[0].numpy()
        print(s_weights)
        np.save('bayes51_211_83_sweights', s_weights)
        # plt.scatter(s_weights[:,0], s_weights[:,1], c=list(range(52)))
        # plt.xlabel('s weights column 1')
        # plt.xlim(-0.5, 0.5)
        # plt.ylim(-0.5, 0.5)
        # plt.ylabel('s weights column 2')
        # plt.title('noz4_1_0 (good)')
        # plt.savefig('noz4_1_0_sweights.png')
    #print(layer.weights)
    #print(layer.bias)
    #print('---')
#


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
