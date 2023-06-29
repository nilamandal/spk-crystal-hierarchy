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
from spektral_badpad_objects import GaussianDistance,MyDataset,HNetSimple, HNetConcat, PartitionedData, HNetMultifilter
import matplotlib.pyplot as plt
from sklearn import svm


begin_time = time.time()
parser = argparse.ArgumentParser(description='crystal hierarchy arguments.')

parser.add_argument('--datadir', dest='datadir',
        help='Directory where dataset is located', default='../Main_fol_Zintl_prepad_poolings/')

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

parser.add_argument('--task', choices=['r', 'c'],
                    default='r', help='complete a regression or '
                        'classification task (default: regression)')

df_reference= pd.read_csv('../Main_fol_Zintl/ICSD_Zintl_TE_pooling.csv')

def random_split(dataset):
    data_tr=[]
    data_va=[]
    data_te=[]
    split= int(len(dataset)/5)
    split_2= split*2
    #print(split)
    data_te=dataset[:split]
    data_va= dataset[split:split_2]
    data_tr=dataset[split_2:len(dataset)]
    print(len(data_te))
    print(len(data_va))
    print(len(data_tr))
    loader_tr = DisjointLoader(PartitionedData(data_tr), batch_size=32, epochs=1)
    loader_va = DisjointLoader(PartitionedData(data_va), batch_size=len(data_va))
    loader_te = DisjointLoader(PartitionedData(data_te), batch_size=len(data_te))
    print(data_te)
    for d in data_te:
        print(d._cif)
    return loader_tr, loader_va, loader_te

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
        num_perfect=0
        num_imperfect=1
        scores=[]
        Cs=[]
        for j in range(len(s_tensor)):
                assign= s_tensor[j]
                individual_error= np.abs(target[j]-pred[0][j])
                crystal= Structure.from_file(os.path.join(args.datadir,cifs[i]))
                ground_truth_P1= df_reference[df_reference['Id']==cifs[i]].P1.values[0]
                savepath=os.path.join(args.datadir,os.path.dirname(cifs[i]))
                assign=assign[:len(crystal)]
                outfile= open(savepath+'/pool.dat', 'w')
                outfile.write('num,species,a,b,c,P1,P2,ground_truth_P1 \n')
                binary_feats=[]
                binary_targets=[]
                for k in range(len(crystal)):
                    if str(crystal[k].specie) in ground_truth_P1:
                        truth_val=1
                    else:
                        truth_val= 0
                    line="{},{},{},{},{},{},{},{} \n".format(k, crystal[k].specie, crystal[k].a, crystal[k].b, crystal[k].c, assign[k,0], assign[k,1], truth_val)
                    binary_targets.append(truth_val)
                    binary_feats.append(assign[k].numpy())
                    outfile.write(line)
                C, score= evaluate_pool(binary_feats, binary_targets)
                scores.append(score)
                Cs.append(C)
                print(C, score)
                if score<1:
                    num_imperfect+=1
                else:
                    num_perfect+=1
                #print(scores)
                line= 'pool accuracy='+str(score)+'C='+str(C)+'\n'
                outfile.write(line)
                ##    print(cifs[i])
                    #print(binary_targets)
                line= '\n absolute error = '+str(float(individual_error))
                outfile.write(line)
                outfile.close()


                i+=1
        print('crystals pooled correctly')
        print(num_perfect)
        print('crystals_pooled incorrectly')
        print(num_imperfect)
        if args.task=='c':
            outs = tf.reduce_mean(sparse_categorical_accuracy(target, pred))

        elif args.task=='r':
            outs = tf.reduce_mean(mean_squared_error(target, pred)),
        plt.scatter(Cs, scores)
        plt.xlabel('C values')
        plt.ylabel('Accuracy Score')
        plt.show()
        output.append(outs)
        if step == loader.steps_per_epoch:

            output = np.array(output)
            return np.average(output), s_tensor, pred#, b

def evaluate_pool(binary_feats, binary_targets):
    for C in range(1, 50):
        clf = svm.LinearSVC(C=C, max_iter=10000)
        clf.fit(binary_feats, binary_targets)
        score= clf.score(binary_feats, binary_targets)
        if score==1:
            return C, score
    return C, score

checkpoint_path = "../concat_results/train_model_74e166f4_Best/goodmodel.ckpt.index"

checkpoint_dir = os.path.dirname(checkpoint_path)

args = parser.parse_args(sys.argv[1:])


val_df = pd.read_csv(os.path.join(args.datadir,'val.csv'), names=['id','target'], header=0)
data= MyDataset(val_df, args.datadir, args.radius_angstroms, args.num_nbrs, args.task)
loader_va= DisjointLoader(data, shuffle=False, batch_size=len(val_df))
cifs=data.get_cifs()
print(data)
# loader_tr, loader_va, loader_te=random_split(data)
# datasettime=time.time()-begin_time
#
paramsdict= json.load(open(checkpoint_dir+'/params.json'))
print(paramsdict)

#
model= HNetConcat('r', 1, embedding_size=paramsdict['embedding_size'], d1=paramsdict['dr1'], d2=paramsdict['dr2'], el=paramsdict['entropy_lambda'], cl=paramsdict['column_lambda'], return_s=True)
# #sys.stdout = open('./debug_max_t.txt', 'w')

latest = tf.train.latest_checkpoint(checkpoint_dir)
model.load_weights(latest)
print(model)
result, s_tensors, pred=evaluate(loader_va ,model, cifs)
print(result)
# # for layer in model.layers:
#      print(layer.name, layer)
#      if layer.name == 'dense_2':
#          np.save('bayes51_211_83_fc',layer.weights[0].numpy())
#      if layer.name=='regularized_diff_pool':
#         rdp_all=layer.weights
#         for i in range(len(rdp_all)):
#             print(rdp_all[i])
#         s_weights=rdp_all[0].numpy()
#         print(s_weights)
#         np.save('bayes51_211_83_sweights', s_weights)
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
