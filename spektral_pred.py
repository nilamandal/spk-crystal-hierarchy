from spektral.data import Dataset, DisjointLoader
#from spektral.data.utils import to_batch
#from spektral.layers import CrystalConv, DiffPool, ops, GlobalMaxPool#, Disjoint2Batch
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
#import time
from spektral_essential_objects import GaussianDistance,MyDataset, HNetConcat, PartitionedData
#import matplotlib.pyplot as plt
from sklearn import svm
import pylab as pl
#from sklearn.inspection import DecisionBoundaryDisplay


#begin_time = time.time()
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
        outfile_main=open('out.csv','w+')
        outfile_main.write('name,abs_error \n')
        pred, s_tensor = model(inputs, training=False)
        num_perfect=0
        num_imperfect=1
        scores=[]
        Cs=[]
        for j in range(len(s_tensor)):
                assign= s_tensor[j]
                individual_error= np.abs(target[j]-pred[0][j])
                crystal= Structure.from_file(os.path.join(args.datadir,cifs[i]))
                #print(crystal)
                ground_truth_P1= df_reference[df_reference['Id']==cifs[i]].P1.values[0]
                savepath=os.path.join(args.datadir,os.path.dirname(cifs[i]))
                assign=assign[:len(crystal)]
                outfile= open(savepath+'/pool.dat', 'w+')
                outfile.write('num,species,a,b,c,P1,P2,ground_truth_P1 \n')
                mainline= cifs[i]+','+str(individual_error)+'\n'
                outfile_main.write(mainline)
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
                #print(C, score)
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
        # plt.scatter(Cs, scores)
        # plt.xlabel('C values')
        # plt.ylabel('Accuracy Score')
        # plt.show()
        output.append(outs)
        if step == loader.steps_per_epoch:

            output = np.array(output)
            return np.average(output), s_tensor, pred#, b

def pool_plots(binary_feats, binary_targets, clf):
    #print('feats:')
    #for f in binary_feats:
    #zz    print(f)
    #print(binary_targets)
    #print(clf.predict(binary_feats))
    #print('vecs:')
    #print(clf.support_vectors_)
    #print(len(binary_feats))
    print(len(clf.support_vectors_))
    print(clf.coef_)

    w =  clf.coef_[0]
    a = -w[0]/w[1]
    xx = np.linspace(-5, 5)
    yy = a*xx - (clf.intercept_[0])/w[1]
    margin = 1/np.sqrt(np.sum(clf.coef_**2))
    print('margin:')
    print(margin)
    print('---')
    yy_down = yy + a*margin
    yy_up   = yy - a*margin
    # plot the line, the points, and the nearest vectors to the plane
    pl.figure(1, figsize=(4, 3))
    pl.clf()
    pl.set_cmap(pl.cm.Paired)
    pl.plot(xx, yy, 'k-')
    pl.plot(xx, yy_down, 'k--')
    pl.plot(xx, yy_up, 'k--')
    pl.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
            s=80, facecolors='none', zorder=10, edgecolors='#000000')
    pl.scatter(binary_feats[:,0], binary_feats[:,1], c=binary_targets, zorder=10, edgecolors='#000000')

    pl.axis('tight')
    x_min = -2
    x_max = 2
    y_min = -2
    y_max = 2

    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    Z = clf.predict(np.c_[XX.ravel(), YY.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(XX.shape)
    pl.figure(1, figsize=(4, 3))
    pl.set_cmap(pl.cm.Paired)
    pl.pcolormesh(XX, YY, Z)

    pl.xlim(x_min, x_max)
    pl.ylim(y_min, y_max)

    #pl.xticks(())
    #pl.yticks(())
    pl.show()

def evaluate_pool(binary_feats, binary_targets):
    binary_feats= np.array(binary_feats)
    for C_param in range(-8, 8):
        C= 10**C_param
        clf = svm.SVC(C=C, kernel='linear')
        clf.fit(binary_feats, binary_targets)
        score= clf.score(binary_feats, binary_targets)
        if score==1:

            pool_plots(binary_feats, binary_targets, clf)
            return C, score
    #print(binary_feats)
    #print(clf.decision_function(binary_feats))
    pool_plots(binary_feats, binary_targets, clf)
    return C, score

checkpoint_path = "./longmodel_b59393f5/goodmodel.ckpt.index"

checkpoint_dir = os.path.dirname(checkpoint_path)

args = parser.parse_args(sys.argv[1:])


val_df = pd.read_csv(os.path.join(args.datadir,'val.csv'), names=['id','target'], header=0)
val_df = val_df.sample(frac=1).reset_index(drop=True)
val_df= val_df.head(10)
data= MyDataset(val_df, args.datadir, args.radius_angstroms, args.num_nbrs, args.task)
loader_va= DisjointLoader(data, shuffle=False, batch_size=len(val_df))
cifs=data.get_cifs()

paramsdict= json.load(open(checkpoint_dir+'/params.json'))

#model= HNetConcat('r', 1, embedding_size=paramsdict['embedding_size'], d1=paramsdict['dr1'], d2=paramsdict['dr2'], el=paramsdict['entropy_lambda'], cl=paramsdict['column_lambda'], return_s=True)
model= HNetConcat('r', 1, el=paramsdict['entropy_lambda'], cl=paramsdict['column_lambda'], return_s=True)

# #sys.stdout = open('./debug_max_t.txt', 'w')

latest = tf.train.latest_checkpoint(checkpoint_dir)
model.load_weights(latest)

result, s_tensors, pred=evaluate(loader_va ,model, cifs)
#print(result)
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
