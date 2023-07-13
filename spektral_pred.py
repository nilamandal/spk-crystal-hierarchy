from spektral.data import Dataset, DisjointLoader
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
from spektral_essential_objects import GaussianDistance,MyDataset, HNetConcat, PartitionedData, HNetConcatJanossy
from sklearn import svm
import pylab as pl

parser = argparse.ArgumentParser(description='crystal hierarchy arguments.')

parser.add_argument('--datadir', dest='datadir',
        help='Directory where dataset is located', default='../Main_fol_Zintl2/')

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
        outfile_main=open('janossy0.csv','w+')
        outfile_main.write('name,abs_error,pool_margin \n')
        pred, s_tensor = model(inputs, training=False)
        num_perfect=0
        num_imperfect=0
        cat_greater=0
        an_greater=0
        scores=[]
        Cs=[]
        maes_for_plot=[]
        class_accuracies_for_plot=[]
        crystal_size=[]
        for j in range(len(s_tensor)):
                assign= s_tensor[j]

                #individual_error= np.abs(target[j]-pred[0][j])
                individual_error= np.abs(target[j]-pred[j])
                maes_for_plot.append(individual_error)
                crystal= Structure.from_file(os.path.join(args.datadir,cifs[i]))
                crystal_size.append(len(crystal))

                ground_truth_P1= df_reference[df_reference['Id']==cifs[i]].P1.values[0]
                savepath=os.path.join(args.datadir,os.path.dirname(cifs[i]))
                assign=assign[:len(crystal)]
                outfile= open(savepath+'/pool.dat', 'w+')
                outfile.write('num,species,a,b,c,P1,P2,ground_truth_P1\n')

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

                C, score, margin= evaluate_pool(binary_feats, binary_targets)
                # if score<0.3:
                #     print(cifs[i])
                #     print(score)
                scores.append(score)
                Cs.append(C)
                class_accuracies_for_plot.append(score)
                mainline= cifs[i]+','+str(individual_error)+','+str(margin)+'\n'
                outfile_main.write(mainline)

                if np.isnan(score):
                    #print(score, 'nan')
                    num_imperfect+=1
                elif score<1:
                    #print(score, 'imperfect')
                    num_imperfect+=1
                else:
                    #print(score, 'perfect')
                    num_perfect+=1

                line= 'pool accuracy='+str(score)+'C='+str(C)+'\n'
                outfile.write(line)

                line= '\n absolute error = '+str(float(individual_error))
                outfile.write(line)
                line= '\n '+ args.datadir+','+cifs[i]
                outfile.close()
                if score==1:
                    poolfile= args.datadir+cifs[i][:-7]+'pool.dat'
                    #print(poolfile)
                    df= pd.read_csv(poolfile)
                    an_mean=np.mean(df[df['ground_truth_P1']==1.0]['P1'])
                    cat_mean=np.mean(df[df['ground_truth_P1']==0.0]['P1'])
                    if an_mean>cat_mean:
                        an_greater+=1
                    else:
                        cat_greater+=1
                    #print('---')
                i+=1
        print('crystals pooled correctly')
        print(num_perfect)
        print('crystals_pooled incorrectly')
        print(num_imperfect)
        print('num cat greater P1')
        print(cat_greater)
        print("num an greater p1")
        print(an_greater)
        plus75= [x for x in class_accuracies_for_plot if x>=0.75]
        print('greater than 75% accuracy')
        print(len(plus75)/len(class_accuracies_for_plot))
        pl.figure()
        pl.scatter(maes_for_plot, class_accuracies_for_plot, alpha=.33)
        pl.xlabel('Absolute error (eV/atom)')
        pl.ylabel('SVM classification accuracy')
        pl.title('Absolute error vs cation/anion classification accuracy')
        pl.show()
        pl.figure()
        pl.scatter(maes_for_plot, crystal_size, alpha=.33)
        pl.xlabel('Absolute error (eV/atom)')
        pl.ylabel('Num atoms in crystal')
        pl.title('Absolute error vs num atoms in crystal')
        pl.show()
        pool_acc_vs_crystal_size(crystal_size,class_accuracies_for_plot)
        if args.task=='c':
            outs = tf.reduce_mean(sparse_categorical_accuracy(target, pred))

        elif args.task=='r':
            outs = tf.reduce_mean(mean_squared_error(target, pred)),

        output.append(outs)
        if step == loader.steps_per_epoch:
            output = np.array(output)
            print('MSE:')
            print(np.average(output))
            print('RMSE:')
            print(np.sqrt(np.average(output)))
            print('MAE:')
            print(np.average(maes_for_plot))
            return np.average(output), s_tensor, pred#, b

def pool_acc_vs_crystal_size(size,accuracies):
    pl.figure()
    pl.scatter(size, accuracies, alpha=.33)
    pl.xlabel('Num atoms in crystal')
    pl.ylabel('SVM classification accuracy')
    pl.title('Crystal size vs cation/anion classification accuracy')
    pl.show()

def pool_db_plots(binary_feats, binary_targets, clf):

    w =  clf.coef_[0]
    a = -w[0]/w[1]
    xx = np.linspace(-5, 5)
    yy = a*xx - (clf.intercept_[0])/w[1]
    margin = 1/np.sqrt(np.sum(clf.coef_**2))

    yy_down = yy + a*margin
    yy_up   = yy - a*margin
    # plot the line, the points, and the nearest vectors to the plane
    pl.figure(1, figsize=(4, 4))
    pl.clf()
    pl.set_cmap(pl.cm.Paired)
    pl.plot(xx, yy, 'k-')
    pl.plot(xx, yy_down, 'k--')
    pl.plot(xx, yy_up, 'k--')
    pl.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80, facecolors='none', zorder=10, edgecolors='#000000')
    pl.scatter(binary_feats[:,0], binary_feats[:,1], c=binary_targets, zorder=10, edgecolors='#000000')

    pl.axis('tight')
    x_min = 0
    x_max = 1
    y_min = 0
    y_max = 1

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
    scores= {}
    margins= {}
    if len(np.unique(binary_feats, axis=0))==1:
        print('ITS ALL 1')
        return np.nan, 0, 0
    else:

        for C_param in range(-5,8):
            C= 10**C_param
            clf = svm.SVC(C=C, kernel='linear', max_iter=10000)
            clf.fit(binary_feats, binary_targets)
            score= clf.score(binary_feats, binary_targets)

            w =  clf.coef_[0]
            a = -w[0]/w[1]
            xx = np.linspace(-5, 5)
            yy = a*xx - (clf.intercept_[0])/w[1]
            margin = 1/np.sqrt(np.sum(clf.coef_**2))
            scores[C]= score
            margins[C]= margin
            if score==1:
                #df_new=pd.DataFrame(clf.decision_function(binary_feats)/np.sqrt(np.sum(clf.coef_**2)))
                #df_new.to_csv('dboundary.csv')
                return C, score, margin

        #pool_plots(binary_feats, binary_targets, clf)
        #df_new=pd.DataFrame(clf.decision_function(binary_feats)/np.sqrt(np.sum(clf.coef_**2)))
        #df_new.to_csv('dboundary.csv')
        best_score= max(scores.values())
        best_C= max(scores, key=scores.get)
        best_margin= margins[best_C]
        return best_C, best_score, best_margin

checkpoint_path = "./janossy_exps2/janossy/0/goodmodel.ckpt.index"
#checkpoint_path = "./train_model_2023-07-03_17-20-23/train_model_1285a2b1_3_batch_size=1,column_lambda=1219992.1467,dr1=0.5991,dr2=0.9550,embedding_size=2,entropy_lambda=17520562.6629_2023-07-03_17-20-41/goodmodel.ckpt.index"



checkpoint_dir = os.path.dirname(checkpoint_path)

args = parser.parse_args(sys.argv[1:])


val_df = pd.read_csv(os.path.join(args.datadir,'val.csv'), names=['id','target'], header=0)
#val_df = val_df.sample(frac=1).reset_index(drop=True)
#val_df= val_df.head(10)
#val_df= val_df[val_df['id']=='./Binary/a4c1/92/As4Mg1/CONTCAR']
data= MyDataset(val_df, args.datadir, args.radius_angstroms, args.num_nbrs, args.task)
loader_va= DisjointLoader(data, shuffle=False, batch_size=len(val_df))
cifs=data.get_cifs()

#paramsdict= json.load(open(checkpoint_dir+'/params.json'))

#model= HNetConcatJanossy('r', 1, embedding_size=paramsdict['embedding_size'], d1=paramsdict['dr1'], d2=paramsdict['dr2'], el=paramsdict['entropy_lambda'], cl=paramsdict['column_lambda'], return_s=True)
#model= HNetConcat('r', 1, el=paramsdict['entropy_lambda'], cl=paramsdict['column_lambda'], return_s=True)
model= HNetConcatJanossy('r', 1, return_s=True)

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
    #print(layer.weights)
    #print(layer.bias)
    #print('---')
