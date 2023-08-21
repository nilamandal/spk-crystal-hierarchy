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
        help='Directory where dataset is located', default='../Main_fol_Zintl/')

parser.add_argument('--filename', dest='filename',
                    help='csv where data is located', default='id_prop.csv')
parser.add_argument('--file-out', dest='file_out',
                    help='output txt file name', default='predscriptout.txt')
#parser.add_argument('--ckpt_path', dest='checkpoint_path',
#                   help='output path', default='./janossy_first_19/train_model_a363460a_1_best_val_error/')
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
parser.add_argument('--task', choices=['r', 'c'],
                    default='r', help='complete a regression or '
                        'classification task (default: regression)')

def evaluate(loader, model, cifs, df_reference, args, main_checkpoint_path, color='#000000', label=''):
    output = []
    step = 0
    all_s=[]
    all_pre_feats=[]
    cif_idx=0
    i=0
    while step < loader.steps_per_epoch:
        step += 1
        inputs, target = loader.__next__()
        rep_csv= open(main_checkpoint_path+'learned_reps.csv','w+')
        outfile_main=open(main_checkpoint_path+'pooling_eval.csv','w+')
        outfile_main.write('name,abs_error,pool_margin,perfect,avg_acc \n')
        pred, s_tensor, learned_rep = model(inputs, training=False)
        rep_csv.write('cif,pool_num,f0,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11 \n')
        #print(learned_rep.shape)
        #print(list(learned_rep.numpy()))
        num_perfect=0
        list_perfect=[]
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
                rep= learned_rep[j].numpy()
                individual_error= np.abs(target[j]-pred[j])
                maes_for_plot.append(individual_error)
                crystal= Structure.from_file(os.path.join(args.datadir,cifs[i]))
                crystal_size.append(len(crystal))

                line0="{},0,{},{},{},{},{},{},{},{},{},{},{},{} \n".format(cifs[i],rep[0,0],rep[0,1],rep[0,2],rep[0,3],rep[0,4],rep[0,5],rep[0,6],rep[0,7],rep[0,8],rep[0,9],rep[0,10],rep[0,11])
                line1="{},1,{},{},{},{},{},{},{},{},{},{},{},{} \n".format(cifs[i],rep[1,0],rep[1,1],rep[1,2],rep[1,3],rep[1,4],rep[1,5],rep[1,6],rep[1,7],rep[1,8],rep[1,9],rep[1,10],rep[1,11])

                #print(line0)
                #print(line1)
                #line0="{},1,{},{},{},{},{},{},{} \n".format(cifs[i])
                rep_csv.write(line0)
                rep_csv.write(line1)


                ground_truth_P1= df_reference[df_reference['Id']==cifs[i]].P1.values[0]
                savepath=os.path.join(args.datadir,os.path.dirname(cifs[i]))
                assign=assign[:len(crystal)]
                outfile= open(savepath+'/pool.dat', 'w+')
                outfile.write('num,species,a,b,c,P1,P2,ground_truth_P1,SVM_pred_P1\n')

                binary_feats=[]
                binary_targets=[]
                for k in range(len(crystal)):
                    if str(crystal[k].specie) in ground_truth_P1:
                        truth_val=1
                    else:
                        truth_val= 0
                    #line="{},{},{},{},{},{},{},{},{} \n".format(k, crystal[k].specie, crystal[k].a, crystal[k].b, crystal[k].c, assign[k,0], assign[k,1], truth_val)
                    binary_targets.append(truth_val)
                    binary_feats.append(assign[k].numpy())
                    #

                C, score, margin, svc_pred= evaluate_pool(binary_feats, binary_targets)

                for k in range(len(crystal)):
                    if str(crystal[k].specie) in ground_truth_P1:
                        truth_val=1
                    else:
                        truth_val= 0
                    line="{},{},{},{},{},{},{},{},{} \n".format(k, crystal[k].specie, crystal[k].a, crystal[k].b, crystal[k].c, assign[k,0], assign[k,1], truth_val, svc_pred[k])
                    outfile.write(line)

                scores.append(score)
                Cs.append(C)
                class_accuracies_for_plot.append(score)

                if np.isnan(score):
                    num_imperfect+=1
                    mainline= cifs[i]+','+str(individual_error)+','+str(margin)+',0,'+str(score)+'\n'
                elif score<1:
                    num_imperfect+=1
                    mainline= cifs[i]+','+str(individual_error)+','+str(margin)+',0,'+str(score)+'\n'
                else:
                    num_perfect+=1
                    mainline= cifs[i]+','+str(individual_error)+','+str(margin)+',1,'+str(score)+'\n'
                outfile_main.write(mainline)
                line= 'pool accuracy='+str(score)+'C='+str(C)+'\n'
                outfile.write(line)

                line= '\n absolute error = '+str(float(individual_error))
                outfile.write(line)
                line= '\n '+ args.datadir+','+cifs[i]
                outfile.close()
                if score==1:
                    poolfile= args.datadir+cifs[i][:-7]+'pool.dat'
                    df= pd.read_csv(poolfile)
                    an_mean=np.mean(df[df['ground_truth_P1']==1.0]['P1'])
                    cat_mean=np.mean(df[df['ground_truth_P1']==0.0]['P1'])
                    if an_mean>cat_mean:
                        an_greater+=1
                    else:
                        cat_greater+=1

                i+=1

        plus75= [x for x in class_accuracies_for_plot if x>=0.75]
        #print('greater than 75% accuracy')
        #print(len(plus75)/len(class_accuracies_for_plot))
        pl.figure()
        pl.scatter(maes_for_plot, class_accuracies_for_plot, alpha=.33)
        pl.xlabel('Absolute error (eV/atom)')
        pl.ylabel('SVM classification accuracy')
        pl.title('Absolute error vs cation/anion classification accuracy')
        pl.savefig(main_checkpoint_path+'mae_vs_acc.png')
        pl.figure()
        pl.scatter(maes_for_plot, crystal_size, alpha=.33)
        pl.xlabel('Absolute error (eV/atom)')
        pl.ylabel('Num atoms in crystal')
        pl.title('Absolute error vs num atoms in crystal')
        pl.savefig(main_checkpoint_path+'mae_vs_size.png')
        pool_acc_vs_crystal_size(crystal_size,class_accuracies_for_plot, main_checkpoint_path)
        if args.task=='c':
            outs = tf.reduce_mean(sparse_categorical_accuracy(target, pred))

        elif args.task=='r':
            outs = tf.reduce_mean(mean_squared_error(target, pred)),

        output.append(outs)
        if step == loader.steps_per_epoch:
            output = np.array(output)

            return_dict={'MSE':np.average(output), 'RMSE':np.sqrt(np.average(output)), 'MAE':np.average(maes_for_plot), 'perfect_pools':num_perfect, 'imperfect_pools':num_imperfect, 'cat_greater':cat_greater, 'an_greater':an_greater}
            return return_dict

def pool_acc_vs_crystal_size(size,accuracies, savepath):
    pl.figure()
    pl.scatter(size, accuracies, alpha=.33)
    pl.xlabel('Num atoms in crystal')
    pl.ylabel('SVM classification accuracy')
    pl.title('Crystal size vs cation/anion classification accuracy')
    pl.savefig(savepath+'size_vs_acc.png')

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

    pl.show()

def evaluate_pool(binary_feats, binary_targets):
    binary_feats= np.array(binary_feats)
    scores= {}
    margins= {}
    preds= {}
    if len(np.unique(binary_feats, axis=0))==1:
        print('ITS ALL 1')
        return np.nan, 0, 0, [0]*len(binary_feats)
    else:

        for C_param in range(-5,8):
            C= 10**C_param
            clf = svm.SVC(C=C, kernel='linear', max_iter=10000)
            clf.fit(binary_feats, binary_targets)
            score= clf.score(binary_feats, binary_targets)
            pred= clf.predict(binary_feats)


            w =  clf.coef_[0]
            a = -w[0]/w[1]
            xx = np.linspace(-5, 5)
            yy = a*xx - (clf.intercept_[0])/w[1]
            margin = 1/np.sqrt(np.sum(clf.coef_**2))
            scores[C]= score
            margins[C]= margin
            preds[C]= pred
            if score==1:
                return C, score, margin, pred

        #pool_plots(binary_feats, binary_targets, clf)
        #df_new=pd.DataFrame(clf.decision_function(binary_feats)/np.sqrt(np.sum(clf.coef_**2)))
        #df_new.to_csv('dboundary.csv')
        best_score= max(scores.values())
        best_C= max(scores, key=scores.get)
        best_margin= margins[best_C]
        best_pred= preds[best_C]
        return best_C, best_score, best_margin, best_pred

def main(main_checkpoint_path):
    df_reference= pd.read_csv('../Main_fol_Zintl/ICSD_Zintl_TE_pooling.csv')
    args = parser.parse_args(sys.argv[1:])


    checkpoint_path = main_checkpoint_path+"goodmodel.ckpt.index"
    #checkpoint_path = "./train_model_2023-07-03_17-20-23/train_model_1285a2b1_3_batch_size=1,column_lambda=1219992.1467,dr1=0.5991,dr2=0.9550,embedding_size=2,entropy_lambda=17520562.6629_2023-07-03_17-20-41/goodmodel.ckpt.index"

    checkpoint_dir = os.path.dirname(checkpoint_path)
    #print(checkpoint_dir)
    val_df = pd.read_csv(os.path.join(args.datadir,'train.csv'), names=['id','target'], header=0)
    #val_df= val_df.head(10)
    #val_df = val_df.sample(frac=1).reset_index(drop=True)
    data= MyDataset(val_df, args.datadir, args.radius_angstroms, args.num_nbrs, args.task)
    loader_va= DisjointLoader(data, shuffle=False, batch_size=len(val_df))
    cifs=data.get_cifs()

    paramsdict= json.load(open(checkpoint_dir+'/params.json'))
    #print(paramsdict)
    #model= HNetConcatJanossy('r', 1, embedding_size=paramsdict['embedding_size'], d1=paramsdict['dr1'], el=paramsdict['entropy_lambda'], cl=paramsdict['column_lambda'], fc_num=paramsdict['fc_num'],fc_size=paramsdict['fc_size'], return_s=True)
    model= HNetConcatJanossy('r', 1, embedding_size=paramsdict['embedding_size'], d1=paramsdict['dr1'], el=paramsdict['entropy_lambda'], cl=paramsdict['column_lambda'], fc_size=paramsdict['fc_size'], return_s=True)

    print(model)

    latest = tf.train.latest_checkpoint(checkpoint_dir)
    model.load_weights(latest)
#
    result_dict=evaluate(loader_va ,model, cifs, df_reference, args, main_checkpoint_path)
    return result_dict

if __name__ == '__main__':
    crazylist=['train_model_a363460a_1_batch_size=8,column_lambda=2569099.8214,dr1=0.5332,embedding_size=12,entropy_lambda=6419201.9280,fc_size=14_2023-07-13_18-39-04']
    df_master_dict={}
    for path in crazylist:
        #print(path)
        fullpath='./zintl_janossy_constant/'+path+'/'
        #try:
        result_dict= main(fullpath)
        df_master_dict[fullpath]=result_dict
        #except:
        #    df_master_dict[fullpath]={}
    #df_master_dict= pd.DataFrame.from_dict(df_master_dict)
    #df_master_dict.to_csv('./janossy_fcs/large_eval.csv')
    #temp='./janossy_arch_longmodel_a363460a/'
    #
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
