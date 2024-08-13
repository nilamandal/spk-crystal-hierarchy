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
from spektral_essential_objects import GaussianDistance,MyDataset, HNetEdgepool
from sklearn import svm
import pylab as pl
from tensorflow.keras import backend as K
from util_functions import entropy_loss, row_e_and_column_p

def evaluate(loader, model, cifs, df, fullpath_of_model, fullpath_of_data_file, write_output_path):
    output = []
    step = 0
    all_s=[]
    all_pre_feats=[]
    cif_idx=0
    i=0
    while step < loader.steps_per_epoch:
        step += 1
        inputs, target = loader.__next__()

        outfile_main=open(write_output_path+'/pooling_eval.csv','w+')
        outfile_main.write('name,pred, target,abs_error,pool_margin,perfect,avg_acc,row_entropy,neg_col_entropy,num_unique \n')

        pred, s_tensor = model(inputs, training=False)

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
                individual_error= np.abs(target[j]-pred[j])
                maes_for_plot.append(individual_error)
                crystal= Structure.from_file(os.path.join(fullpath_of_data_file,cifs[i]))
                crystal_size.append(len(crystal))

                ground_truth_P1= df[df['id']==cifs[i]].P1.values[0]

                tempcifname= cifs[i].split('/')[-1]

                savepath=os.path.join(write_output_path,os.path.dirname(cifs[i]))
                assign=assign[:len(crystal)]
                print(assign)
        #         try:
        #             outfile= open(savepath+'/'+tempcifname+'pool.csv', 'w+')
        #         except:
        #             if not os.path.exists(os.path.dirname(savepath+'/'+tempcifname+'pool.csv')):
        #                 os.makedirs(os.path.dirname(savepath+'/'+tempcifname+'pool.csv'))
        #             outfile= open(savepath+'/'+tempcifname+'pool.csv', 'w+')
        #         outfile.write('num,species,a,b,c,P1,P2,ground_truth_P1,SVM_pred_P1\n')
        #
        #         binary_feats=[]
        #         binary_targets=[]
        #         for k in range(len(crystal)):
        #             if str(crystal[k].specie) in ground_truth_P1:
        #                 truth_val=1
        #             else:
        #                 truth_val= 0
        #             #line="{},{},{},{},{},{},{},{},{} \n".format(k, crystal[k].specie, crystal[k].a, crystal[k].b, crystal[k].c, assign[k,0], assign[k,1], truth_val)
        #             binary_targets.append(truth_val)
        #             binary_feats.append(assign[k].numpy())
        #         pl.figure()
        #
        #         pl.scatter(np.array(binary_feats)[:,0], np.array(binary_feats)[:,1], c=binary_targets)
        #         pl.xlabel('P1 assignment')
        #         pl.ylabel('P2 assignment')
        #         pl.title(tempcifname)
        #         pl.savefig(savepath+'/'+tempcifname+'coloredbytarget.png')
        #
        #         #C, score, margin, svc_pred= evaluate_pool(binary_feats, binary_targets)
        #         column_entropy, row_entropy= row_e_and_column_p(np.array(binary_feats))
        #
        #         specieslist=[]
        #
        #         for k in range(len(crystal)):
        #             specieslist.append(crystal[k].specie)
        #
        #             if str(crystal[k].specie) in ground_truth_P1:
        #                 truth_val=1
        #             else:
        #                 truth_val= 0
        #
        #             line="{},{},{},{},{},{},{},{},{} \n".format(k, crystal[k].specie, crystal[k].a, crystal[k].b, crystal[k].c, assign[k,0], assign[k,1], truth_val, svc_pred[k])
        #             outfile.write(line)
        #
        #         pl.figure()
        #         coloroptions=['#214cc9', '#c20d42', '#287f1b']
        #         colorcode=0
        #         sizecode=100
        #         print(specieslist)
        #         specieslist= np.array(specieslist)
        #         species_unique= np.unique(specieslist)
        #         print(species_unique)
        #         for x in species_unique:
        #             useful_index= np.where(specieslist==x)[0]
        #             pl.scatter(np.array(binary_feats)[useful_index,0], np.array(binary_feats)[useful_index,1], s=sizecode, c=coloroptions[colorcode], label=x)
        #             colorcode+=1
        #             sizecode= sizecode/2
        #         print('---')
        #
        #         pl.xlabel('P1 assignment')
        #         pl.ylabel('P2 assignment')
        #         pl.legend()
        #         pl.title(tempcifname)
        #         pl.savefig(savepath+'/'+tempcifname+'coloredbyspecies.png')
        #
        #         scores.append(score)
        #         Cs.append(C)
        #         class_accuracies_for_plot.append(score)
        #
        #         if np.isnan(score):
        #             num_imperfect+=1
        #         #    mainline= cifs[i]+','+str(individual_error)+','+str(margin)+',0,'+str(score)+','+str(row_entropy)+','+str(column_entropy)+','+str(len(np.unique(np.around(binary_feats, 3), axis=0)))+'\n'
        #         elif score<1:
        #             num_imperfect+=1
        #         #    mainline= cifs[i]+','+str(individual_error)+','+str(margin)+',0,'+str(score)+','+str(row_entropy)+','+str(column_entropy)+','+str(len(np.unique(np.around(binary_feats, 3), axis=0)))+'\n'
        #         else:
        #             num_perfect+=1
        #         mainline= cifs[i]+','+str(float(pred[j]))+','+str(float(target[j]))+','+str(individual_error)+','+str(margin)+',1,'+str(score)+','+str(row_entropy)+','+str(column_entropy)+','+str(len(np.unique(np.around(binary_feats, 3), axis=0)))+'\n'
        #         outfile_main.write(mainline)
        #         line= 'pool accuracy='+str(score)+'C='+str(C)+'\n'
        #         outfile.write(line)
        #
        #         line= '\n absolute error = '+str(float(individual_error))
        #         outfile.write(line)
        #         line= '\n '+ fullpath_of_data_file+','+cifs[i]
        #         outfile.close()
        #         if score==1:
        #             poolfile= savepath+'/'+tempcifname+'pool.csv'
        #             df_pool= pd.read_csv(poolfile)
        #             an_mean=np.mean(df_pool[df_pool['ground_truth_P1']==1.0]['P1'])
        #             cat_mean=np.mean(df_pool[df_pool['ground_truth_P1']==0.0]['P1'])
        #             if an_mean>cat_mean:
        #                 an_greater+=1
        #             else:
        #                 cat_greater+=1
        #
        #         i+=1
        #
        # plus75= [x for x in class_accuracies_for_plot if x>=0.75]
        #
        # pl.figure()
        # pl.scatter(maes_for_plot, class_accuracies_for_plot, alpha=.33)
        # pl.xlabel('Absolute error (eV/atom)')
        # pl.ylabel('SVM classification accuracy')
        # pl.title('Absolute error vs cation/anion classification accuracy')
        # pl.savefig(write_output_path+'/mae_vs_acc.png')
        # pl.figure()
        # pl.scatter(maes_for_plot, crystal_size, alpha=.33)
        # pl.xlabel('Absolute error (eV/atom)')
        # pl.ylabel('Num atoms in crystal')
        # pl.title('Absolute error vs num atoms in crystal')
        # pl.savefig(write_output_path+'/mae_vs_size.png')
        # pool_acc_vs_crystal_size(crystal_size,class_accuracies_for_plot, write_output_path)
        #
        # outs = tf.reduce_mean(mean_squared_error(target, pred))
        #
        # output.append(outs)
        # if step == loader.steps_per_epoch:
        #     output = np.array(output)
        #
        #     return_dict={'MSE':np.average(output), 'RMSE':np.sqrt(np.average(output)), 'MAE':np.average(maes_for_plot), 'perfect_pools':num_perfect, 'imperfect_pools':num_imperfect, 'cat_greater':cat_greater, 'an_greater':an_greater}
        #     return return_dict



def main(fullpath_of_model, fullpath_of_data_file, write_output_path):
    checkpoint_path = fullpath_of_model+"goodmodel.ckpt.index"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    data_dir = os.path.dirname(fullpath_of_data_file)

    val_df = pd.read_csv(fullpath_of_data_file, header=0)
    data= MyDataset(val_df, data_dir, 8, 12, 'r')
    loader_va= DisjointLoader(data, shuffle=False, batch_size=len(val_df))
    cifs=data.get_cifs()

    paramsdict= json.load(open(checkpoint_dir+'/params.json'))

    model=HNetEdgepool('r', 1, embedding_size=paramsdict['embedding_size'], cgcnn_num2= paramsdict['cgcnn_2'], el=paramsdict['entropy_lambda'], cl=paramsdict['column_lambda'], return_s=True)

    latest = tf.train.latest_checkpoint(checkpoint_dir)
    model.load_weights(latest)
    if not os.path.exists(write_output_path):
        os.makedirs(write_output_path)
    result_dict=evaluate(loader_va, model, cifs, val_df, fullpath_of_model, os.path.dirname(fullpath_of_data_file), write_output_path)

    return result_dict

if __name__ == '__main__':
    #fullpath of model is the path to the DIRECTORY where the saved model is located.
    fullpath_of_model='../zintl_4/train_model_003c08ce_128_trial_index=3,batch_size=8,cgcnn_2=3,column_lambda=5888740.0127,embedding_size=8,entropy_lambda=3.2821,lr_2024-06-25_09-30-43/'
    #fullpath of data file is the path to the CSV FILE where the list of crystals and target values is stored.
    fullpath_of_data_file='../Main_fol_Zintl/val_no_metals.csv'
    #write output path is the DIRECTORY where you want the output files to be saved.
    #Best practice is to use a new directory every time you run this script, to avoid past results being overwritten.
    write_output_path='../zintl_4/train_model_003c08ce_128_trial_index=3,batch_size=8,cgcnn_2=3,column_lambda=5888740.0127,embedding_size=8,entropy_lambda=3.2821,lr_2024-06-25_09-30-43/val_no_metals/'
    result_dict= main(fullpath_of_model, fullpath_of_data_file, write_output_path)
