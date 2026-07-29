from spektral.data import Dataset, DisjointLoader
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import MeanSquaredError, SparseCategoricalCrossentropy
import numpy as np
import pandas as pd
import os
import sys
from pymatgen.core.structure import Structure
import json
from spektral_essential_objects import GaussianDistance, MyDataset, SparseEdgepool, AtomFeaDataset, CGCNNModel, NotShrinking
from sklearn import svm
import pylab as pl
from tensorflow.keras import backend as K

def mean_squared_error(real, pred):
   return np.mean((real-pred)**2)


#This function handles all evaluation of the data. It computes the model's prediction for each crystal,
#the error for each crystal, and the poolings for each crystal. The prediction, error value, and a performance
#metric for pooling based on linear SVM will be written to the csv file "pooling_eval.csv".
#It also writes, for each crystal:
#1. contcarname_pool.csv; This file contains species, atom id, coordinates in the unit cell, learned assignment
#   values (P1 and P2), and the ground truth values for P1 according to the CSM team's approximate guidelines
#2. A plot of the P1 and P2 assignment values, colored by "ground truth" assignment of each atom
#3. A plot of the P1 and P2 assignment values, colored by element of each atom
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
        outfile_main.write('name,pred,target,perfect,num_unique \n')

        pred, s_tensor = model(inputs, training=False)

        num_perfect=0
        list_perfect=[]
        num_imperfect=0
        cat_greater=0
        an_greater=0

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
                try:
                    outfile= open(savepath+'/'+tempcifname+'pool.csv', 'w+')
                except:
                    if not os.path.exists(os.path.dirname(savepath+'/'+tempcifname+'pool.csv')):
                        os.makedirs(os.path.dirname(savepath+'/'+tempcifname+'pool.csv'))
                    outfile= open(savepath+'/'+tempcifname+'pool.csv', 'w+')
                outfile.write('num,species,a,b,c,P1,P2,ground_truth_P1\n')
###
                binary_feats=[]
                binary_targets=[]
                for k in range(len(crystal)):
                    if str(crystal[k].specie) in ground_truth_P1:
                        truth_val=1
                    else:
                        truth_val= 0
                    binary_targets.append(truth_val)
                    binary_feats.append(assign[k].numpy())
                pl.figure()

                pl.scatter(np.array(binary_feats)[:,0], np.array(binary_feats)[:,1], c=binary_targets)
                pl.xlabel('P1 assignment')
                pl.ylabel('P2 assignment')
                pl.title(tempcifname)
                pl.savefig(savepath+'/'+tempcifname+'coloredbytarget.png')


                specieslist=[]

                for k in range(len(crystal)):
                    specieslist.append(crystal[k].specie)

                    if str(crystal[k].specie) in ground_truth_P1:
                        truth_val=1
                    else:
                        truth_val= 0

                    line="{},{},{},{},{},{},{},{} \n".format(k, crystal[k].specie, crystal[k].a, crystal[k].b, crystal[k].c, assign[k,0], assign[k,1], truth_val,)
                    outfile.write(line)

                pl.figure()
                coloroptions=['#214cc9', '#c20d42', '#287f1b', '#d8c800']
                colorcode=0
                sizecode=100

                specieslist= np.array(specieslist)
                species_unique= np.unique(specieslist)

                for x in species_unique:
                    useful_index= np.where(specieslist==x)[0]
                    pl.scatter(np.array(binary_feats)[useful_index,0], np.array(binary_feats)[useful_index,1], s=sizecode, c=coloroptions[colorcode], label=x)
                    colorcode+=1
                    sizecode= sizecode/2

                pl.xlabel('P1 assignment')
                pl.ylabel('P2 assignment')
                pl.legend()
                pl.title(tempcifname)
                pl.savefig(savepath+'/'+tempcifname+'coloredbyspecies.png')


                line= '\n '+ fullpath_of_data_file+','+cifs[i]
                outfile.close()
                poolfile= savepath+'/'+tempcifname+'pool.csv'
                df_pool= pd.read_csv(poolfile)
                df_pool=df_pool.dropna()

                df_pool['heuristic_match']= df_pool['P1']==df_pool['ground_truth_P1']
                df_pool.to_csv(poolfile)

                perfect=0
                if df_pool['heuristic_match'].all():
                    perfect=1
                    num_perfect+=1
                else:
                    num_imperfect+=1
                mainline= cifs[i]+','+str(float(pred[j]))+','+str(float(target[j]))+','+str(perfect)+','+str(len(np.unique(np.around(binary_feats, 3), axis=0)))+'\n'
                outfile_main.write(mainline)
                i+=1

        outs = tf.reduce_mean(mean_squared_error(target, pred))

        output.append(outs)
        if step == loader.steps_per_epoch:
            output = np.array(output)

            return_dict={'MSE':np.average(output), 'RMSE':np.sqrt(np.average(output)), 'MAE':np.average(maes_for_plot), 'perfect_pools':num_perfect, 'imperfect_pools':num_imperfect, 'cat_greater':cat_greater, 'an_greater':an_greater}
            return return_dict

def eval_for_3_pools(loader, model, cifs, df, fullpath_of_model, fullpath_of_data_file, write_output_path):
    #evaluation function for models where k=3. These experiments are not included in 2026 paper.
    output = []
    step = 0
    all_s=[]
    all_pre_feats=[]
    cif_idx=0
    while step < loader.steps_per_epoch:
        step += 1
        inputs, target = loader.__next__()

        pred, s_tensor = model(inputs, training=False)
        mse = tf.reduce_mean((target-pred)**2)
        rmse= np.sqrt(mse)
        mae= tf.reduce_mean(np.abs(target-pred))
        print('MSE:')
        print(mse)
        print('RMSE:')
        print(rmse)
        print('MAE')
        print(mae)
        scores=[]
        Cs=[]
        class_accuracies_for_plot=[]
        crystal_size=[]

        for j in range(len(s_tensor)):
            assign= s_tensor[j]
            print(target[j],pred[j],np.abs(target[j]-pred[j]))
            individual_error= np.abs(target[j]-pred[j])
            crystal= Structure.from_file(os.path.join(fullpath_of_data_file,cifs[j]))
            crystal_size.append(len(crystal))

            ground_truth_P1= df[df['id']==cifs[j]].P1_new.values[0]
            ground_truth_P2= df[df['id']==cifs[j]].P2_new.values[0]
            tempcifname= cifs[j].split('/')[-1]
            savepath=os.path.join(write_output_path,os.path.dirname(cifs[j]))
            assign=assign[:len(crystal)]

            try:
                outfile= open(savepath+'/'+tempcifname+'_pool.csv', 'w+')
            except:
                if not os.path.exists(os.path.dirname(savepath+'/'+tempcifname+'_pool.csv')):
                    os.makedirs(os.path.dirname(savepath+'/'+tempcifname+'_pool.csv'))
                outfile= open(savepath+'/'+tempcifname+'_pool.csv', 'w+')
            outfile.write('num,species,a,b,c,P1,P2,P3\n')

            for k in range(len(crystal)):
                line="{},{},{},{},{},{},{},{} \n".format(k, crystal[k].specie, crystal[k].a, crystal[k].b, crystal[k].c, assign[k,0], assign[k,1], assign[k,2])
                outfile.write(line)
            outfile.close()

def eval_cgcnn(loader, model, write_path):
   #function to evaluate basic CGCNN results, which have no poolings.
   step = 0
   while step < loader.steps_per_epoch:
        step += 1
        inputs, target = loader.__next__()

        pred = model(inputs, training=False)
        df= pd.DataFrame(
             data=list(zip(pred, target)),
             columns=["pred", "target"]
        )
        df.to_csv(write_path+'predictions.csv')
   return mean_squared_error(target, pred)

def main(fullpath_of_model, fullpath_of_data_file, write_output_path, parampath):
    #load hyperparameter files
    checkpoint_path = fullpath_of_model+"goodmodel.keras"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    data_dir = os.path.dirname(fullpath_of_data_file)
    config= json.load(open(parampath+'params.json'))

    #load data
    val_df = pd.read_csv(fullpath_of_data_file, header=0)
    mini= val_df.head(2)
    minidata= AtomFeaDataset(mini, data_dir, 8, 12, 'r')
    miniloader= DisjointLoader(minidata, shuffle=False, batch_size=len(val_df))

    data= AtomFeaDataset(val_df, data_dir, 8, 12, 'r')
    loader_va= DisjointLoader(data, shuffle=False, batch_size=len(val_df))
    cifs=data.get_cifs()

    if not os.path.exists(write_output_path):
        os.makedirs(write_output_path)
    model= NotShrinking('r', 1, config['embedding_size'], config['cgcnn_num'], config['cgcnn_num2'], softmax_beta=config['softmax_beta'], k=2)
    model.return_s= True
    #model= CGCNNModel(config['embedding_size'], config['hidden_size'], config['num_layers'])ß

    #initialize model prior to loading weights
    mini_input, mini_target= miniloader.__next__()
    pred, s_tensor = model(mini_input, training=False)
    model.summary()
    model.load_weights(fullpath_of_model+'goodmodel.weights.h5')

    result_dict=evaluate(loader_va, model, cifs, val_df, fullpath_of_model, os.path.dirname(fullpath_of_data_file), write_output_path)

    return result_dict


if __name__ == '__main__':
    subpaths=['../full_tern_varied_patience/p50_id143/']

    #fullpath of data file is the path to the CSV FILE where the list of crystals and target values is stored.
    fullpath_of_data_file='../Main_fol_Zintl/quat_2_cat.csv'
    #fullpath_of_data_file='../Main_fol_Zintl/Zintl_bonding_analysis_new_heuristic.csv'

    for pathstring in subpaths:
        pathstring= str(pathstring)
        #fullpath of model is the path to the DIRECTORY where the saved model is located.
        fullpath_of_model= './'+pathstring+'/'
        parampath_for_model= fullpath_of_model

        #write output path is the DIRECTORY where you want the output files to be saved.
        #Best practice is to use a new directory every time you run this script, to avoid past results being overwritten.
        write_output_path=fullpath_of_model+'quaternary_2electro/'

        result_dict= main(fullpath_of_model, fullpath_of_data_file, write_output_path, parampath_for_model)
