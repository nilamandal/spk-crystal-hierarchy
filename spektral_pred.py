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
#parser.add_argument('--batch-size', dest='batch_size', type=int,
#                    help='Batch size.', default=256)

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
        outfile_main=open(main_checkpoint_path+'pooling_eval.csv','w+')
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

                scores.append(score)
                Cs.append(C)
                class_accuracies_for_plot.append(score)
                mainline= cifs[i]+','+str(individual_error)+','+str(margin)+'\n'
                outfile_main.write(mainline)

                if np.isnan(score):
                    num_imperfect+=1
                elif score<1:
                    num_imperfect+=1
                else:
                    num_perfect+=1

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
        # print('crystals pooled correctly')
        # print(num_perfect)
        # print('crystals_pooled incorrectly')
        # print(num_imperfect)
        # print('num cat greater P1')
        # print(cat_greater)
        # print("num an greater p1")
        # print(an_greater)
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
            # print('MSE:')
            # print(np.average(output))
            # print('RMSE:')
            # print(np.sqrt(np.average(output)))
            # print('MAE:')
            # print(np.average(maes_for_plot))
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
                return C, score, margin

        #pool_plots(binary_feats, binary_targets, clf)
        #df_new=pd.DataFrame(clf.decision_function(binary_feats)/np.sqrt(np.sum(clf.coef_**2)))
        #df_new.to_csv('dboundary.csv')
        best_score= max(scores.values())
        best_C= max(scores, key=scores.get)
        best_margin= margins[best_C]
        return best_C, best_score, best_margin

def main(main_checkpoint_path):
    df_reference= pd.read_csv('../Main_fol_Zintl/ICSD_Zintl_TE_pooling.csv')
    args = parser.parse_args(sys.argv[1:])


    checkpoint_path = main_checkpoint_path+"goodmodel.ckpt.index"
    #checkpoint_path = "./train_model_2023-07-03_17-20-23/train_model_1285a2b1_3_batch_size=1,column_lambda=1219992.1467,dr1=0.5991,dr2=0.9550,embedding_size=2,entropy_lambda=17520562.6629_2023-07-03_17-20-41/goodmodel.ckpt.index"

    checkpoint_dir = os.path.dirname(checkpoint_path)

    val_df = pd.read_csv(os.path.join(args.datadir,'val.csv'), names=['id','target'], header=0)
    #val_df = val_df.sample(frac=1).reset_index(drop=True)
    data= MyDataset(val_df, args.datadir, args.radius_angstroms, args.num_nbrs, args.task)
    loader_va= DisjointLoader(data, shuffle=False, batch_size=len(val_df))
    cifs=data.get_cifs()

    paramsdict= json.load(open(checkpoint_dir+'/params.json'))
    #print(paramsdict)
    model= HNetConcatJanossy('r', 1, embedding_size=paramsdict['embedding_size'], d1=paramsdict['dr1'], el=paramsdict['entropy_lambda'], cl=paramsdict['column_lambda'], fc_size=paramsdict['fc_size'], return_s=True)
#model= HNetConcat('r', 1, el=paramsdict['entropy_lambda'], cl=paramsdict['column_lambda'], return_s=True)
# model= HNetConcatJanossy('r', 1, return_s=True)
#
# # #sys.stdout = open('./debug_max_t.txt', 'w')
#
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    model.load_weights(latest)
#
    result_dict=evaluate(loader_va ,model, cifs, df_reference, args, main_checkpoint_path)
    return result_dict

if __name__ == '__main__':
    crazylist=['train_model_479f6796_41_batch_size=56,column_lambda=283.2342,dr1=0.9080,embedding_size=3,entropy_lambda=19689023.2878,fc_size=3,lr_2023-07-15_14-18-38', 'train_model_ca62b9b9_72_batch_size=116,column_lambda=391.8751,dr1=0.1808,embedding_size=2,entropy_lambda=2.7255,fc_size=116,lr=0.0_2023-07-16_14-45-07', 'train_model_c4ea3e80_5_batch_size=1,column_lambda=55.1488,dr1=0.1684,embedding_size=5,entropy_lambda=317097.0464,fc_size=9,lr=0.00_2023-07-13_20-10-39', 'train_model_89013358_67_batch_size=90,column_lambda=25.8785,dr1=0.6924,embedding_size=59,entropy_lambda=6875.4571,fc_size=10,lr=0._2023-07-16_06-50-54', 'train_model_ffbd2c35_93_batch_size=2,column_lambda=7983230.9073,dr1=0.7016,embedding_size=58,entropy_lambda=15656302.6949,fc_size=_2023-07-17_10-02-26', 'train_model_e134dca5_94_batch_size=54,column_lambda=1068.3220,dr1=0.0990,embedding_size=50,entropy_lambda=4.4834,fc_size=19,lr=0.0_2023-07-17_10-28-08', 'train_model_ad2578d6_78_batch_size=35,column_lambda=4148317.4686,dr1=0.4616,embedding_size=21,entropy_lambda=1079.9261,fc_size=3,l_2023-07-16_21-09-14', 'train_model_425dc757_68_batch_size=1,column_lambda=3553831.1087,dr1=0.1665,embedding_size=2,entropy_lambda=720339.9276,fc_size=25,_2023-07-16_11-20-37', 'train_model_f60fc428_15_batch_size=1,column_lambda=29.4570,dr1=0.7119,embedding_size=114,entropy_lambda=19130907.8210,fc_size=6,lr_2023-07-14_06-02-16', 'train_model_94c74810_55_batch_size=12,column_lambda=32.5851,dr1=0.8909,embedding_size=4,entropy_lambda=21978.0829,fc_size=4,lr=0.0_2023-07-16_00-55-25', 'train_model_7652f61f_32_batch_size=2,column_lambda=1.3833,dr1=0.5774,embedding_size=35,entropy_lambda=10717.5972,fc_size=1,lr=0.00_2023-07-15_03-38-08', 'train_model_0f0c06e5_86_batch_size=1,column_lambda=367382.3462,dr1=0.7413,embedding_size=127,entropy_lambda=6412.4454,fc_size=1,lr_2023-07-17_03-26-58', 'train_model_bf40feec_47_batch_size=10,column_lambda=1946.8368,dr1=0.3058,embedding_size=1,entropy_lambda=2139480.5031,fc_size=37,l_2023-07-15_18-41-55', 'train_model_20a6b458_26_batch_size=1,column_lambda=4087.9084,dr1=0.4605,embedding_size=66,entropy_lambda=35132985.4878,fc_size=1,l_2023-07-14_16-50-54', 'train_model_a38485ee_4_batch_size=7,column_lambda=854.4434,dr1=0.7738,embedding_size=112,entropy_lambda=5553.2787,fc_size=4,lr=0.0_2023-07-13_18-39-22', 'train_model_b2ce4a5d_35_batch_size=1,column_lambda=2.4166,dr1=0.4685,embedding_size=3,entropy_lambda=35331.5798,fc_size=1,lr=0.002_2023-07-15_07-06-13', 'train_model_d195cec3_19_batch_size=3,column_lambda=8.5487,dr1=0.4247,embedding_size=1,entropy_lambda=18937.1479,fc_size=20,lr=0.00_2023-07-14_10-14-02', 'train_model_f603ba31_14_batch_size=1,column_lambda=1.5736,dr1=0.7992,embedding_size=1,entropy_lambda=13.7532,fc_size=2,lr=0.0554_2023-07-14_05-56-12', 'train_model_9f9a3944_92_batch_size=37,column_lambda=96056516.8967,dr1=0.6529,embedding_size=4,entropy_lambda=58.9143,fc_size=4,lr=_2023-07-17_09-02-55', 'train_model_a5322807_12_batch_size=27,column_lambda=662.7352,dr1=0.3513,embedding_size=106,entropy_lambda=2.9061,fc_size=1,lr=0.00_2023-07-14_04-32-37', 'train_model_e9d4537f_13_batch_size=76,column_lambda=43.6357,dr1=0.0237,embedding_size=15,entropy_lambda=17.2857,fc_size=47,lr=0.00_2023-07-14_04-58-29', 'train_model_4964170d_56_batch_size=12,column_lambda=192638.3451,dr1=0.4595,embedding_size=11,entropy_lambda=1174.9795,fc_size=24,l_2023-07-16_01-16-46', 'train_model_3d8a36ae_76_batch_size=65,column_lambda=181.9498,dr1=0.5039,embedding_size=2,entropy_lambda=2612.5315,fc_size=69,lr=0._2023-07-16_20-16-36', 'train_model_f76d13a7_31_batch_size=76,column_lambda=9.4574,dr1=0.5636,embedding_size=8,entropy_lambda=22701251.5997,fc_size=1,lr=0_2023-07-15_02-15-01', 'train_model_dba29b36_25_batch_size=16,column_lambda=1953910.6855,dr1=0.3003,embedding_size=8,entropy_lambda=2701037.1315,fc_size=3_2023-07-14_16-32-02', 'train_model_d464dad1_22_batch_size=15,column_lambda=598.0190,dr1=0.1352,embedding_size=1,entropy_lambda=703601.2199,fc_size=2,lr=0_2023-07-14_15-15-18', 'train_model_ccb7a84d_84_batch_size=4,column_lambda=179310.1654,dr1=0.0558,embedding_size=18,entropy_lambda=2939.3681,fc_size=5,lr=_2023-07-17_00-49-37', 'train_model_2e8b2d63_87_batch_size=3,column_lambda=182444.5326,dr1=0.4160,embedding_size=4,entropy_lambda=2533.6762,fc_size=5,lr=0_2023-07-17_03-51-02', 'train_model_f2d27791_20_batch_size=7,column_lambda=0.1312,dr1=0.8443,embedding_size=1,entropy_lambda=8.7019,fc_size=3,lr=0.0471_2023-07-14_13-07-44', 'train_model_489c622b_75_batch_size=32,column_lambda=1055939.7071,dr1=0.6379,embedding_size=13,entropy_lambda=4954266.2629,fc_size=_2023-07-16_19-58-48', 'train_model_06ea7517_48_batch_size=8,column_lambda=49.5576,dr1=0.0073,embedding_size=1,entropy_lambda=5652.7662,fc_size=50,lr=0.01_2023-07-15_19-30-43', 'train_model_28da5356_29_batch_size=3,column_lambda=10892.9106,dr1=0.5725,embedding_size=1,entropy_lambda=3876.4359,fc_size=1,lr=0._2023-07-15_00-35-53', 'train_model_5eae0835_79_batch_size=18,column_lambda=3155.0372,dr1=0.4065,embedding_size=11,entropy_lambda=656305.2276,fc_size=26,l_2023-07-16_21-35-55', 'train_model_02c8b645_53_batch_size=68,column_lambda=961350.7327,dr1=0.4790,embedding_size=14,entropy_lambda=39675.7904,fc_size=16,_2023-07-15_22-24-24', 'train_model_f2bad073_8_batch_size=3,column_lambda=6582077.3454,dr1=0.9051,embedding_size=1,entropy_lambda=0.6799,fc_size=7,lr=0.00_2023-07-13_22-34-20', 'train_model_195414df_46_batch_size=30,column_lambda=1.5149,dr1=0.8239,embedding_size=60,entropy_lambda=0.6678,fc_size=3,lr=0.0000_2023-07-15_18-03-16', 'train_model_ae1e6b22_91_batch_size=14,column_lambda=601071.1477,dr1=0.8581,embedding_size=23,entropy_lambda=3222.6094,fc_size=1,lr_2023-07-17_08-16-53', 'train_model_c7f2f7bb_58_batch_size=28,column_lambda=64219.2381,dr1=0.9155,embedding_size=52,entropy_lambda=37.8365,fc_size=9,lr=0._2023-07-16_02-22-05', 'train_model_f0a7930e_40_batch_size=19,column_lambda=0.5437,dr1=0.5617,embedding_size=111,entropy_lambda=0.2981,fc_size=9,lr=0.0014_2023-07-15_13-29-37', 'train_model_a13c8858_81_batch_size=3,column_lambda=3204482.7885,dr1=0.1469,embedding_size=50,entropy_lambda=2.7876,fc_size=4,lr=0._2023-07-16_22-46-19', 'train_model_483dc74c_38_batch_size=1,column_lambda=25299.0134,dr1=0.0445,embedding_size=1,entropy_lambda=935211.8736,fc_size=1,lr=_2023-07-15_12-10-23', 'train_model_9b876682_10_batch_size=2,column_lambda=9.2355,dr1=0.6464,embedding_size=73,entropy_lambda=159.9632,fc_size=50,lr=0.000_2023-07-14_01-45-44', 'train_model_a30a4dbd_63_batch_size=112,column_lambda=52.3443,dr1=0.5127,embedding_size=30,entropy_lambda=1265.0553,fc_size=3,lr=0._2023-07-16_04-54-45', 'train_model_049eae52_45_batch_size=23,column_lambda=1580.8779,dr1=0.9634,embedding_size=11,entropy_lambda=318934.6593,fc_size=10,l_2023-07-15_17-37-31', 'train_model_c64df619_27_batch_size=1,column_lambda=141.2777,dr1=0.1423,embedding_size=4,entropy_lambda=58.2090,fc_size=1,lr=0.0068_2023-07-14_17-36-10', 'train_model_70a98b16_39_batch_size=14,column_lambda=1.4908,dr1=0.7022,embedding_size=4,entropy_lambda=34860.0544,fc_size=12,lr=0.0_2023-07-15_12-58-22', 'train_model_a7d0f2f2_83_batch_size=3,column_lambda=2.6600,dr1=0.7863,embedding_size=17,entropy_lambda=51863002.3891,fc_size=1,lr=0_2023-07-16_23-24-24', 'train_model_8308dad4_65_batch_size=2,column_lambda=6.6033,dr1=0.2269,embedding_size=41,entropy_lambda=16614180.9693,fc_size=2,lr=0_2023-07-16_06-23-47', 'train_model_b99cbf01_50_batch_size=122,column_lambda=33460.5169,dr1=0.1372,embedding_size=8,entropy_lambda=337956.9274,fc_size=1,l_2023-07-15_21-03-55', 'train_model_d0f7a3ec_89_batch_size=1,column_lambda=1.8011,dr1=0.4240,embedding_size=5,entropy_lambda=107566.9052,fc_size=43,lr=0.0_2023-07-17_04-49-53', 'train_model_624ebd6b_66_batch_size=1,column_lambda=15999.8684,dr1=0.9113,embedding_size=1,entropy_lambda=0.1702,fc_size=15,lr=0.00_2023-07-16_06-47-37', 'train_model_aad439ad_36_batch_size=62,column_lambda=48.0023,dr1=0.4862,embedding_size=2,entropy_lambda=2937.9551,fc_size=17,lr=0.0_2023-07-15_08-20-09', 'train_model_8cb871e7_82_batch_size=2,column_lambda=6027.3146,dr1=0.1900,embedding_size=1,entropy_lambda=16.9379,fc_size=101,lr=0.0_2023-07-16_22-56-20', 'train_model_11aa4a25_34_batch_size=2,column_lambda=393278.8117,dr1=0.8230,embedding_size=64,entropy_lambda=25719526.3039,fc_size=3_2023-07-15_05-23-03', 'train_model_5978d1e9_30_batch_size=3,column_lambda=7715959.7516,dr1=0.0145,embedding_size=33,entropy_lambda=0.2041,fc_size=1,lr=0._2023-07-15_00-55-58', 'train_model_84b26f7b_11_batch_size=121,column_lambda=462.5819,dr1=0.7885,embedding_size=35,entropy_lambda=1100734.5598,fc_size=68,_2023-07-14_01-51-11', 'train_model_01b66100_73_batch_size=1,column_lambda=12532.0172,dr1=0.0673,embedding_size=124,entropy_lambda=865.2115,fc_size=1,lr=0_2023-07-16_14-50-28', 'train_model_1300ae8f_71_batch_size=1,column_lambda=20900.4679,dr1=0.9698,embedding_size=1,entropy_lambda=381.1516,fc_size=2,lr=0.0_2023-07-16_12-40-15', 'train_model_5739b347_80_batch_size=18,column_lambda=3.8992,dr1=0.2548,embedding_size=1,entropy_lambda=1942.0095,fc_size=1,lr=0.000_2023-07-16_22-11-16', 'train_model_e95a0b82_49_batch_size=8,column_lambda=325.4905,dr1=0.7746,embedding_size=15,entropy_lambda=0.1584,fc_size=17,lr=0.000_2023-07-15_20-29-53', 'train_model_9a24f515_24_batch_size=1,column_lambda=7812.6917,dr1=0.9276,embedding_size=23,entropy_lambda=2290.4923,fc_size=23,lr=0_2023-07-14_15-51-34', 'train_model_278c2d5a_6_batch_size=2,column_lambda=7.2818,dr1=0.3462,embedding_size=11,entropy_lambda=351759.8151,fc_size=17,lr=0.0_2023-07-13_21-16-58', 'train_model_0d8674ab_57_batch_size=2,column_lambda=1.0619,dr1=0.0918,embedding_size=5,entropy_lambda=89691623.7465,fc_size=21,lr=0_2023-07-16_02-10-20', 'train_model_471f4ab6_16_batch_size=66,column_lambda=23054833.4660,dr1=0.1965,embedding_size=12,entropy_lambda=473203.3735,fc_size=_2023-07-14_06-20-21', 'train_model_011ecd63_42_batch_size=9,column_lambda=911287.4793,dr1=0.0896,embedding_size=83,entropy_lambda=4004.1832,fc_size=1,lr=_2023-07-15_15-34-30', 'train_model_72bdd0c6_3_batch_size=4,column_lambda=0.1479,dr1=0.8783,embedding_size=4,entropy_lambda=8318575.4999,fc_size=1,lr=0.03_2023-07-13_18-39-16', 'train_model_c0333bfb_62_batch_size=11,column_lambda=4.5463,dr1=0.5201,embedding_size=13,entropy_lambda=4823261.0454,fc_size=5,lr=0_2023-07-16_04-35-29', 'train_model_b2e8c50a_7_batch_size=62,column_lambda=3.3611,dr1=0.2353,embedding_size=3,entropy_lambda=34411.7917,fc_size=10,lr=0.00_2023-07-13_21-17-24', 'train_model_3143c021_60_batch_size=8,column_lambda=989.2617,dr1=0.1844,embedding_size=12,entropy_lambda=2383.7449,fc_size=2,lr=0.0_2023-07-16_03-13-11', 'train_model_faeb0bfd_52_batch_size=3,column_lambda=14.9630,dr1=0.0684,embedding_size=6,entropy_lambda=14982452.1321,fc_size=5,lr=0_2023-07-15_21-58-43', 'train_model_10b71be9_21_batch_size=1,column_lambda=94899195.6840,dr1=0.9151,embedding_size=8,entropy_lambda=82566.1171,fc_size=115_2023-07-14_13-58-34', 'train_model_748e2549_51_batch_size=2,column_lambda=2962.2394,dr1=0.3073,embedding_size=19,entropy_lambda=462.9800,fc_size=1,lr=0.0_2023-07-15_21-40-45', 'train_model_a468d652_43_batch_size=1,column_lambda=3455.2417,dr1=0.9206,embedding_size=1,entropy_lambda=79243680.8388,fc_size=36,l_2023-07-15_15-56-16', 'train_model_d510e6ea_64_batch_size=87,column_lambda=623.0308,dr1=0.2780,embedding_size=24,entropy_lambda=135010.0338,fc_size=43,lr_2023-07-16_05-57-53', 'train_model_9b9da313_33_batch_size=1,column_lambda=4.6028,dr1=0.4046,embedding_size=10,entropy_lambda=0.8945,fc_size=1,lr=0.0076_2023-07-15_03-57-44', 'train_model_44025395_90_batch_size=42,column_lambda=1.9648,dr1=0.7138,embedding_size=80,entropy_lambda=12378.4840,fc_size=1,lr=0.0_2023-07-17_06-58-58', 'train_model_a363460a_1_batch_size=8,column_lambda=2569099.8214,dr1=0.5332,embedding_size=12,entropy_lambda=6419201.9280,fc_size=14_2023-07-13_18-39-04', 'train_model_d427d684_74_batch_size=68,column_lambda=820999.9963,dr1=0.0339,embedding_size=1,entropy_lambda=17.8694,fc_size=6,lr=0._2023-07-16_15-06-27', 'train_model_3250549b_17_batch_size=3,column_lambda=13737311.1636,dr1=0.2469,embedding_size=3,entropy_lambda=5.6911,fc_size=84,lr=0_2023-07-14_07-03-20', 'train_model_daaa3218_2_batch_size=6,column_lambda=498463.3019,dr1=0.7440,embedding_size=124,entropy_lambda=45086.7653,fc_size=103,_2023-07-13_18-39-10', 'train_model_a7cf74f4_77_batch_size=69,column_lambda=3.4050,dr1=0.9298,embedding_size=24,entropy_lambda=97069.4477,fc_size=104,lr=0_2023-07-16_20-49-35', 'train_model_f01173ba_54_batch_size=6,column_lambda=12115.4111,dr1=0.2191,embedding_size=2,entropy_lambda=91493.5748,fc_size=73,lr=_2023-07-16_00-32-57', 'train_model_5319c7bb_70_batch_size=4,column_lambda=2951618.3149,dr1=0.2115,embedding_size=1,entropy_lambda=15675160.5129,fc_size=1_2023-07-16_12-15-40', 'train_model_aa8f8866_88_batch_size=4,column_lambda=0.4765,dr1=0.8171,embedding_size=127,entropy_lambda=221376.8877,fc_size=4,lr=0._2023-07-17_04-04-48', 'train_model_be711b18_28_batch_size=99,column_lambda=534.7807,dr1=0.3203,embedding_size=13,entropy_lambda=0.1627,fc_size=22,lr=0.00_2023-07-14_23-11-14', 'train_model_fa2ba944_69_batch_size=56,column_lambda=0.9247,dr1=0.6230,embedding_size=15,entropy_lambda=22895.5566,fc_size=1,lr=0.0_2023-07-16_11-56-28', 'train_model_c74f9d9b_23_batch_size=13,column_lambda=107.9339,dr1=0.6354,embedding_size=23,entropy_lambda=337811.8115,fc_size=7,lr=_2023-07-14_15-49-22', 'train_model_aa24ca4f_37_batch_size=25,column_lambda=2.8251,dr1=0.6530,embedding_size=54,entropy_lambda=127164.8476,fc_size=2,lr=0._2023-07-15_11-49-22', 'train_model_861a5c6a_44_batch_size=59,column_lambda=40121371.2088,dr1=0.4547,embedding_size=14,entropy_lambda=29224376.0340,fc_siz_2023-07-15_16-29-29', 'train_model_8700397c_18_batch_size=4,column_lambda=994933.5654,dr1=0.6650,embedding_size=78,entropy_lambda=39123924.3844,fc_size=1_2023-07-14_07-26-11', 'train_model_e81c81eb_61_batch_size=94,column_lambda=3079421.5531,dr1=0.8300,embedding_size=3,entropy_lambda=119.6178,fc_size=22,lr_2023-07-16_03-16-43', 'train_model_522d73bb_85_batch_size=22,column_lambda=38769.1519,dr1=0.5129,embedding_size=7,entropy_lambda=95.4817,fc_size=10,lr=0._2023-07-17_02-24-32', 'train_model_2641a02e_59_batch_size=1,column_lambda=6.6747,dr1=0.5670,embedding_size=76,entropy_lambda=14639.9544,fc_size=28,lr=0.0_2023-07-16_02-27-37', 'train_model_56faf0e3_9_batch_size=3,column_lambda=1906180.9122,dr1=0.8071,embedding_size=2,entropy_lambda=183.0639,fc_size=7,lr=0._2023-07-13_22-58-22']
    df_master_dict={}
    for path in crazylist:
        fullpath='./janossy_first_94/'+path+'/'
        try:
            result_dict= main(fullpath)
            df_master_dict[fullpath]=result_dict
        except:
            df_master_dict[fullpath]={}
    df_master_dict= pd.DataFrame.from_dict(df_master_dict)
    df_master_dict.to_csv('./check_if_pred_script_works_correctly.csv')
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
