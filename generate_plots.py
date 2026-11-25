import pylab as pl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.rcParams.update({'font.size': 15})
plt.rc('axes', labelsize=15) #fontsize of the x and y labels
plt.rc('legend', fontsize=15) #fontsize of the legend

square_size= (6,6)
landscape_size= (6,2)
circle_size= 75

def pool_acc_vs_crystal_size(size,accuracies, savepath):
    pl.figure()
    pl.scatter(size, accuracies, alpha=.33)
    pl.xlabel('Num atoms in crystal')
    pl.ylabel('SVM classification accuracy')
    pl.title('Crystal size vs cation/anion classification accuracy')
    pl.savefig(write_output_path+'/size_vs_acc.png')
    
def atoms_per_crystal(df_path):
    pass

def target_range_plot():
    pass

def get_correlation_vals(path=None, df=[]):
    pass
    #r squared
    #spearmans
    #rmse and stdev of target
    #mae and baseline
    #mape and baseline

def pred_vs_target_bpool():
    df= pd.read_csv('../full_tern_varied_patience/p50_id143/test_set/pooling_eval_recheck.csv')
    plt.figure(figsize=square_size)
    plt.scatter(df['target'],df['pred'],s=circle_size, c='#369aff')
    plt.xlabel('Target Values (eV)')
    plt.ylabel('Predicted Values (eV)')
    #plt.legend()
    x = np.arange(np.min(df['target'])-.2, np.max(df['target'])+1)
    y = x
    plt.plot(x, y, c='#000000')
    #plt.show()
    plt.savefig('./standardized_figs/bpool_pred_v_targ.pdf', bbox_inches='tight', format="pdf")

def pred_vs_target_cgcnn():
    df= pd.read_csv('../full_cgcnn_all/83/test_set/predictions.csv')
    plt.figure(figsize=square_size)
    plt.scatter(df['target'],df['pred'],s=circle_size, c='#369aff')
    plt.xlabel('Target Values (eV)')
    plt.ylabel('Predicted Values (eV)')
    #plt.legend()
    x = np.arange(np.min(df['target'])-.2, np.max(df['target'])+1)
    y = x
    plt.plot(x, y, c='#000000')
    #plt.show()
    plt.savefig('./standardized_figs/cgcnn_pred_v_targ.pdf', bbox_inches='tight', format="pdf")

if __name__ == '__main__':
    #pred_vs_target_bpool()
    pred_vs_target_cgcnn()
