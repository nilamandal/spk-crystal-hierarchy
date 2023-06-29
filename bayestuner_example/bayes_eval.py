import numpy as np
import pandas as pd
import os
import sys
import argparse
import time
import matplotlib.pyplot as plt

df= pd.DataFrame(columns=['filename', 'trialnum', 'batchsize', 'embeddingsize', 'dr1', 'dr2', 'cl', 'el', 'lr', 'val_error'])

def bayes_reader(filename):
    f= open(filename)
    f= f.readlines()
    for linenum in range(len(f)):
        line= f[linenum]
        if "Search: Running Trial" in line:
            if '#1\n' in line:
                val_loss=None
                i=1
                while val_loss==None:
                    if "val_loss:" in f[linenum+i]:
                        val_loss= float(f[linenum+i].split(': ')[1].strip())
                    else:
                        i+=1
                df.loc[len(df.index)] = [filename, 1, 8, 8, 0, 0, 0, 0, 1, val_loss]
            else:
                trialnum= line.split('#')[1].strip()
                embedding_size= f[linenum+3].split()[0]
                dr1= f[linenum+4].split()[0]
                dr2= f[linenum+5].split()[0]
                cl= f[linenum+6].split()[0]
                el= f[linenum+7].split()[0]
                lr= f[linenum+8].split()[0]
                val_loss=None
                train_history=[]
                val_history=[]
                epoch=[]
                i=1
                while val_loss==None:
                    if "val_loss:" in f[linenum+i]:
                        val_loss= float(f[linenum+i].split(': ')[1].strip())
                    elif 'train loss=' in f[linenum+i]:
                        mse= float(f[linenum+i].strip().split('=')[1])
                        train_history.append(mse)
                        i+=1
                    elif 'Epoch loss:' in f[linenum+i]:
                        mse= float(f[linenum+i].strip().split(':')[1])
                        val_history.append(mse)
                        i+=1
                    elif 'Epoch:' in f[linenum+i]:
                        mse= int(f[linenum+i].strip().split(':')[1])
                        epoch.append(mse)
                        i+=1
                    else:
                        i+=1
                middle_idx=[index for index, value in enumerate(epoch) if value == 0][1]

                plt.figure()
                plt.plot(epoch[:middle_idx], train_history[:middle_idx], label='training loss')
                plt.plot(epoch[:middle_idx], val_history[:middle_idx], label='val loss')
                titleline= filename+', '+trialnum
                figtitle='./plots/'+filename.split('_')[0]+'_'+trialnum+'_1result.png'
                plt.title(titleline, wrap=True)
                plt.xlabel('epochs')
                plt.legend()
                plt.ylabel('mean square error')
                plt.savefig(figtitle)

                plt.figure()
                plt.plot(epoch[middle_idx:], train_history[middle_idx:], label='training loss')
                plt.plot(epoch[middle_idx:], val_history[middle_idx:], label='val loss')
                titleline= filename+', '+trialnum
                figtitle='./plots/'+filename.split('_')[0]+'_'+trialnum+'_2result.png'
                plt.title(titleline, wrap=True)
                plt.xlabel('epochs')
                plt.legend()
                plt.ylabel('mean square error')
                plt.savefig(figtitle)

                df.loc[len(df.index)] = [filename, trialnum, 8, embedding_size, dr1, dr2, cl, el, lr, val_loss]


file_list=['33_bayes_2','51_bayes_2','83_bayes_2']
for file in file_list:
    bayes_reader(file+'/out.txt')
#print(df)
#df.to_csv('bayesresults2.csv')
