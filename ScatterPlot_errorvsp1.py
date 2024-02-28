#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 09:59:05 2024

@author: farihatahosin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import os
#from math import ceil
import csv

# Define the root folder path
root_folder = './testing_new_pooling_info_no_fere/'


def find_absolute_values(string):
    numbers = re.search(r'[-+]?\d*\.\d+|\d+', string)
    numbers = numbers.group()
    print(numbers)
    numbers = float(numbers)
    return numbers

# Function to process CSV files
def process_csv(file_path, foldername):
    df = pd.read_csv(file_path)
    #if not df['P1'].isnull().any() and not df['P2'].isnull().any():
    df['P1'] = np.round(df['P1'],2)
    df['P2'] = np.round(df['P2'],2)
    last_value = df['num'].iloc[-1]
    abs_error = find_absolute_values(last_value)
    df['abs_error'] = abs_error
    return df
    '''dfa.to_csv(foldername + '/scatter_plot.png')
    plt.scatter(df['P1'], df['abs_error'] )
    plt.xlabel('P1')
    plt.ylabel('Error')
    print(foldername + '/scatter_plot.png')
    plt.savefig(foldername + '/scatter_plot.png')'''

names=[]
errors=[]
num_clusters=[]
actuals=[]
# Walk through the root folder and process CSV files in subfolders
for foldername, subfolders, filenames in os.walk(root_folder):
    for filename in filenames:
        if filename.endswith('.csv'):
            if filename!='a.csv':
                file_path = os.path.join(foldername, filename)
                print("Processing:", file_path)
                #print("folder:", foldername)
                #print("filename:", filename)
                #output_csv = os.path.join(foldername,filename)
                dfa = process_csv(file_path, foldername)

                dfa= dfa.dropna()

                #print(dfa)
                num_clusters.append(len(np.unique(dfa['P1'])))
                actuals.append(np.unique(dfa['P1']))
                #print(np.unique(dfa['abs_error']))
                names.append(file_path)
                errors.append(np.unique(dfa['abs_error'])[0])
callback_df=pd.DataFrame({'filename':names,
                            'mae':errors,
                            'num_clusters': num_clusters,
                            'assignments': actuals})
callback_df.to_csv('num_clusters_vs_error_icsd_ubem.csv')
            # out_path = os.path.join(foldername, 'a.csv')
            # #print(Output, out_path)
            # #dfa.to_csv(out_path)
            # #plota = pd.read_csv(
            # plt.scatter( dfa['P1'],dfa['abs_error'])
            # plt.xlabel('P1')
            # plt.ylabel('Error')
            # print(foldername + '/scatter_plot.png')
            # plt.savefig(foldername + '/scatter_plot.png')
            # plt.show()
            # #print("output_csv")
            # #print(output_csv)
            # #dfa.to_csv(output_csv)
