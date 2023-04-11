import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df=pd.read_csv('bayesresults.csv')
print(df['lr'])
lr_list=df['lr'].to_list()
for i in range(len(lr_list)):
    try:
        lr_list[i]=float(lr_list[i])
    except:
        print(lr_list[i])
df['lr']=pd.to_numeric(df['lr'])

###
plt.figure(figsize=(24, 16))
plt.rc('font', size=40)
plt.scatter(df['lr'], df['val_error'], s=100)
plt.title('LR vs Avg. Val MSE across folds')
plt.xlabel('negative log_10 of learning rate')
plt.ylabel('Val MSE')
plt.savefig('bayes/lr.png')


plt.figure(figsize=(24, 16))
plt.rc('font', size=40)
plt.scatter(df['lr'], df['val_error'], s=100)
plt.ylim(0,1)
plt.title('LR vs Avg. Val MSE across folds')
plt.xlabel('negative log_10 of learning rate')
plt.ylabel('Val MSE')
plt.savefig('bayes/lr_limited.png')

# ###
# plt.figure(figsize=(24, 16))
# plt.rc('font', size=40)
# plt.scatter(df['bs'], df['val_error'], s=100)
# plt.title('Batch size vs Avg. Val MSE across folds')
# plt.xlabel('Batch size')
# plt.ylabel('Val MSE')
# plt.savefig('bayes/bs.png')
#
#
# plt.figure(figsize=(24, 16))
# plt.rc('font', size=40)
# plt.scatter(df['bs'], df['val_error'], s=100)
# plt.ylim(0,1)
# plt.title('Batch size vs Avg. Val MSE across folds')
# plt.xlabel('Batch size')
# plt.ylabel('Val MSE')
# plt.savefig('bayes/bs_limited.png')

###
plt.figure(figsize=(24, 16))
plt.rc('font', size=40)
plt.scatter(df['dr1'], df['val_error'], s=100)
plt.title('Dropout rate #1 vs Avg. Val MSE across folds')
plt.xlabel('Dropout rate')
plt.ylabel('Val MSE')
plt.savefig('bayes/dr1.png')

plt.figure(figsize=(24, 16))
plt.rc('font', size=40)
plt.scatter(df['dr1'], df['val_error'], s=100)
plt.ylim(0,1)
plt.title('Dropout rate #1 vs Avg. Val MSE across folds')
plt.xlabel('Dropout rate')
plt.ylabel('Val MSE')
plt.savefig('bayes/dr1_limited.png')

###
plt.figure(figsize=(24, 16))
plt.rc('font', size=40)
plt.scatter(df['dr2'], df['val_error'], s=100)
plt.title('Dropout rate #2 vs Avg. Val MSE across folds')
plt.xlabel('Dropout rate')
plt.ylabel('Val MSE')
plt.savefig('bayes/dr2.png')

plt.figure(figsize=(24, 16))
plt.rc('font', size=40)
plt.scatter(df['dr2'], df['val_error'], s=100)
plt.ylim(0,1)
plt.title('Dropout rate #2 vs Avg. Val MSE across folds')
plt.xlabel('Dropout rate')
plt.ylabel('Val MSE')
plt.savefig('bayes/dr2_limited.png')




###
plt.figure(figsize=(24, 16))
plt.rc('font', size=40)
plt.scatter(df['el'], df['val_error'], s=100)
plt.title('Entropy lambda vs Avg. Val MSE across folds')
plt.xlabel('log_10 of Entropy lambda')
plt.ylabel('Val MSE')
plt.savefig('bayes/el.png')

plt.figure(figsize=(24, 16))
plt.rc('font', size=40)
plt.scatter(df['el'], df['val_error'], s=100)
plt.ylim(0,1)
plt.title('Entropy lambda vs Avg. Val MSE across folds')
plt.xlabel('log_10 of Entropy lambda')
plt.ylabel('Val MSE')
plt.savefig('bayes/el_limited.png')

###
plt.figure(figsize=(24, 16))
plt.rc('font', size=40)
plt.scatter(df['cl'], df['val_error'], s=100)
plt.title('column lambda vs Avg. Val MSE across folds')
plt.xlabel('log_10 of column lambda')
plt.ylabel('Val MSE')
plt.savefig('bayes/cl.png')

plt.figure(figsize=(24, 16))
plt.rc('font', size=40)
plt.scatter(df['cl'], df['val_error'], s=100)
plt.ylim(0,1)
plt.title('column lambda vs Avg. Val MSE across folds')
plt.xlabel('log_10 of column lambda')
plt.ylabel('Val MSE')
plt.savefig('bayes/cl_limited.png')
