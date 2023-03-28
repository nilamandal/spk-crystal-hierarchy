import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df=pd.read_csv('noz_sweep2.csv')
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
plt.scatter(np.log10(df['lr']), df['val_error'], s=100)
plt.title('LR vs Avg. Val MSE across folds')
plt.xlabel('log_10 of learning rate')
plt.ylabel('Val MSE')
plt.savefig('noz2/lr.png')


plt.figure(figsize=(24, 16))
plt.rc('font', size=40)
plt.scatter(np.log10(df['lr']), df['val_error'], s=100)
plt.ylim(0,1)
plt.title('LR vs Avg. Val MSE across folds')
plt.xlabel('log_10 of learning rate')
plt.ylabel('Val MSE')
plt.savefig('noz2/lr_limited.png')

###
plt.figure(figsize=(24, 16))
plt.rc('font', size=40)
plt.scatter(df['bs'], df['val_error'], s=100)
plt.title('Batch size vs Avg. Val MSE across folds')
plt.xlabel('Batch size')
plt.ylabel('Val MSE')
plt.savefig('noz2/bs.png')


plt.figure(figsize=(24, 16))
plt.rc('font', size=40)
plt.scatter(df['bs'], df['val_error'], s=100)
plt.ylim(0,1)
plt.title('Batch size vs Avg. Val MSE across folds')
plt.xlabel('Batch size')
plt.ylabel('Val MSE')
plt.savefig('noz2/bs_limited.png')

###
plt.figure(figsize=(24, 16))
plt.rc('font', size=40)
plt.scatter(df['dr1'], df['val_error'], s=100)
plt.title('Dropout rate #1 vs Avg. Val MSE across folds')
plt.xlabel('Dropout rate')
plt.ylabel('Val MSE')
plt.savefig('noz2/dr1.png')

plt.figure(figsize=(24, 16))
plt.rc('font', size=40)
plt.scatter(df['dr1'], df['val_error'], s=100)
plt.ylim(0,1)
plt.title('Dropout rate #1 vs Avg. Val MSE across folds')
plt.xlabel('Dropout rate')
plt.ylabel('Val MSE')
plt.savefig('noz2/dr1_limited.png')

###
plt.figure(figsize=(24, 16))
plt.rc('font', size=40)
plt.scatter(df['dr2'], df['val_error'], s=100)
plt.title('Dropout rate #2 vs Avg. Val MSE across folds')
plt.xlabel('Dropout rate')
plt.ylabel('Val MSE')
plt.savefig('noz2/dr2.png')

plt.figure(figsize=(24, 16))
plt.rc('font', size=40)
plt.scatter(df['dr2'], df['val_error'], s=100)
plt.ylim(0,1)
plt.title('Dropout rate #2 vs Avg. Val MSE across folds')
plt.xlabel('Dropout rate')
plt.ylabel('Val MSE')
plt.savefig('noz2/dr2_limited.png')


# ###
# plt.figure(figsize=(24, 16))
# plt.rc('font', size=40)
# plt.scatter(df['dr3'], df['val_error'], s=100)
# plt.title('Dropout rate #3 vs Avg. Val MSE across folds')
# plt.xlabel('Dropout rate')
# plt.ylabel('Val MSE')
# plt.savefig('noz2/dr3.png')
#
# plt.figure(figsize=(24, 16))
# plt.rc('font', size=40)
# plt.scatter(df['dr3'], df['val_error'], s=100)
# plt.ylim(0,1)
# plt.title('Dropout rate #3 vs Avg. Val MSE across folds')
# plt.xlabel('Dropout rate')
# plt.ylabel('Val MSE')
# plt.savefig('noz2/dr3_limited.png')

###
plt.figure(figsize=(24, 16))
plt.rc('font', size=40)
plt.scatter(np.log10(df['el']), df['val_error'], s=100)
plt.title('Entropy lambda vs Avg. Val MSE across folds')
plt.xlabel('log_10 of Entropy lambda')
plt.ylabel('Val MSE')
plt.savefig('noz2/el.png')

plt.figure(figsize=(24, 16))
plt.rc('font', size=40)
plt.scatter(np.log10(df['el']), df['val_error'], s=100)
plt.ylim(0,1)
plt.title('Entropy lambda vs Avg. Val MSE across folds')
plt.xlabel('log_10 of Entropy lambda')
plt.ylabel('Val MSE')
plt.savefig('noz2/el_limited.png')

###
plt.figure(figsize=(24, 16))
plt.rc('font', size=40)
plt.scatter(np.log10(df['cl']), df['val_error'], s=100)
plt.title('column lambda vs Avg. Val MSE across folds')
plt.xlabel('log_10 of column lambda')
plt.ylabel('Val MSE')
plt.savefig('noz2/cl.png')

plt.figure(figsize=(24, 16))
plt.rc('font', size=40)
plt.scatter(np.log10(df['cl']), df['val_error'], s=100)
plt.ylim(0,1)
plt.title('column lambda vs Avg. Val MSE across folds')
plt.xlabel('log_10 of column lambda')
plt.ylabel('Val MSE')
plt.savefig('noz2/cl_limited.png')

##

#
#
# ###
# plt.figure(figsize=(24, 16))
# plt.rc('font', size=40)
# plt.scatter(np.log10(df['l2_1']), df['val_error'], s=100)
# plt.title('L2 lambda #1 vs Avg. Val MSE across folds')
# plt.xlabel('log_10 of L2 lambda')
# plt.ylabel('Val MSE')
# plt.savefig('noz2/l2_1.png')
#
# plt.figure(figsize=(24, 16))
# plt.rc('font', size=40)
# plt.scatter(np.log10(df['l2_1']), df['val_error'], s=100)
# plt.ylim(0,1)
# plt.title('L2 lambda #1 vs Avg. Val MSE across folds')
# plt.xlabel('log_10 of L2 lambda')
# plt.ylabel('Val MSE')
# plt.savefig('noz2/l2_1_limited.png')
#
# ###
# plt.figure(figsize=(24, 16))
# plt.rc('font', size=40)
# plt.scatter(np.log10(df['l2_2']), df['val_error'], s=100)
# plt.title('L2 lambda #2 vs Avg. Val MSE across folds')
# plt.xlabel('log_10 of L2 lambda')
# plt.ylabel('Val MSE')
# plt.savefig('noz2/l2_2.png')
#
# plt.figure(figsize=(24, 16))
# plt.rc('font', size=40)
# plt.scatter(np.log10(df['l2_2']), df['val_error'], s=100)
# plt.ylim(0,1)
# plt.title('L2 lambda #2 vs Avg. Val MSE across folds')
# plt.xlabel('log_10 of L2 lambda')
# plt.ylabel('Val MSE')
# plt.savefig('noz2/l2_2_limited.png')
#
#
# ###
# plt.figure(figsize=(24, 16))
# plt.rc('font', size=40)
# plt.scatter(np.log10(df['l2_3']), df['val_error'], s=100)
# plt.title('L2 lambda #3 vs Avg. Val MSE across folds')
# plt.xlabel('log_10 of L2 lambda')
# plt.ylabel('Val MSE')
# plt.savefig('noz2/l2_3.png')
#
# plt.figure(figsize=(24, 16))
# plt.rc('font', size=40)
# plt.scatter(np.log10(df['l2_3']), df['val_error'], s=100)
# plt.ylim(0,1)
# plt.title('L2 lambda #3 vs Avg. Val MSE across folds')
# plt.xlabel('log_10 of L2 lambda')
# plt.ylabel('Val MSE')
# plt.savefig('noz2/l2_3_limited.png')


# plt.figure(figsize=(24, 16))
# plt.rc('font', size=40)
# plt.scatter(df['decay_rate'], df['val_error'], s=100)
# plt.title('LR Decay rate vs Avg. Val MSE across folds')
# plt.xlabel('decay rate')
# plt.ylabel('Val MSE')
# plt.savefig('noz2/decayrate.png')
#
# plt.figure(figsize=(24, 16))
# plt.rc('font', size=40)
# plt.scatter(df['decay_rate'], df['val_error'], s=100)
# plt.ylim(0,1)
# plt.title('LR Decay rate vs Avg. Val MSE across folds')
# plt.xlabel('decay rate')
# plt.ylabel('Val MSE')
# plt.savefig('noz2/decayrate_limited.png')
#
# plt.figure(figsize=(24, 16))
# plt.rc('font', size=40)
# plt.scatter(df['decay_steps'], df['val_error'], s=100)
# plt.title('LR Decay steps vs Avg. Val MSE across folds')
# plt.xlabel('decay steps')
# plt.ylabel('Val MSE')
# plt.savefig('noz2/decaysteps.png')
#
# plt.figure(figsize=(24, 16))
# plt.rc('font', size=40)
# plt.scatter(df['decay_steps'], df['val_error'], s=100)
# plt.ylim(0,1)
# plt.title('LR Decay steps vs Avg. Val MSE across folds')
# plt.xlabel('decay steps')
# plt.ylabel('Val MSE')
# plt.savefig('noz2/decaysteps_limited.png')
