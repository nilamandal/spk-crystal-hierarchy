import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df=pd.read_csv('val_sigmoid.csv')
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
plt.scatter(np.log10(df['lr']), df['val_mse_mean'], s=100)
plt.title('LR vs Avg. Val MSE across folds')
plt.xlabel('log_10 of learning rate')
plt.ylabel('Val MSE')
plt.savefig('sigmoid_param_sweep/lr.png')


plt.figure(figsize=(24, 16))
plt.rc('font', size=40)
plt.scatter(np.log10(df['lr']), df['val_mse_mean'], s=100)
plt.ylim(0,1)
plt.title('LR vs Avg. Val MSE across folds')
plt.xlabel('log_10 of learning rate')
plt.ylabel('Val MSE')
plt.savefig('sigmoid_param_sweep/lr_limited.png')

###
plt.figure(figsize=(24, 16))
plt.rc('font', size=40)
plt.scatter(df['bs'], df['val_mse_mean'], s=100)
plt.title('Batch size vs Avg. Val MSE across folds')
plt.xlabel('Batch size')
plt.ylabel('Val MSE')
plt.savefig('sigmoid_param_sweep/bs.png')


plt.figure(figsize=(24, 16))
plt.rc('font', size=40)
plt.scatter(df['bs'], df['val_mse_mean'], s=100)
plt.ylim(0,1)
plt.title('Batch size vs Avg. Val MSE across folds')
plt.xlabel('Batch size')
plt.ylabel('Val MSE')
plt.savefig('sigmoid_param_sweep/bs_limited.png')

###
plt.figure(figsize=(24, 16))
plt.rc('font', size=40)
plt.scatter(df['dr1'], df['val_mse_mean'], s=100)
plt.title('Dropout rate #1 vs Avg. Val MSE across folds')
plt.xlabel('Dropout rate')
plt.ylabel('Val MSE')
plt.savefig('sigmoid_param_sweep/dr1.png')

plt.figure(figsize=(24, 16))
plt.rc('font', size=40)
plt.scatter(df['dr1'], df['val_mse_mean'], s=100)
plt.ylim(0,1)
plt.title('Dropout rate #1 vs Avg. Val MSE across folds')
plt.xlabel('Dropout rate')
plt.ylabel('Val MSE')
plt.savefig('sigmoid_param_sweep/dr1_limited.png')

###
plt.figure(figsize=(24, 16))
plt.rc('font', size=40)
plt.scatter(df['dr2'], df['val_mse_mean'], s=100)
plt.title('Dropout rate #2 vs Avg. Val MSE across folds')
plt.xlabel('Dropout rate')
plt.ylabel('Val MSE')
plt.savefig('sigmoid_param_sweep/dr2.png')

plt.figure(figsize=(24, 16))
plt.rc('font', size=40)
plt.scatter(df['dr2'], df['val_mse_mean'], s=100)
plt.ylim(0,1)
plt.title('Dropout rate #2 vs Avg. Val MSE across folds')
plt.xlabel('Dropout rate')
plt.ylabel('Val MSE')
plt.savefig('sigmoid_param_sweep/dr2_limited.png')


# ###
# plt.figure(figsize=(24, 16))
# plt.rc('font', size=40)
# plt.scatter(df['dr3'], df['val_mse_mean'], s=100)
# plt.title('Dropout rate #3 vs Avg. Val MSE across folds')
# plt.xlabel('Dropout rate')
# plt.ylabel('Val MSE')
# plt.savefig('sigmoid_param_sweep/dr3.png')
#
# plt.figure(figsize=(24, 16))
# plt.rc('font', size=40)
# plt.scatter(df['dr3'], df['val_mse_mean'], s=100)
# plt.ylim(0,1)
# plt.title('Dropout rate #3 vs Avg. Val MSE across folds')
# plt.xlabel('Dropout rate')
# plt.ylabel('Val MSE')
# plt.savefig('sigmoid_param_sweep/dr3_limited.png')

###
plt.figure(figsize=(24, 16))
plt.rc('font', size=40)
plt.scatter(np.log10(df['el']), df['val_mse_mean'], s=100)
plt.title('Entropy lambda vs Avg. Val MSE across folds')
plt.xlabel('log_10 of Entropy lambda')
plt.ylabel('Val MSE')
plt.savefig('sigmoid_param_sweep/el.png')

plt.figure(figsize=(24, 16))
plt.rc('font', size=40)
plt.scatter(np.log10(df['el']), df['val_mse_mean'], s=100)
plt.ylim(0,1)
plt.title('Entropy lambda vs Avg. Val MSE across folds')
plt.xlabel('log_10 of Entropy lambda')
plt.ylabel('Val MSE')
plt.savefig('sigmoid_param_sweep/el_limited.png')

###
plt.figure(figsize=(24, 16))
plt.rc('font', size=40)
plt.scatter(np.log10(df['cl']), df['val_mse_mean'], s=100)
plt.title('column lambda vs Avg. Val MSE across folds')
plt.xlabel('log_10 of column lambda')
plt.ylabel('Val MSE')
plt.savefig('sigmoid_param_sweep/cl.png')

plt.figure(figsize=(24, 16))
plt.rc('font', size=40)
plt.scatter(np.log10(df['cl']), df['val_mse_mean'], s=100)
plt.ylim(0,1)
plt.title('column lambda vs Avg. Val MSE across folds')
plt.xlabel('log_10 of column lambda')
plt.ylabel('Val MSE')
plt.savefig('sigmoid_param_sweep/cl_limited.png')

##

#
#
# ###
# plt.figure(figsize=(24, 16))
# plt.rc('font', size=40)
# plt.scatter(np.log10(df['l2_1']), df['val_mse_mean'], s=100)
# plt.title('L2 lambda #1 vs Avg. Val MSE across folds')
# plt.xlabel('log_10 of L2 lambda')
# plt.ylabel('Val MSE')
# plt.savefig('sigmoid_param_sweep/l2_1.png')
#
# plt.figure(figsize=(24, 16))
# plt.rc('font', size=40)
# plt.scatter(np.log10(df['l2_1']), df['val_mse_mean'], s=100)
# plt.ylim(0,1)
# plt.title('L2 lambda #1 vs Avg. Val MSE across folds')
# plt.xlabel('log_10 of L2 lambda')
# plt.ylabel('Val MSE')
# plt.savefig('sigmoid_param_sweep/l2_1_limited.png')
#
# ###
# plt.figure(figsize=(24, 16))
# plt.rc('font', size=40)
# plt.scatter(np.log10(df['l2_2']), df['val_mse_mean'], s=100)
# plt.title('L2 lambda #2 vs Avg. Val MSE across folds')
# plt.xlabel('log_10 of L2 lambda')
# plt.ylabel('Val MSE')
# plt.savefig('sigmoid_param_sweep/l2_2.png')
#
# plt.figure(figsize=(24, 16))
# plt.rc('font', size=40)
# plt.scatter(np.log10(df['l2_2']), df['val_mse_mean'], s=100)
# plt.ylim(0,1)
# plt.title('L2 lambda #2 vs Avg. Val MSE across folds')
# plt.xlabel('log_10 of L2 lambda')
# plt.ylabel('Val MSE')
# plt.savefig('sigmoid_param_sweep/l2_2_limited.png')
#
#
# ###
# plt.figure(figsize=(24, 16))
# plt.rc('font', size=40)
# plt.scatter(np.log10(df['l2_3']), df['val_mse_mean'], s=100)
# plt.title('L2 lambda #3 vs Avg. Val MSE across folds')
# plt.xlabel('log_10 of L2 lambda')
# plt.ylabel('Val MSE')
# plt.savefig('sigmoid_param_sweep/l2_3.png')
#
# plt.figure(figsize=(24, 16))
# plt.rc('font', size=40)
# plt.scatter(np.log10(df['l2_3']), df['val_mse_mean'], s=100)
# plt.ylim(0,1)
# plt.title('L2 lambda #3 vs Avg. Val MSE across folds')
# plt.xlabel('log_10 of L2 lambda')
# plt.ylabel('Val MSE')
# plt.savefig('sigmoid_param_sweep/l2_3_limited.png')


plt.figure(figsize=(24, 16))
plt.rc('font', size=40)
plt.scatter(df['decay_rate'], df['val_mse_mean'], s=100)
plt.title('LR Decay rate vs Avg. Val MSE across folds')
plt.xlabel('decay rate')
plt.ylabel('Val MSE')
plt.savefig('sigmoid_param_sweep/decayrate.png')

plt.figure(figsize=(24, 16))
plt.rc('font', size=40)
plt.scatter(df['decay_rate'], df['val_mse_mean'], s=100)
plt.ylim(0,1)
plt.title('LR Decay rate vs Avg. Val MSE across folds')
plt.xlabel('decay rate')
plt.ylabel('Val MSE')
plt.savefig('sigmoid_param_sweep/decayrate_limited.png')

plt.figure(figsize=(24, 16))
plt.rc('font', size=40)
plt.scatter(df['decay_steps'], df['val_mse_mean'], s=100)
plt.title('LR Decay steps vs Avg. Val MSE across folds')
plt.xlabel('decay steps')
plt.ylabel('Val MSE')
plt.savefig('sigmoid_param_sweep/decaysteps.png')

plt.figure(figsize=(24, 16))
plt.rc('font', size=40)
plt.scatter(df['decay_steps'], df['val_mse_mean'], s=100)
plt.ylim(0,1)
plt.title('LR Decay steps vs Avg. Val MSE across folds')
plt.xlabel('decay steps')
plt.ylabel('Val MSE')
plt.savefig('sigmoid_param_sweep/decaysteps_limited.png')
