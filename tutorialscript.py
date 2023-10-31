import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

allpaths=['./train_model_2cdfc07a_5_batch_size=8,column_lambda=3.2909,dr1=0.1695,embedding_size=16,entropy_lambda=2990.0183,fc_num=1,fc_num2=1,_2023-09-14_17-43-56',
'./train_model_13dcd8fc_4_batch_size=16,column_lambda=27130.0631,dr1=0.4824,embedding_size=32,entropy_lambda=28.0892,fc_num=3,fc_num2_2023-09-14_17-43-52',
'./train_model_3a2f3847_12_batch_size=32,column_lambda=0.2294,dr1=0.9481,embedding_size=64,entropy_lambda=8528936.2598,fc_num=3,fc_nu_2023-09-14_17-44-34',
'./train_model_46ddffaa_16_batch_size=8,column_lambda=998.9974,dr1=0.8135,embedding_size=128,entropy_lambda=2615.9967,fc_num=2,fc_num_2023-09-15_22-08-28',
'./train_model_64275344_10_batch_size=16,column_lambda=1893426.7012,dr1=0.1045,embedding_size=16,entropy_lambda=0.6772,fc_num=3,fc_nu_2023-09-14_17-44-22',
'./train_model_be25cda0_6_batch_size=16,column_lambda=1397.5178,dr1=0.9609,embedding_size=4,entropy_lambda=0.9638,fc_num=2,fc_num2=3,_2023-09-14_17-44-01',
'./train_model_dba4e923_22_batch_size=32,column_lambda=15072.7017,dr1=0.6097,embedding_size=8,entropy_lambda=2403.1445,fc_num=3,fc_nu_2023-09-16_09-16-40',
'./train_model_ea5a966d_3_batch_size=32,column_lambda=24987189.4457,dr1=0.3052,embedding_size=64,entropy_lambda=2126.0027,fc_num=2,fc_2023-09-14_17-43-47',
'./train_model_5cd01513_17_batch_size=64,column_lambda=2068.3238,dr1=0.4154,embedding_size=64,entropy_lambda=1912.1705,fc_num=3,fc_nu_2023-09-16_05-04-18',
'./train_model_5f162727_7_batch_size=64,column_lambda=1339.7064,dr1=0.5028,embedding_size=4,entropy_lambda=5943574.8462,fc_num=1,fc_n_2023-09-14_17-44-06',
'./train_model_4a598c34_18_batch_size=16,column_lambda=234.9775,dr1=0.8384,embedding_size=32,entropy_lambda=4442.4202,fc_num=3,fc_num_2023-09-16_05-49-20',
'./train_model_3b9f5817_8_batch_size=16,column_lambda=0.1617,dr1=0.5883,embedding_size=16,entropy_lambda=48069.7772,fc_num=1,fc_num2=_2023-09-14_17-44-11',
'./train_model_e40b47ce_1_batch_size=4,column_lambda=18178756.6491,dr1=0.9875,embedding_size=16,entropy_lambda=4.0136,fc_num=2,fc_num_2023-09-14_17-43-38',
'./train_model_2c4f6c4b_11_batch_size=4,column_lambda=272.0161,dr1=0.5048,embedding_size=32,entropy_lambda=94.3016,fc_num=1,fc_num2=2_2023-09-14_17-44-29',
'./train_model_08e6d145_13_batch_size=8,column_lambda=199152.8561,dr1=0.8762,embedding_size=128,entropy_lambda=5839530.5303,fc_num=1,_2023-09-14_17-44-39',
'./train_model_109d2513_23_batch_size=64,column_lambda=3019.4853,dr1=0.3155,embedding_size=64,entropy_lambda=960.0818,fc_num=3,fc_num_2023-09-16_15-02-43']

for path in allpaths:

    df= pd.read_csv('./row_entropy_and_column_product/'+path+'/callback_results.csv')
    df['train_column_penalty']= df['train_column_penalty']+(-1*np.min(df['train_column_penalty']))
    df['val_column_penalty']= df['val_column_penalty']+(-1*np.min(df['val_column_penalty']))


    plt.figure()
    plt.plot(df['epoch'], np.log(df['train_mse']), c='#4597eb', label='train_mse')
    plt.plot(df['epoch'], np.log(df['val_mse']), c='#ffa50b', label='val_mse')
    plt.plot(df['epoch'], np.log(df['train_row_penalty']), c='#ff0b0b', label='train row penalty')
    plt.plot(df['epoch'], np.log(df['val_row_penalty']), c='#a327fe', label='val row penalty')
    plt.plot(df['epoch'], np.log(df['train_column_penalty']), c='#17960a', label='train col penalty')
    plt.plot(df['epoch'], np.log(df['val_column_penalty']), c='#dee41f', label='val col penalty')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('various loss values on log scale')
    #plt.show()
    plt.savefig('./row_entropy_and_column_product/'+path+'/all_loss_components_log.png')
