import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df= pd.read_csv('./jsq_asof_10_16/train_model_ba08ade7_17_batch_size=32,column_lambda=41903766.3588,dr1=0.2514,embedding_size=4,entropy_lambda=108506.3980,fc_num=1,_2023-10-13_08-44-52/callback_results.csv')

df['whole_train']= (df['train_column_penalty']*41903766.3588113)+df['train_mse']+(df['train_row_penalty']*108506.39801000628)
df['whole_val']= (df['val_column_penalty']*41903766.3588113)+df['val_mse']+(df['val_row_penalty']*108506.39801000628)


plt.plot(df['whole_train'], label='train all loss')

plt.plot(df['whole_val'], label='val all loss')
plt.legend()
plt.xlabel('training epochs')
plt.ylabel('values of components of loss function on log scale')
plt.title('model f01f713f')
plt.show()
