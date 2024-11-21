import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
main_path='./'

subs= os.walk(main_path)
print(subs)
all_dicts=[]
for sub in subs:
    print(sub)
    if 'params.json' in sub[2]:
        params_dict= json.load(open(sub[0]+'/params.json'))
        all_dicts.append(params_dict)
    #     print(params_dict)
    #     print('---')
#     for i in range(10):
#         #i=0
#
#         dfpath= str(i)+'_callback.csv'
#
#         if dfpath in sub[2]:
#             df_callback= pd.read_csv(sub[0]+'/'+str(i)+'_callback.csv')
#             #df_callback['valmse']= df_callback['valmse'].str.replace('[', '').str.replace(']', '').astype(float)
#
#             plt.figure()
#             plt.plot(list(range(len(df_callback))), df_callback['trainloss'],label='train')
#             plt.plot(list(range(len(df_callback))), df_callback['valmse'], label='val')
#             plt.xlabel('epoch')
#             plt.ylabel('mse')
#             plt.legend()
#             plt.savefig(sub[0]+'/'+str(i)+'_train_val_curve.png')
#             #df_callback.to_csv(sub[0]+'/'+str(i)+'_callback_corrected.csv')
#
# #     except:
# #         print('csv error')
# #     try:
# #         for i in range(10):
# #             df_callback= pd.read_csv(sub[0]+'/'+str(i)+'_callback.csv')
# #             plt.figure()
# #             plt.plot(list(range(len(df_callback))), df_callback['trainloss'],label='train')
# #             plt.plot(list(range(len(df_callback))), df_callback['valmse'], label='val')
# #             plt.xlabel('epoch')
# #             plt.ylabel('mse')
# #             plt.legend()
# #             plt.savefig(sub[0]+'/'+str(i)+'_train_val_curve.png')
# #         #print('Ok')
# #     except:
# #         print('plot error')
df= pd.DataFrame(all_dicts)
df.to_csv('evaluated_results_timestamps.csv')
