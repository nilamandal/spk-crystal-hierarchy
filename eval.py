#This file gathers all results from a completed raytune trial and saves them to a single csv file.
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
main_path='./'

subs= os.walk(main_path)
#print(subs)
all_dicts=[]
for sub in subs:
    #print(sub)
    if 'params.json' in sub[2]:
        params_dict= json.load(open(sub[0]+'/params.json'))
        try:
            result_dict= json.load(open(sub[0]+'/result.json'))
            params_dict['score']= result_dict['score']
            params_dict['timestamp']= result_dict["timestamp"]
            params_dict['path']= sub[0]
            all_dicts.append(params_dict)

        except:
            print('---')
            results=[]
            params_dict['score']= np.nan
            params_dict['path']= sub[0]
            with open(sub[0]+'/result.json') as file:
                for line in file:
                    results.append(json.loads(line))
            print('results file is readable')
            if len(results)==0:
                params_dict['timestamp']= 'no result'

            else:
                print('results>0')
                params_dict['timestamp']= 'multiple trials'
                for result_dict in results:
                    print(result_dict)
                    print(type(result_dict))
                    if np.isnan(params_dict['score']):
                        params_dict['score']= result_dict['score']
                    elif params_dict['score']>result_dict['score']:
                        params_dict['score']= result_dict['score']

            all_dicts.append(params_dict)

df= pd.DataFrame(all_dicts)
df.to_csv('evaluated_results_timestamps2.csv')
