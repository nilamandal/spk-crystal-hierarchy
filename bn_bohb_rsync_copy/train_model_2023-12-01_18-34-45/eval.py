import os
import json
import pandas as pd
subs= os.walk('.')
print(subs)
all_dicts=[]
for sub in subs:
    try:
        params_dict= json.load(open(sub[0]+'/params.json'))
        result_dict= json.load(open(sub[0]+'/result.json'))
        params_dict['score']= result_dict['score']
        params_dict['path']= sub[0]
        all_dicts.append(params_dict)
        print(params_dict)
    except:
        print(sub)
df= pd.DataFrame(all_dicts)
df.to_csv('evaluated_results.csv')
