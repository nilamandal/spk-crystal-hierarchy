import pandas as pd
import numpy as np

electronegativity_lookup= {'Cs':0.79, 'K':0.82, 'Rb':0.82, 'Ba':0.89, 'Na':0.93, 'Sr':0.95, 'Li':0.98,
    'Ca':1, 'Yb':1.1, 'Mg':1.31, 'Mn':1.55, 'Be':1.57, 'Al':1.61, 'Tl':1.62, 'Zn':1.65, 'Cd':1.69, 'In':1.69,
    'Ga':1.81, 'Si':1.9, 'Sn':1.96, 'Hg':2, 'Ge':2.01, 'Bi':2.02, 'Sb':2.05, 'As':2.18, 'P':2.19, 'H':2.2,
    'Pb':2.33}

def cleanse(pstr):
    pstr= pstr.split(',')
    for i in range(len(pstr)):
        pstr[i]=pstr[i].strip()
    return pstr

def does_it_match(p1, p2):
    if len(p1)==0 or len(p2)==0:
        return 0
    elif len(p1)==1 and len(p2)==1:
        return 1
    else:
        for i in range(len(p1)):
            p1[i]= electronegativity_lookup[p1[i]]
        for i in range(len(p2)):
            p2[i]= electronegativity_lookup[p2[i]]

        if np.max(p1)<np.min(p2):
            return 1
        elif np.max(p2)<np.min(p1):
            return 1
        else:
            return 0
    return -1

            #check if a. everything in p1 is less than everything in p2
            #b. check if everything in p2 is less than p1

electronegativity_lookup= {'Cs':0.79, 'K':0.82, 'Rb':0.82, 'Ba':0.89, 'Na':0.93, 'Sr':0.95, 'Li':0.98,
    'Ca':1, 'Yb':1.1, 'Mg':1.31, 'Mn':1.55, 'Be':1.57, 'Al':1.61, 'Tl':1.62, 'Zn':1.65, 'Cd':1.69, 'In':1.69,
    'Ga':1.81, 'Si':1.9, 'Sn':1.96, 'Hg':2, 'Ge':2.01, 'Bi':2.02, 'Sb':2.05, 'As':2.18, 'P':2.19, 'H':2.2,
    'Pb':2.33}

df= pd.read_csv('../Main_fol_Zintl/all_nonmetals_by_fam.csv')
p1= df.P1.tolist()
p2= df.P2.tolist()

for i in range(len(p1)):
    try:
        p1[i]=cleanse(p1[i])
    except:
        p1[i]=[]

for i in range(len(p2)):
    try:
        p2[i]=cleanse(p2[i])
    except:
        p2[i]=[]


matchlist=[]
for i in range(len(p2)):
    x=does_it_match(p1[i],p2[i])
    matchlist.append(x)

df['match']=matchlist
df.to_csv('matchlist.csv')
