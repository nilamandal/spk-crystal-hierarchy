import pubchempy as pcp
import pandas as pd

def pcp_query_by_smile(formula):
    try:
        d= pcp.get_compounds(formula, namespace='smiles', record_type='2d')
        #temp= d[0].to_dict(properties=['atoms', 'bonds'])
        print(d)
        return d
    except:
        return 'retry'
    return 'retry'

def get_bonds_by_pid(pid):
    #print(pid)
    try:
        d= pcp.Compound.from_cid(pid)
        g= d.to_dict(properties=['atoms', 'bonds'])
        print(pid)
        return g
    except:
        print(pid, 'retry')
        return 'retry'

if __name__ == "__main__":
    df= pd.read_csv('tox21_updated.csv')
    df= df[df['pid']>0]
    #mol_list= df['smiles'].tolist()

    df['structure']= df['pid'].apply(get_bonds_by_pid)
    df.to_csv('tox21_updated_2.csv')
