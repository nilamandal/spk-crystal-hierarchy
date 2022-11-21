import os
import numpy as np
import pandas as pd
from pymatgen.core.structure import Structure

df = pd.read_csv('../crystalhierarchydata/sc10_scaled/id_prop_500.csv', names=['id','target'], header=0)
#print(df)
df= df[df['target']==2]
print(df)
allgraphs=[]
cifs=list(df['id'])

datadir='../crystalhierarchydata/sc10_scaled'
radius_angstroms=8
num_nbrs=12


for c in cifs:
    c=str(c)

    try:
        crystal= Structure.from_file(os.path.join(datadir,c+'.cif'))
    except:
        crystal= Structure.from_file(os.path.join(datadir,c))
    num_atoms=len(crystal)


    #all_atomic_numbers= all_atomic_numbers + atomic_numbers

    #atom_fea = np.vstack([ari.get_atom_fea(crystal[i].specie.number) for i in range(len(crystal))]) #the features of each element in the atom, in no particular order
    all_nbrs = crystal.get_all_neighbors(radius_angstroms, include_index=True)
    all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
    nbr_fea_idx = []
    for nbr in all_nbrs:
        if len(nbr) < num_nbrs:
            warnings.warn('{} not find enough neighbors to build graph. '
                          'If it happens frequently, consider increase '
                          'radius.'.format(cif_id))
            nbr_fea_idx.append(list(map(lambda x: x[2], nbr)) +
                               [0] * (num_nbrs - len(nbr)))

        else:
            nbr_fea_idx.append(list(map(lambda x: x[2],
                                        nbr[:num_nbrs])))

    adj = np.zeros((num_atoms, num_atoms))
    #edges= np.zeros((num_atoms, num_atoms, 41))

    for i in range(len(nbr_fea_idx)):
        for j in range(len(nbr_fea_idx[i])):
            k=nbr_fea_idx[i][j]
            adj[i,k]+=1

    allgraphs.append(adj)

allgraphs=np.array(allgraphs)
temp=np.unique(allgraphs, axis=0)
print(temp)
print(len(temp))
#print(temp[0]-temp[1])
#print()
            #edges[i,k]= nbr_fea[i][j]
