import os
import numpy as np
import pandas as pd
from pymatgen.core.structure import Structure
import matplotlib.pyplot as plt

color_dict={15: '#d479cf',20: '#b179d4',38: '#4400ff', 4: '#0057d1', 12:'#4aedca',25:'#d02669', 30:'#a83238', 33:'#2bcc59', 48:'#264dd0', 51:'#de8a2a', 56:'#ccc72b', 70:'#71cc2b', 80: '#40bde3', 83:'#5bad60'}
electronegativity_dict={80: 2, 48: 1.69, 30: 1.65, 4: 1.57, 25: 1.55, 12: 1.31, 70: 1.1, 20: 1, 38: 0.95, 56: 0.89, 15: 2.19, 33: 2.18, 51: 2.05, 83: 2.02}

f= open('noz2_1234_22_assignments.txt')
f= f.readlines()
assignment_dict={'good':[], 'allinone':[], 'other':[]}
for i in range(len(f)):
    line= f[i]
    if 'a2c1c2' in line:
        plt.figure()
        crystal= Structure.from_file('../cgcnn-pretrained-models/data/10atom_relaxed_cifs/'+line.strip()+'.cif')
        #print(len(crystal))
        error= f[i+1].split('[')[1].split(']')[0]
        coords=[]

        for j in range(3, 13):
            temp=f[i+j].strip(' [').split()

            #print(temp)
            #print(temp[1])
            x=float(temp[0])
            try:
                y=float(temp[1].strip(',').strip(']'))
            except:
                print(temp[1])


            coords.append([x,y])
            #print(x, y)

        assignment=np.array(coords)
        sums=assignment.sum(axis=0)
        if sums[0]==0 or sums[1]==0 or sums[1]==len(crystal) or sums[0]==len(crystal):
            assignment_dict['allinone'].append(line.strip())
        else:
            assignment_dict['other'].append(line.strip())
        atomic_numbers=[crystal[i].specie.number for i in range(len(crystal))]
        colors=[color_dict[i] for i in atomic_numbers]
        electros= [electronegativity_dict[i] for i in atomic_numbers]
        plt.scatter(assignment[:,0],assignment[:,1],c=colors)
        for i, txt in enumerate(crystal.species):
            name=str(txt)+' '+str(i)
            #print(name)
            plt.annotate(name, (assignment[i,0], assignment[i,1]))
        plt.title(line.strip()+', sq. error='+error)
        plt.xlabel('assignment column 1')
        plt.ylabel('assignment column 2')
        plt.savefig('./noz2/22_assignments/'+line.strip()+'.png')
print(assignment_dict)
