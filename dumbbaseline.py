import os
import numpy as np
import pandas as pd
from pymatgen.core.structure import Structure
import matplotlib.pyplot as plt

color_dict={15: '#d479cf',20: '#b179d4',38: '#4400ff', 4: '#0057d1', 12:'#4aedca',25:'#d02669', 30:'#a83238', 33:'#2bcc59', 48:'#264dd0', 51:'#de8a2a', 56:'#ccc72b', 70:'#71cc2b', 80: '#40bde3', 83:'#5bad60'}
f= open('./presentation_prep_assignments2.txt')
f= f.readlines()
for i in range(len(f)):
    line= f[i]
    if 'a2c1c2' in line:
        plt.figure()
        crystal= Structure.from_file('../cgcnn-pretrained-models/data/10atom_relaxed_cifs/'+line.strip()+'.cif')
        error= f[i+1].split('[[')[1].split(']]')[0]
        coords=[]
        for j in range(3, 13):
            temp=f[i+j].strip(' [').split()
            x=float(temp[0])
            y=float(temp[1][:-3])
            coords.append([x,y])
            #print(x, y)

        assignment=np.array(coords)
        atomic_numbers=[crystal[i].specie.number for i in range(len(crystal))]
        colors=[color_dict[i] for i in atomic_numbers]
        plt.scatter(assignment[:,0],assignment[:,1],c=colors)
        for i, txt in enumerate(crystal.species):
            name=str(txt)+' '+str(i)
            #print(name)
            plt.annotate(name, (assignment[i,0], assignment[i,1]))
        plt.title(line.strip()+', sq. error='+error)
        plt.xlabel('assignment column 1')
        plt.ylabel('assignment column 2')
        plt.savefig('110_assingments/'+line.strip()+'_good_assignment.png')


# #title='Bi2Mn2Cd1_a2c1c2_sg139_cscaled'
# #error=0.00876775
