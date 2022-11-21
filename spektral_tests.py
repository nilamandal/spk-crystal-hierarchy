import pandas as pd
from pymatgen.core.structure import Structure

def get_atoms(c):
    c=str(c)
    crystal= Structure.from_file('../crystalhierarchydata/formationcifs/'+c+'.cif')
    atomic_numbers=set([crystal[i].specie.number for i in range(len(crystal))])
    return atomic_numbers

df = pd.read_csv('complete_info.csv')
#df['atoms']=df['id'].apply(get_atoms)
cifs=df['id']
contents=df['atoms']
#df.to_csv('complete_info.csv')
elements={9: 5086, 90: 345, 57: 1393, 60: 893, 65: 592, 70: 581, 67: 684, 62: 808, 38: 1657, 56: 2332, 69: 522, 39: 1066, 68: 666, 19: 2468, 89: 76, 59: 757, 8: 26390, 71: 496, 37: 1417, 58: 705, 21: 701, 40: 938, 20: 1670, 11: 3185, 64: 330, 91: 77, 72: 584, 13: 1958, 17: 1849, 63: 274, 92: 686, 3: 10455, 12: 1428, 66: 670, 22: 2164, 94: 78, 82: 932, 14: 3238, 4: 398, 73: 757, 55: 1070, 5: 2448, 93: 112, 61: 120, 23: 3528, 34: 1668, 41: 1358, 74: 939, 24: 2185, 7: 2124, 49: 1242, 83: 1413, 31: 1156, 35: 827, 25: 4209, 6: 2146, 15: 6782, 75: 426, 52: 1493, 26: 4206, 30: 1354, 16: 3271, 81: 943, 79: 796, 50: 1810, 27: 2786, 28: 2630, 1: 3248, 29: 2754, 32: 1605, 48: 878, 51: 1823, 46: 917, 47: 1149, 42: 965, 45: 789, 33: 1337, 53: 987, 44: 701, 76: 315, 80: 698, 77: 593, 78: 728, 43: 169, 2: 2, 54: 46, 36: 8}
useful=list(elements.keys())
temp=[]
for key in useful:
    if elements[key]<7000:
        temp.append(key)
print(temp)        
# test_filter=[]
# test_cifs=[]
# val_filter=[]
# val_cifs=[]
# for i in range(len(contents)):
#     c=contents[i]
#     c=c[1:-1].split(',')
#     c=[int(x) for x in c]
#     contents[i]=c
#
# while len(test_cifs)<7000:
#     mol= useful.pop()
#     if elements[mol]<7000:
#         test_filter.append(mol)
#         for i in range(len(contents)):
#             if mol in contents[i]:
#                 test_cifs.append(cifs[i])
#
# print(test_filter)
# print(len(test_cifs))
#
# while len(val_cifs)<7000:
#     mol= useful.pop()
#     if elements[mol]<7000:
#         val_filter.append(mol)
#         for i in range(len(contents)):
#             if mol in contents[i]:
#                 if cifs[i] not in test_cifs:
#                     val_cifs.append(cifs[i])
#
# print(val_filter)
# print(len(val_cifs))


#         if a in elements.keys():

#
# print(elements)
