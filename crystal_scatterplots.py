import matplotlib.pyplot as plt
import pandas as pd
from pymatgen.core.structure import Structure

#color_dict={15: '#d479cf',20: '#b179d4',38: '#4400ff', 4: '#0057d1', 12:'#4aedca',25:'#d02669', 30:'#a83238', 33:'#2bcc59', 48:'#264dd0', 51:'#de8a2a', 56:'#ccc72b', 70:'#71cc2b', 80: '#40bde3', 83:'#5bad60'}
df=pd.read_csv('../Main_fol_Zintl/Ternary/a4b8e1/227/Na8Sb4Sn1/pool.dat')
#crystal= Structure.from_file('../Main_fol_Zintl/Binary/a11b3/60/As11K3/CONTCAR')
df = df[df['P1'].notna()]
df.to_csv('temp.csv')
#print(df)
#print(crystal)
plt.scatter(df['P1'],df['P2'])
plt.xlabel('P1')
plt.ylabel('P2')
plt.title('Pooling Values for Na8Sb4Sn1')

plt.annotate('Sb1, Sb4, Sb5, Sb6', (0.978740335,0.021259682))
plt.annotate('Sb2, Sb7, Sb8', (0.979532003, 0.020468079))
plt.annotate('Sb3', (0.980210364, 0.019789664))
plt.annotate('Na2', (0.98704195,0.012958024))
plt.annotate('Na6, Na9, Na12, Na15', (0.990084767, 0.009915276))

plt.annotate('Na7, Na8, Na16', (0.990350187, 0.009462364))

plt.annotate("Na11, Na13", (0.990770459,0.009229505))
plt.annotate('Na10, Na14', (0.991003752,0.00899628))
plt.annotate('Na5', (0.991187274,0.008812704))
plt.annotate('Na4', (0.992880702,0.007119241))
plt.annotate('Sn1, Sn2', (0.993290305,0.006709622))
plt.annotate('Na1, Na3', (0.994630098, 0.005369914))

# plt.annotate('As1, As2', (0.982061148,0.017938863))
# plt.annotate('Zn1, Zn4', (0.989925265, 0.010074695))
# plt.annotate('Zn2, Zn3', (0.991493344,0.008506636))
# plt.annotate('Rb1', (0.993142486,0.006857537))
# plt.annotate('As3', (0.998129308, 0.001870678))
#print(df_As)
#as1= list(df_As['P1'])
#as2= list(df_As['P2'])
#print(as1)
#print(as2)
#print(df_As[df_As['P1'].duplicated() == True])

# plt.annotate('K1, K5', (0.7797118425369263, 0.2202881872653961))
# plt.annotate('K3, K7', (0.7835243940353394, 0.216475561261177))
# plt.annotate('K2, K6', (0.7900175452232361, 0.2099824249744415))
# plt.annotate('K4, K8', (0.7938756942749023, 0.2061242759227752))
# plt.annotate('K9, K10, K11, K12', (0.8926683664321899,0.1073316484689712))
# plt.annotate('As9, As11, As13, As15', (0.839677751, 0.160322294))
# plt.annotate('As10, As12, As14, As16', (0.845053196, 0.154946804))
# plt.annotate('As26, As28, As30, As32', (0.868744195, 0.131255761))
# plt.annotate('As34, As38', (0.871451855, 0.128548071))
# #plt.annotate('As25, As29', (0.872982919, 0.127017081))
# plt.annotate('As25, As27, As29, As31, As36, As40', (0.873515129,0.126484856))
# plt.annotate('As1, As3, As5, As7', (0.875327945,0.12467204))
# plt.annotate('As2, As4, As6, As8', (0.878196001,0.121803999))
# plt.annotate('As33, As37', (0.880178511, 0.119821548))
# plt.annotate('As35, As39', (0.882780731, 0.117219217))
# plt.annotate('As41, As42, As43, As44', (0.897506118, 0.102493942))
# plt.annotate('As18, As22', (0.907441676, 0.092558339))
# plt.annotate('As20, As24', (0.909023821,0.090976246))
# plt.annotate('As17, As21', (0.910940051, 0.089059986))
# plt.annotate('As19, As23', (0.912268996, 0.087731004))
#for i in range(1, len(as1)):
#    name=str(i+1)
#    plt.annotate(name, (as1[i],as2[i]))
plt.show()
# for i in range(len(f)):
#     line= f[i]
#     if 'a2c1c2' in line:
#         plt.figure()
#
#         error= f[i+1].split('[[')[1].split(']]')[0]
#         coords=[]
#         for j in range(3, 13):
#             temp=f[i+j].strip(' [').split()
#             x=float(temp[0])
#             y=float(temp[1][:-3])
#             coords.append([x,y])
#             #print(x, y)
#
#         assignment=np.array(coords)
#         atomic_numbers=[crystal[i].specie.number for i in range(len(crystal))]
#         colors=[color_dict[i] for i in atomic_numbers]
#         plt.scatter(assignment[:,0],assignment[:,1],c=colors)
#         for i, txt in enumerate(crystal.species):
#             name=str(txt)+' '+str(i)
#             #print(name)
#             plt.annotate(name, (assignment[i,0], assignment[i,1]))
#         plt.title(line.strip()+', sq. error='+error)
#         plt.xlabel('assignment column 1')
#         plt.ylabel('assignment column 2')
#         plt.savefig('110_assingments/'+line.strip()+'_good_assignment.png')
