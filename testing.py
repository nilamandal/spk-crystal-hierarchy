import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance
#
# assignments=[[1.68770953e-01, 6.29828873e-01, 2.01400173e-01],
#  [1.68770953e-01, 6.29828873e-01, 2.01400173e-01],
#  [2.70070764e-02, 9.33198575e-01, 3.97943488e-02],
#  [2.70070764e-02, 9.33198575e-01, 3.97943488e-02],
#  [5.57164402e-02, 8.76281042e-01, 6.80025180e-02],
#  [5.57164402e-02, 8.76281042e-01, 6.80025180e-02]]
elements=['K','K','Sn','Sn',"Sb",'Sb']
#print(assignments)
x=np.load('./feats28.npz')
ours=x['x'][2][:6]

x4=np.load('./temptemp.npz')
ours4=x4['x'][2][:6]


cgcnn=np.load('./cgcnnfeats.npz')
cgcnn=cgcnn['x']
#print(ours.shape)
#print(cgcnn.shape)
for i in range(len(ours)):
    print(elements[i]+':')
    print(distance.cosine(ours[i],ours4[i]))
    print(distance.euclidean(ours[i],ours4[i]))
# for i in range(len(assignments)):
#     row=assignments[i]
#     #print(elements[i])
#     labels=['pool 1', 'pool 2', 'pool 3']
#     fig1, ax1 = plt.subplots()
#     ax1.pie(row, labels=labels, startangle=90)
#     ax1.set(aspect="equal", title=elements[i])
#     ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
#
#     plt.savefig('./'+elements[i]+str(i)+'.png')
