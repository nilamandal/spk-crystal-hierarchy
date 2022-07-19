import matplotlib.pyplot as plt
import numpy as np

files=['lr1e-2/lr1e-2.txt','lr1e-3/lr1e-3.txt','lr1e-4/lr1e-4.txt','lr1e-5/lr1e-5.txt','lr1e-6/lr1e-6.txt']
path='./separatefeats/spk16/'
for file in files:
    #figtitle=file.split('.')[0]+'.png'
    figtitle=path+file.split('.')[0]+'.png'
    f= open(path+file)
    f= f.readlines()
    #lr=f[0].split(',')
    #lr=lr[7]
    plt.figure(figsize=(18, 16))
    plt.rc('font', size=40)
    #print(temp[7])
    #plt.ylim([0,10])
    train_mse=[]
    val_mse=[]
    temp_train=[]
    for line_num in range(len(f)):
        if 'val loss' in f[line_num]:
            loss= float(f[line_num+1].split(' ')[0])
            #print(loss)
            val_mse.append(loss)
        if 'train loss' in f[line_num]:
            if 'Loss:' in f[line_num+2]:
                #print()
                train_mse.append(float(f[line_num+2].split(' ')[1].strip()))
            #print(f[line_num])

            #temp_train.append(float(f[line_num+3].split(',')[0].split('(')[1]))
            #print(temp_train)
        #if len(temp_train)==703:
        #    train_mse.append(np.mean(temp_train))
        #    temp_train=[]
    print(train_mse)
    print(len(train_mse))
    print(len(val_mse))
        # if 'mse' in line:
        #     mse=float(line.split(':')[-1].strip()[:-1])
        #     #print(mse)
        #     if "Train" in line:
        #         train_mse.append(mse)
        #     elif "Validation" in line:
        #         val_mse.append(mse)
    epochs=list(range(len(train_mse)))
    plt.plot(epochs, train_mse, label='training loss')
    plt.plot(epochs, val_mse, label='val loss')
    plt.title('learning curves')
    plt.xlabel('epochs')
    plt.legend()
    plt.ylabel('categorical cross entropy loss')
    plt.savefig(figtitle)
