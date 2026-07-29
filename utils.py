import pandas as pd
import tensorflow as tf
import numpy as np
import json
import os
from spektral.data import Graph, Dataset, DisjointLoader
from tensorflow.keras.callbacks import CallbackList, CSVLogger
from pymatgen.core.structure import Structure
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
import matplotlib.pyplot as plt
from spektral_essential_objects import AtomFeaDataset, GaussianDistance

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

def gen_plots(train_metric, val_metric, idx):
    plt.switch_backend('Agg')
    plt.figure()

    epochs=list(range(len(train_metric)))
    min_train= 'train min='+str(np.round(np.min(train_metric), decimals=3))+','
    min_val= 'val min='+str(np.round(np.min(val_metric), decimals=3))
    plt.plot(epochs, np.log(train_metric), label='training loss')
    plt.plot(epochs, np.log(val_metric), label='val loss')

    figtitle=idx+'_result.png'
    plt.xlabel('epochs')
    plt.legend()
    plt.ylabel('log of mean square error ')
    plt.savefig(figtitle)

def check_env_versions():
    import tensorflow as tf
    print(tf.__version__)
    import spektral
    print(spektral.__version__)
    import numpy
    print(numpy.__version__)
    import optuna
    print(optuna.__version__)

def get_available(filename):
    try:
        crystal= Structure.from_file('../Main_fol_Zintl/'+filename)
        ana= SpacegroupAnalyzer(crystal)
        sym_crystal= ana.get_symmetrized_structure()
        return len(sym_crystal.equivalent_indices)
    except:
        print(filename)

def scale_by_pred_vol(structure, site_bias, dls_vol_predictor):
    # first predict the volume using the average volume per element (from ICSD)
    site_counts = pd.Series(Counter(
        str(site.specie) for site in structure.sites)).fillna(0)
    curr_site_bias = site_bias[site_bias.index.isin(site_counts.index)]

    try:
        linear_pred = site_counts @ curr_site_bias
        structure.scale_lattice(linear_pred)
    except:
        pass
    # then apply Pymatgen's DLS predictor
    pred_volume = dls_vol_predictor.predict(structure)
    structure.scale_lattice(pred_volume)
    #
    return structure

def scale_dls_only(c):
    c=str(c)
    try:
        from pymatgen.core.structure import Structure
    except:
        crystal= Structure.from_file(os.path.join(data_path,c))
    structure= dls_vol_predictor.get_predicted_structure(crystal)
    newpath='./sc24_scaled/'+c.split('/')[-1][:-7]+'.cif'
    structure.to(filename=newpath)
    return newpath

def train_step(inputs, target, model, loss_fn, optimizer, task='r'):
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        loss = loss_fn(target, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    if task=='r':
        mse = tf.reduce_mean((target-predictions)**2)

        return loss, mse
    if task=='c':
        sca= tf.reduce_mean(categorical_accuracy(target, predictions))
        return loss, sca

def evaluate(loader, model, loss_fn, test=False, task='r'):
    step = 0
    output=[]
    while step < loader.steps_per_epoch:
        step += 1
        inputs, target = loader.__next__()
        pred = model(inputs, training=False)

        if task=='c':
            outs = (
                loss_fn(target, pred),
                len(target),  # Keep track of batch size
            )
        elif task=='r':
            mse = loss_fn(target, pred) #ASSUMES REGRESSION LOSS IS MSE
            rmse= np.sqrt(mse)
            mae= tf.reduce_mean(np.abs(target-pred))
            outs = (
                mse,
                rmse,
                mae,
                len(target),  # Keep track of batch size
            )
        output.append(outs)
        if step == loader.steps_per_epoch:
            output = np.array(output)
            return np.average(output[:, :-1], 0, weights=output[:, -1])

def train_single_model(model, load_tr, load_tr_eval, load_va, optim, trial, path_i):
    print('BEGUN INDIVIDUAL TRAINING')

    checkpoint_path=path_i+'goodmodel.keras'
    csv_log = CSVLogger(path_i+"callback_results.csv")
    all_callbacks= CallbackList([csv_log], add_history=True, model=model)

    loss_fn= MeanSquaredError()

    train_metric=[]
    val_metric_list=[]
    early_stop_counter= 0
    patience= 50
    epoch = step = 0

    best_val_loss = np.inf
    logs = {}
    all_callbacks.on_train_begin(logs=logs)

    for batch in load_tr:
        if step==0:
            all_callbacks.on_epoch_begin(epoch, logs=logs)
        step += 1

        all_callbacks.on_train_batch_begin(step)
        loss, metric = train_step(*batch, model, loss_fn, optim)
        all_callbacks.on_train_batch_end(step, logs)

        if tf.math.is_nan(loss):
            print('nan occurred')
            all_callbacks.on_train_end(logs)
            if epoch>1:
                gen_plots(train_metric, val_metric_list, path_i)
            return np.inf

        if step == load_tr.steps_per_epoch:
            step = 0
            loss_str="Loss: {}".format(loss / load_tr.steps_per_epoch)
            tr_loss, tr_rmse, tr_mae= evaluate(load_tr_eval, model, loss_fn)
            val_loss, val_rmse, val_mae= evaluate(load_va, model, loss_fn)
            val_metric_list.append(val_loss)
            train_metric.append(tr_loss)

            if epoch>0:
                if val_loss<best_val_loss:
                    early_stop_counter=0
                    model.save(checkpoint_path)
                    best_val_loss= val_loss
                else:
                    early_stop_counter+=1

            all_callbacks.on_epoch_end(epoch, {'train_mse':tr_loss, 'train_rmse':tr_rmse, 'train_mae':tr_mae, 'val_mse':val_loss, 'val_rmse:':val_rmse, 'val_mae':val_mae})
            if early_stop_counter==patience:
                all_callbacks.on_train_end(logs)
                gen_plots(train_metric, val_metric_list, path_i)
                return best_val_loss
            else:
                epoch+=1

    all_callbacks.on_train_end(logs)
    gen_plots(train_metric, val_metric_list, path_i)
    return best_val_loss

def target_v_pred_plot(df, path='./'):
    plt.figure()
    plt.scatter(df['target'],df['pred'])
    x = np.arange(-5, 0, 5)
    y = x
    plt.plot(x,y)
    plt.xlabel('target')
    plt.ylabel('pred')
    plt.savefig(path+'target_v_pred.png')

def evaluate_pools_from_disk(model_num):
    poolings= '../400_ternary_all/'+str(model_num)+'/val_set/pooling_eval.csv'
    df= pd.read_csv(poolings)
    return np.sum(df['perfect'])


def evaluate_pools_any_separation(name):
    df=pd.read_csv('../full_tern_varied_patience/'+main_dir+'/'+name+'pool.csv')
    if df['P1'].equals(df['ground_truth_P1'].astype('float64')):
        return 1
    elif df['P2'].equals(df['ground_truth_P1'].astype('float64')):
        return 1
    else:
        return 0

def assign_by_atom(contcar_path, p1):
   from pymatgen.core.structure import Structure
   import random
   crystal= Structure.from_file(os.path.join('../Main_fol_Zintl/',contcar_path))
   s= random.choices([0,1], k=len(crystal))
   correct_list=[]
   for i in range(len(crystal)):
      if str(crystal[i].specie) in p1:
          correct_list.append(1)
      else:
          correct_list.append(0)

   correctcount=0
   for i in range(len(crystal)):
      if s[i]==correct_list[i]:
          correctcount+=1

   if correctcount==0:
      return True
   elif correctcount==len(crystal):
      return True
   else:
      return False

def assign_by_Wyckoff(contcarpath, p1):
   from pymatgen.core.structure import Structure
   from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
   import random
   crystal= Structure.from_file(os.path.join('../Main_fol_Zintl/',contcarpath))
   correct_list=[]
   for i in range(len(crystal)):
      if str(crystal[i].specie) in p1:
          correct_list.append(1)
      else:
          correct_list.append(0)

   sga= SpacegroupAnalyzer(crystal)
   all_equiv =sga.get_symmetry_dataset()['equivalent_atoms']
   unique_equiv= np.unique(all_equiv)
   s= random.choices([0,1], k=len(unique_equiv))
   if np.sum(s)==0:
     return -1
   elif np.sum(s)==len(unique_equiv):
     return -1

   full_s= list(range(len(all_equiv)))
   for i in range(len(unique_equiv)):
      for j in range(len(all_equiv)):
          if unique_equiv[i]==all_equiv[j]:
              full_s[j]= s[i]

   correctcount=0
   for i in range(len(crystal)):
      if full_s[i]==correct_list[i]:
          correctcount+=1
   if correctcount==0:
      return 0
   elif correctcount==len(crystal):
      return 0
   else:
      return 1

def assign_by_element(contcarpath, p1):
   from pymatgen.core.structure import Structure
   import random
   crystal= Structure.from_file(os.path.join('../Main_fol_Zintl/',contcarpath))
   correct_list=[]
   species_list=[]
   for i in range(len(crystal)):
      species_list.append(str(crystal[i].specie))
   unique_species= np.unique(species_list)
   s= random.choices([0,1], k=len(unique_species))
   list_p1= p1.split(',')
   for e in unique_species:
      if e in list_p1:
         correct_list.append(0)
      else:
         correct_list.append(1)

   correctcount=0
   for i in range(len(unique_species)):
      if s[i]==correct_list[i]:
         correctcount+=1
   if correctcount==0:
      return True
   elif correctcount==len(unique_species):
      return True
   else:
      return False

def electronegativity_heuristic(name):
    crystal= Structure.from_file(os.path.join('../Main_fol_Zintl/',name))
    species_list=[]
    for i in range(len(crystal)):
        species_list.append(str(crystal[i].specie))
    unique_species= np.unique(species_list).tolist()
    #print(unique_species)
    elc= []
    for i in range(len(unique_species)):
        #print(unique_species[i])
        elc.append(electronegativity_lookup[unique_species[i]])

    cation= unique_species.pop(np.argmin(elc))

    return str(cation), str(unique_species)


def electronegativity_quaternary_heuristic(name):
    crystal= Structure.from_file(os.path.join('../Main_fol_Zintl/',name))
    species_list=[]
    for i in range(len(crystal)):
        species_list.append(str(crystal[i].specie))
    unique_species= np.unique(species_list).tolist()
    #print(unique_species)
    elc= []
    for i in range(len(unique_species)):
        #print(unique_species[i])
        elc.append(electronegativity_lookup[unique_species[i]])

    cation= []
    cation.append(unique_species.pop(np.argmin(elc)))
    elc.remove(min(elc))
    cation.append(unique_species.pop(np.argmin(elc)))
    return str(cation), str(unique_species)

main_dir= 'p50_id143/quaternary_2electro'

if __name__ == "__main__":
    #
    df= pd.read_csv('../full_tern_varied_patience/'+main_dir+'/pooling_eval.csv')

    df['heuristic_match']=df['name'].apply(evaluate_pools_any_separation)
    df.to_csv('../full_tern_varied_patience/'+main_dir+'/pooling_eval.csv')
