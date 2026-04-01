#import pubchempy as pcp
import pandas as pd
import tensorflow as tf
#from tensorflow.keras.metrics import sparse_categorical_accuracy, categorical_accuracy
import numpy as np
import json
import os
from spektral.data import Graph, Dataset, DisjointLoader
from tensorflow.keras.callbacks import CallbackList, CSVLogger
#from spektral_essential_objects import NotShrinking
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
import matplotlib.pyplot as plt
from spektral_essential_objects import MyDataset, SparseEdgepool, AtomFeaDataset, NotShrinking, CGCNNModel
# def pcp_query_by_smile(formula):
#     try:
#         d= pcp.get_compounds(formula, namespace='smiles', record_type='2d')
#         #temp= d[0].to_dict(properties=['atoms', 'bonds'])
#         print(d)
#         return d
#     except:
#         return 'retry'
#     return 'retry'
#
# def get_bonds_by_pid(pid):
#     #print(pid)
#     try:
#         d= pcp.Compound.from_cid(pid)
#         g= d.to_dict(properties=['atoms', 'bonds'])
#         print(pid)
#         return g
#     except:
#         print(pid, 'retry')
#         return 'retry'

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
    #import ray
    #print(ray.__version__)
    #import ConfigSpace
    #print(ConfigSpace.__version__)
    import optuna
    print(optuna.__version__)


def entropy_loss(s):
    entr = tf.negative(
        tf.reduce_sum(tf.multiply(s, tf.math.log(s + 10**-30)), axis=-1)
    )
    entr_loss = tf.reduce_mean(entr, axis=-1)
    return entr_loss

def row_e_and_column_p(s, i):
    batch_size= s.shape[0]
    column_prod_sum=0
    row_entropy_sum=0

    for g in range(batch_size):
        count= np.count_nonzero(i==g)
        s_g=s[g,:count]
        #---
        row= entropy_loss(s_g)
        row_entropy_sum+=row

        column_product= tf.math.reduce_prod(tf.divide(tf.reduce_sum(s_g, axis=0),s_g.shape[0]))
        column_prod_sum+= column_product

    return -1*column_prod_sum, row_entropy_sum



def get_available(filename):
    try:
        crystal= Structure.from_file('../Main_fol_Zintl/'+filename)
        ana= SpacegroupAnalyzer(crystal)
        #print(ana)
        sym_crystal= ana.get_symmetrized_structure()
        #print(len(sym_crystal.equivalent_indices))
        return len(sym_crystal.equivalent_indices)

    except:
        print(filename)
        #return False

def scale_by_pred_vol(structure, site_bias, dls_vol_predictor):
    #global count
    # first predict the volume using the average volume per element (from ICSD)
    site_counts = pd.Series(Counter(
        str(site.specie) for site in structure.sites)).fillna(0)
    curr_site_bias = site_bias[site_bias.index.isin(site_counts.index)]

    try:
        linear_pred = site_counts @ curr_site_bias
        structure.scale_lattice(linear_pred)
    except:
        pass
        #count+=1
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
        #print(target)
        #print(predictions)
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
        #x, a, e, i = inputs
        #pred, s = model(inputs, training=False)
        pred = model(inputs, training=False)

        #c_p, r_e= row_e_and_column_p(s, i)
        if task=='c':
            outs = (
                loss_fn(target, pred),
                #tf.reduce_mean(categorical_accuracy(target, pred)),
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
    #print('loop entered')
    all_callbacks= CallbackList([csv_log], add_history=True, model=model)


    loss_fn= MeanSquaredError()
    #else:
    #    loss_fn= CategoricalCrossentropy()


    train_metric=[]
    val_metric_list=[]
    early_stop_counter= 0
    patience= 50
    epoch = step = 0

    best_val_loss = np.inf
    logs = {}
    all_callbacks.on_train_begin(logs=logs)

    for batch in load_tr:
        #print(epoch, step)
        if step==0:
            all_callbacks.on_epoch_begin(epoch, logs=logs)
        step += 1
        #print(epoch, step)

        all_callbacks.on_train_batch_begin(step)
        loss, metric = train_step(*batch, model, loss_fn, optim)
        all_callbacks.on_train_batch_end(step, logs)
        #print('---')
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
            #total_val_loss= val_loss
            print(val_loss)
            if epoch>0:
                if val_loss<best_val_loss:
                    early_stop_counter=0
                    model.save(checkpoint_path)
                    best_val_loss= val_loss

                else:
                    early_stop_counter+=1
                #if epoch%50==0:
                   #trial.report(val_loss,epoch)
                   #print(trial)
                   #if trial.should_prune():
                   #    print(trial)
                   #    raise optuna.TrialPruned()
            #if args.task=='r':
            all_callbacks.on_epoch_end(epoch, {'train_mse':tr_loss, 'train_rmse':tr_rmse, 'train_mae':tr_mae, 'val_mse':val_loss, 'val_rmse:':val_rmse, 'val_mae':val_mae})
            #else:
            #    all_callbacks.on_epoch_end(epoch, {'train_loss':tr_loss, 'val_loss:':val_loss})
            if early_stop_counter==patience:
                all_callbacks.on_train_end(logs)
                gen_plots(train_metric, val_metric_list, path_i)
                return best_val_loss
            else:
                epoch+=1

    all_callbacks.on_train_end(logs)
    gen_plots(train_metric, val_metric_list, path_i)

    return best_val_loss


if __name__ == "__main__":
    check_env_versions()
    parampath= './800_cgcnn/143/'
    datadir= '/home/nim18004/Main_fol_Zintl/'
    trial_name= './800_cgcnn/final/'

    config= json.load(open(parampath+'params.json'))

    train_csv= pd.read_csv(datadir+'train_by_fam_ternary.csv')
    val_csv= pd.read_csv(datadir+'val_by_fam_ternary.csv')

    epochs = 1000
    if epochs<1000:
        print('WARNING: CURRENTLY RUNNING IN DEBUG MODE WITH '+str(epochs)+' EPOCHS')

    train_data= AtomFeaDataset(train_csv, datadir, 8, 12, 'r')
    val_data= AtomFeaDataset(val_csv, datadir, 8, 12, 'r')
    load_tr= DisjointLoader(train_data, batch_size=config['batch_size'], epochs=epochs)
    load_tr_eval= DisjointLoader(train_data, batch_size=len(train_data))
    load_va= DisjointLoader(val_data, batch_size=len(val_data))

    #model= NotShrinking('r', 1, config['embedding_size'], config['cgcnn_num'], config['cgcnn_num2'], softmax_beta=config['softmax_beta'])
    model= CGCNNModel(config['embedding_size'], config['hidden_size'], config['num_layers'])
    optim=Adam(config['lr'], clipnorm=config['clipnorm'])

    if not os.path.exists(trial_name):
        os.makedirs(trial_name)
    print(trial_name)
    with open(trial_name+'params.json', 'w') as to_file:
        json.dump(config, to_file)
    trial_score= train_single_model(model, load_tr, load_tr_eval, load_va, optim, 'final', trial_name)
#(model, load_tr, load_tr_eval, load_va, optim, trial, path_i
