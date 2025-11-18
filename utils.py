import pubchempy as pcp
import pandas as pd
import tensorflow as tf
from tensorflow.keras.metrics import sparse_categorical_accuracy, categorical_accuracy
import numpy as np
from spektral.data import Graph, Dataset, DisjointLoader
from tensorflow.keras.callbacks import CallbackList, CSVLogger
from spektral_essential_objects import NotShrinking
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
import matplotlib.pyplot as plt

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
    import ray
    print(ray.__version__)
    import ConfigSpace
    print(ConfigSpace.__version__)


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
        predictions, s = model(inputs, training=True)
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
        pred, s = model(inputs, training=False)

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

def train_single_model(config, train_data, val_data, epochs=1000, save_path= './'):
    print('BEGUN INDIVIDUAL TRAINING')

    checkpoint_path=save_path+'goodmodel.ckpt'

    if epochs<1000:
        print('WARNING: CURRENTLY RUNNING IN DEBUG MODE WITH '+str(epochs)+' EPOCHS')
    #
    # embedding_size= config['embedding_size']
    # batch_size= config['batch_size']
    # entropy_lambda= config['entropy_lambda']
    # softmax_beta= config['softmax_beta']
    # lr= config['lr']

    load_train= DisjointLoader(train_data, batch_size=int(config['batch_size']), epochs=epochs)
    load_train_eval= DisjointLoader(train_data, batch_size=len(train_data))
    load_val= DisjointLoader(val_data, batch_size=len(val_data))
    csv_log = CSVLogger(save_path+"_callback_results.csv")

    model= NotShrinking(config['task'], 2, config['embedding_size'], config['cgcnn_num'], config['cgcnn_num2'], softmax_beta=config['softmax_beta'])
    all_callbacks= CallbackList([csv_log], add_history=True, model=model)
    #
    optim=Adam(config['lr'])
    loss_fn= BinaryCrossentropy()
    early_stop_counter= 0
    patience= 100
    epoch = step = 0
    logs = {}
    all_callbacks.on_train_begin(logs=logs)
    train_metric=[]
    val_metric=[]
    best_val_loss= np.inf

    for batch in load_train:
        #print(batch)
        if step==0:
            all_callbacks.on_epoch_begin(epoch, logs=logs)
        step += 1

        all_callbacks.on_train_batch_begin(step)
        loss, metric = train_step(*batch, model, loss_fn, optim, config['task'])
        all_callbacks.on_train_batch_end(step, logs)

        if tf.math.is_nan(loss):
            all_callbacks.on_train_end(logs)
            if epoch>1:
                gen_plots(train_metric, val_metric, save_path)
            return {"score": np.inf}

        if step == load_train.steps_per_epoch:
            step = 0
            loss_str="Loss: {}".format(loss / load_train.steps_per_epoch)

            tr_loss= evaluate(load_train_eval, model, loss_fn, task=config['task'])#binary BinaryCrossentropy
            val_loss = evaluate(load_val, model, loss_fn, task=config['task'])
            val_metric.append(val_loss)
            train_metric.append(tr_loss)

            if epoch>0:
                if val_loss<best_val_loss:
                    early_stop_counter=0
                    model.save_weights(checkpoint_path)
                    best_val_loss= val_loss

                else:
                    early_stop_counter+=1

            all_callbacks.on_epoch_end(epoch, {'train_loss':tr_loss, 'val_loss':val_loss})

            if early_stop_counter==patience:
                all_callbacks.on_train_end(logs)
                gen_plots(train_metric, val_metric, save_path)
                return {"score": best_val_loss}
            else:
                epoch+=1
    all_callbacks.on_train_end(logs)
    gen_plots(train_metric, val_metric, save_path)

    return best_val_loss


if __name__ == "__main__":
    df= pd.read_csv('tox21_updated.csv')
    df= df[df['pid']>0]
    #mol_list= df['smiles'].tolist()

    df['structure']= df['pid'].apply(get_bonds_by_pid)
    df.to_csv('tox21_updated_2.csv')
