import pubchempy as pcp
import pandas as pd
import tensorflow as tf
from tensorflow.keras.metrics import sparse_categorical_accuracy, categorical_accuracy
import numpy as np

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

def get_len(path):
        df= pd.read_csv('./train_model_2023-10-27_15-40-05/'+path+'/callback_results.csv')
        return len(df)

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

def train_step(inputs, target, model, loss_fn, optimizer, task='r'):
    with tf.GradientTape() as tape:
        predictions, s = model(inputs, training=True)
        print(target)
        print(predictions)
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
        x, a, e, i = inputs
        pred, s = model(inputs, training=False)

        #c_p, r_e= row_e_and_column_p(s, i)
        if task=='c':
            outs = (
                loss_fn(target, pred),
                #tf.reduce_mean(categorical_accuracy(target, pred)),
                len(target),  # Keep track of batch size
            )
        elif task=='r':
            mse = tf.reduce_mean((target-pred)**2)
            rmse= np.sqrt(mse)
            mae= tf.reduce_mean(np.abs(target-pred))
            outs = (
                loss_fn(target, pred),
                mse,
                rmse,
                mae,
                len(target),  # Keep track of batch size
            )
        output.append(outs)
        if step == loader.steps_per_epoch:
            output = np.array(output)
            return np.average(output[:, :-1], 0, weights=output[:, -1])




if __name__ == "__main__":
    df= pd.read_csv('tox21_updated.csv')
    df= df[df['pid']>0]
    #mol_list= df['smiles'].tolist()

    df['structure']= df['pid'].apply(get_bonds_by_pid)
    df.to_csv('tox21_updated_2.csv')
