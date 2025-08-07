import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Dropout, Input
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


from spektral.data import Graph, Dataset, DisjointLoader

import json
import pandas as pd
from scipy.spatial import distance
import argparse
import sys
from sklearn.model_selection import train_test_split
from spektral_essential_objects import NotShrinking
from train_single_model import gen_plots, train_step


parser = argparse.ArgumentParser(description='crystal hierarchy arguments.')
parser.add_argument('--datadir', dest='datadir',
        help='Directory where dataset is located', default='/Users/nilamandal/desktop/Main_fol_Zintl')
#parser.add_argument('--num-nbrs', dest='num_nbrs', type=int,
#                    help='num neighbors per atom', default=12)
#parser.add_argument('--radius-angstroms', dest='radius_angstroms', type=int,
#                    help='search radius for neighbors', default=10)
parser.add_argument('--task', choices=['r', 'c'],
                    default='c', help='complete a regression or classification task (default: classification)')
args = parser.parse_args(sys.argv[1:])

class Dataset_from_json(Dataset):
    def __init__(self, df, task='c'):
        self.df= df
        ref= pd.read_csv('PeriodicTableCSV.csv')[['symbol', 'period', 'group']]
        self.element_ref = {}
        for i, row in ref.iterrows():
            self.element_ref[row.symbol] = [row.period, row.group]

        super().__init__()

    def read(self):
        pre_graph= self.df['structure'].tolist()
        graphs=[]
        for g in pre_graph:
            g2= self.make_dataset(g)
            graphs.append(g2)

        return graphs

    def make_dataset(self, g):
        json_graph= json.loads(g.replace("'", "\""))
        atoms= json_graph['atoms']
        bonds= json_graph['bonds']

        x=[] #atom features
        for i in range(len(atoms)):
            a= atoms[i]
            encoding= self.element_ref[a['element']]
            group_encoding= np.zeros(18)
            group_encoding[encoding[1]-1]=1
            row_encoding= np.zeros(9)
            row_encoding[encoding[0]-1]=1
            atom_hot=np.concatenate((group_encoding, row_encoding))
            x.append(atom_hot)
        x= np.vstack(x)
        #print(x)

        #adjacency matrix and edge feature(s)
        adj = np.zeros((len(atoms), len(atoms)))
        e= np.zeros((len(atoms), len(atoms)))
        for i in range(len(bonds)):
            b= bonds[i]
            idx_u= b['aid1']-1
            idx_v= b['aid2']-1
            adj[idx_u, idx_v]= b['order']
            adj[idx_v, idx_u]= b['order']
            u= atoms[idx_u]
            v= atoms[idx_v]
            dist=distance.euclidean([u['x'],u['y']], [v['x'],v['y']])
            e[idx_u, idx_v]= dist
            e[idx_v, idx_u]= dist
        #print(e)

        MG=Graph(x=x, a=adj, e=e)
        return MG

def main_workflow(config):

    checkpoint_path='./goodmodel.ckpt'

    epochs = 1000
    if epochs<1000:
        print('WARNING: CURRENTLY RUNNING IN DEBUG MODE WITH '+str(epochs)+' EPOCHS')

    df= pd.read_csv('./tox_train_set.csv')
    cv_scores=[]
    for i in range(5):
        train= df[df['fold']!=i]
        train= Dataset_from_json(train)

        val= df[df['fold']==i]
        val= Dataset_from_json(val)

        load_train= DisjointLoader(train, batch_size=int(config['batch_size']), epochs=epochs)
        load_train_eval= DisjointLoader(train, batch_size=len(train))
        load_val= DisjointLoader(val, batch_size=len(val))

        csv_log = CSVLogger("./callback_results"+str(i)+".csv")
        model= SparseEdgepool('r', 1, embedding_size=int(config['embedding_size']), cgcnn_num=int(config['cgcnn_num']), cgcnn_num2=int(config['cgcnn_num2']), softmax_beta=config['softmax_beta'], return_s=True)
        all_callbacks= CallbackList([csv_log], add_history=True, model=model)
        #
        optim=Adam(config['lr'])
        loss_fn= SparseCategoricalCrossentropy()

        for batch in load_tr:
            if step==0:
                all_callbacks.on_epoch_begin(epoch, logs=logs)
            step += 1

            all_callbacks.on_train_batch_begin(step)
            loss, metric = train_step(*batch, model, loss_fn, optim)
            all_callbacks.on_train_batch_end(step, logs)

            if tf.math.is_nan(loss):
                all_callbacks.on_train_end(logs)
                if epoch>1:
                    gen_plots(train_metric, val_metric_list)
                return {"score": np.inf}

            if step == load_tr.steps_per_epoch:
                step = 0
                loss_str="Loss: {}".format(loss / load_tr.steps_per_epoch)

                tr_loss, tr_mse, tr_rmse, tr_mae, tr_ce, tr_re= evaluate(load_tr_eval, model, loss_fn)
                val_loss, val_mse, val_rmse, val_mae, val_ce, val_re = evaluate(load_va, model, loss_fn)
                val_metric_list.append(val_loss)
                train_metric.append(tr_loss)
                total_val_loss= val_mse

                if epoch>0:
                    if total_val_loss<best_val_loss:
                        early_stop_counter=0
                        model.save_weights(checkpoint_path)
                        best_val_loss= total_val_loss
                        best_model_mse= val_mse
                    else:
                        early_stop_counter+=1

                all_callbacks.on_epoch_end(epoch, {'train_mse':tr_mse, 'train_rmse':tr_rmse, 'train_mae':tr_mae, 'val_mse':val_mse, 'val_rmse:':val_rmse, 'val_mae':val_mae, 'train_row_penalty':tr_re, 'train_column_penalty':tr_ce, 'val_row_penalty':val_re, 'val_column_penalty':val_ce, 'val_total':total_val_loss})

                if early_stop_counter==patience:
                    all_callbacks.on_train_end(logs)
                    gen_plots(train_metric, val_metric_list)
                    return {"score": best_model_mse}
                else:
                    epoch+=1
    all_callbacks.on_train_end(logs)
    gen_plots(train_metric, val_metric_list)

    return {"score": best_model_mse}


if __name__ == "__main__":
    df= pd.read_csv('./tox_train_set.csv')
    df0= df[df['tox_y_n']==0]

    df1= df[df['tox_y_n']!=0]
    df0['target']= 0
    df1['target']=1
    #print(df0)
    #print(df1)
    df_new= pd.concat([df0,df1])
    df_new.to_csv('./tox_train_set_2.csv')
    #config={'batch_size': 256,
    #        'lr': 1e-4
    #}
    #main_workflow(config)
