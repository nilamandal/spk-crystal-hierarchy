import pandas as pd
from pymatgen.core.structure import Structure
import os
from pymatgen.analysis.structure_prediction.volume_predictor import DLSVolumePredictor
from collections import Counter
import numpy as np
import json
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import backend as K

#this function computes the row regularization value for help with model evaluation.
def entropy_loss(s):
    entr = tf.negative(
        tf.reduce_sum(tf.multiply(s, K.log(s + K.epsilon())), axis=-1)
    )
    entr_loss = tf.reduce_mean(entr, axis=-1)
    return entr_loss

#this is a wrapper function for computing model regularization values
def row_e_and_column_p(s):
    row= entropy_loss(s)
    column_product= tf.math.reduce_prod(tf.divide(tf.reduce_sum(s, axis=0),s.shape[0]))

    return float(column_product), float(row)

def callback_plots():
    crazylist=['../train_model_2023-12-05_10-16-27/']
    for path in crazylist:
        df_getpaths= pd.read_csv(path+'evaluated_results.csv')
        getpaths= list(df_getpaths['path'])
#'train_model_03ec7b5e_41_trial_index=0,batch_size=4,column_lambda=84.4158,dr1=0.7048,embedding_size=128,entropy_lambda=4971905.8384_2023-11-07_21-09-21']
        for subpath in getpaths:
            fullpath=path+str(subpath)+'/'
            df=pd.read_csv(fullpath+'callback_results.csv')
            paramsdict= json.load(open(fullpath+'params.json'))

            el=paramsdict['entropy_lambda']
            cl=paramsdict['column_lambda']
            plt.figure(figsize=(10,10))
            plt.plot(np.log10(-1*df['train_column_penalty']), label='train column product')
            plt.plot(np.log10(df['train_mse']), label='train_mse')
            #plt.plot(np.log10(df['train_mse']), label='train_mse')
            plt.plot(np.log10(df['train_row_penalty']), label='train_row_entropy')
            plt.plot(np.log10(-1*df['val_column_penalty']), label='val column product')
            plt.plot(np.log10(df['val_mse']), label='val_mse')
            plt.plot(np.log10(df['val_row_penalty']), label='val_row_entropy')
            plt.legend()
            plt.xlabel('training epochs')
            plt.ylabel('log10 values of loss function components')
            plt.title(subpath)
            plt.savefig(fullpath+'/allcomponents.png')
            plt.figure(figsize=(10,10))
            df['train_total']=df['train_mse']+(el*df['train_row_penalty'])+(cl*df['train_column_penalty'])
            df['val_total']=df['val_mse']+(el*df['val_row_penalty'])+(cl*df['val_column_penalty'])
            plt.plot(df['train_total'], label='train loss')
            plt.plot(df['val_total'], label='val loss')
            plt.legend()
            plt.xlabel('training epochs')
            plt.ylabel('sum of validation components')
            plt.savefig(fullpath+'/sum_losscomponents.png')

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


def get_available(filename):
    try:
        crystal= Structure.from_file('../Main_fol_Zintl/'+filename+'/CONTCAR')
        return True
    except:
        print(filename)
        return False

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

check_env_versions()

class HNetEdgepool(Model):
    def __init__(self, task, num_classes, embedding_size=52, cgcnn_num=3, cgcnn_num2=3, el=427, cl=265, regularizer='l2', return_s=False,  random_seed=0, path='./', **kwargs):
        super().__init__()
        glorot_initializer= initializers.glorot_uniform(seed=random_seed)
        he_initializer= initializers.he_uniform(seed=random_seed)

        self.return_s=return_s
        self.task=task
        self.num_classes=num_classes

        self.embedding= Dense(embedding_size, kernel_initializer=glorot_initializer, kernel_regularizer=regularizer)

        self.conv_list=[]
        for i in range(cgcnn_num):
            conv= ModifiedCrystalConv(activation= 'softplus', kernel_initializer=he_initializer)#does this l2 have a lambda
            self.conv_list.append(conv)

        self.disjoint2batch= Disjoint2Batch()

        self.pool= RegularizedDiffPool(k=2, kernel_initializer=he_initializer, column_lambda=cl, entr_lambda=el, return_selection=True, path=path)

        self.conv_list2=[]
        for i in range(cgcnn_num2):
            conv= ModifiedCrystalConv(activation= 'softplus', kernel_initializer=he_initializer)
            self.conv_list2.append(conv)

        if self.task=='c':
            self.out_layer= Dense(self.num_classes, activation='softmax', kernel_initializer=glorot_initializer)
        elif self.task=='r':
            self.out_layer= Dense(1, kernel_initializer=glorot_initializer)
        self.saveindex=1
        self.savepath=path


    def call(self, inputs):
        x, a, e, i = inputs
        element_idx=np.empty((len(x)))
        for id in range(len(x)):
            temp=np.nonzero(x[id])[0]
            element_idx[id]=int(str(temp[0])+str(temp[1]))

        x= self.embedding(x)

        for cgcnn in self.conv_list:
            x= cgcnn([x, a, e])
            x= tf.nn.softplus(x)

        batch_X, batch_A= self.disjoint2batch([x, a, i])

        x_pool, a_pool, i_pool, s= self.pool([batch_X, batch_A, i])
        x_pool= tf.reshape(x_pool, [x_pool.shape[0]*x_pool.shape[1], x_pool.shape[2]]) #reshape to disjoint form

        e_pool= self.edgepool(e, a, s, i, a_pool)




        temp_a= tf.unstack(a_pool)
        total_nodes= len(i_pool)
        disjoint_a= np.zeros((total_nodes, total_nodes))
        begin=0
        step=len(temp_a[0])
        end=begin+step

        for j in temp_a:
            disjoint_a[begin:end, begin:end]= np.ones((2,2))
            begin= begin+step
            end= begin+step
        disjoint_a= tf.sparse.from_dense(disjoint_a)
        #print('new adj matrix after pooling')
        #print(disjoint_a)


        for cgcnn2 in self.conv_list2:
            x_pool= cgcnn2([x_pool, disjoint_a])
            x_pool= tf.nn.softplus(x_pool)

        x_pool= tf.reshape(x_pool, [int(x_pool.shape[0]/2), int(x_pool.shape[1]*2)])

        x=self.out_layer(x_pool)

        if self.return_s:
            return x, s
        else:
            return x

    def edgepool(self, e, a, s, i, a_pool):
        indices = a.indices
        i_nodes, j_nodes = indices[:, 0], indices[:, 1]

        graph_sizes = tf.math.segment_sum(tf.ones_like(i), i)
        max_n_nodes = tf.reduce_max(graph_sizes)
        n_graphs = tf.shape(graph_sizes)[0]
        relative_j_nodes = j_nodes - self._vectorised_get_cum_graph_size(j_nodes, graph_sizes)

        new_indices = tf.transpose(tf.stack([i_nodes, relative_j_nodes]))

        new_indices = tf.cast(new_indices, tf.int32)
        n_graphs = tf.cast(n_graphs, tf.int32)
        max_n_nodes = tf.cast(max_n_nodes, tf.int32)

        dense_edge = tf.scatter_nd(
            new_indices, e, (n_graphs * max_n_nodes, max_n_nodes, 41)
        )

        batch_edge = tf.reshape(dense_edge, (n_graphs, max_n_nodes, max_n_nodes, 41))
        batch_edge = tf.cast(batch_edge, tf.float32)

        #print(graph_sizes)
        for g in batch_edge:
            g_debug= tf.math.reduce_sum(g, axis=2)
            print(g_debug)
            print(tf.transpose(g_debug))
            new_debug= tf.math.maximum(g_debug, tf.transpose(g_debug))
            print(new_debug)
            #now make this work when the features each have 41 elements
            print('----------')
        return np.nan

    def _vectorised_get_cum_graph_size(self, nodes, graph_sizes):
        """Takes a list of node ids and graph sizes ordered by segment ID and returns the number of nodes contained in graphs with smaller segment ID.
        :param nodes: List of node ids of shape (nodes)
        :param graph_sizes: List of graph sizes (i.e. tf.math.segment_sum(tf.ones_like(I), I) where I are the segment IDs).
        :return: A list of shape (nodes) where each entry corresponds to the number of nodes contained in graphs with smaller segment ID for each node.
        """
        def get_cum_graph_size(node):
            cum_graph_sizes = tf.cumsum(graph_sizes, exclusive=True)
            indicator_if_smaller = tf.cast(node - cum_graph_sizes >= 0, tf.int32)
            graph_id = tf.reduce_sum(indicator_if_smaller) - 1
            return tf.cumsum(graph_sizes, exclusive=True)[graph_id]

        return tf.map_fn(get_cum_graph_size, nodes)
