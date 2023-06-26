import keras_tuner
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from spektral_essential_objects import GaussianDistance, MyDataset, HNetConcat, PartitionedData
import argparse
import sys
from spektral.data import DisjointLoader
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.losses import MeanSquaredError
import numpy as np

parser = argparse.ArgumentParser(description='crystal hierarchy arguments.')

parser.add_argument('--datadir', dest='datadir',
        help='Directory where dataset is located', default='../cgcnn-pretrained-models/data/10atom_relaxed_cifs')
parser.add_argument('--filename', dest='filename',
                    help='csv where data is located', default='id_mini.csv')
parser.add_argument('--random-seed', dest='random_seed', type=int,
                    help='random seed for numpy', default=2)
parser.add_argument('--num-atoms', dest='num_atoms', type=int,
                    help='Maximum number of nodes', default=200)
parser.add_argument('--num-nbrs', dest='num_nbrs', type=int,
                    help='num neighbors per atom', default=12)
parser.add_argument('--num-classes', dest='num_classes', type=int,
                    help='Number of label classes', default=1)
parser.add_argument('--radius-angstroms', dest='radius_angstroms', type=int,
                    help='search radius for neighbors', default=10)
parser.add_argument('--task', choices=['r', 'c'],
                    default='r', help='complete a regression or '
                        'classification task (default: regression)')
parser.add_argument('--dataset', choices=['prashun', 'mp'],
                    default='prashun')

input_args = parser.parse_args(sys.argv[1:])
filename='bayes_debug'


def split_for_prashuns_data(data, test_element, val_element):
    data_tr=[]
    data_va=[]
    data_te=[]
    data_ex=[]
    for d in data:
        atomset= set(d._atomlist)
        if test_element in atomset:
            if val_element in atomset:
                data_ex.append(d._cif)
            else:
                data_te.append(d)
        elif val_element in atomset:
            data_va.append(d)
        else:
            data_tr.append(d)

    return data_tr, data_va, data_te, data_ex

class LoeoBayesianOptimizer(keras_tuner.tuners.BayesianOptimization):
    def run_trial(self, trial, *args, **kwargs):
        dataset= args[0]
        # Get the hp from trial.
        hp = trial.hyperparameters
        embedding_size= hp.Int('embedding_size', min_value=8, max_value=128, step=4)
        dr1=hp.Float("drop_rate1", min_value=0, max_value=1.0, step=0.01)
        dr2= hp.Float("drop_rate2", min_value=0, max_value=1.0, step=0.01)

        column_lambda_hp= tf.cast(hp.Float("column_lambda", min_value=0, max_value=4, step=0.01), tf.float32)
        column_lambda= 10**column_lambda_hp
        entropy_lambda_hp=tf.cast(hp.Float("entropy_lambda", min_value=0, max_value=4, step=0.01), tf.float32)
        entropy_lambda= 10**entropy_lambda_hp

        learning_rate_hp= hp.Float("lr_exp", min_value=1, max_value=4, step=0.01)
        lr= 10**(-1*learning_rate_hp)
        batch_size = 8
        num_epochs= kwargs['epochs']

        te_list=[83]
        va_list=[33,51,83]
        val_score_manager=[]
        for te in te_list:
            for va in va_list:
                if te!=va:

                    fullpath= filename+'/trial_'+str(trial.trial_id)+'/'+str(te)+'_'+str(va)+'_'+str(trial.trial_id)
                    checkpoint_path=fullpath+'/goodmodel.ckpt'
                    model= HNetConcat('r', 1, embedding_size, dr1, dr2, entropy_lambda, column_lambda)
                    optim=Adam(lr)
                    loss_fn= MeanSquaredError()
                    epoch_loss_metric = keras.metrics.Mean()
                    data_tr, data_va, data_te, data_ex= split_for_prashuns_data(dataset, te, va)
                    loader_tr = DisjointLoader(PartitionedData(data_tr), batch_size=batch_size, epochs=num_epochs)
                    loader_va = DisjointLoader(PartitionedData(data_va), batch_size=len(data_va))
                    model.compile(optimizer=optim, loss=loss_fn)

                    def run_train_step(graph_inputs, labels):
                        with tf.GradientTape() as tape:
                            predictions = model(graph_inputs)
                            loss = loss_fn(labels, predictions)
                            # Add any regularization losses.
                            if model.losses:
                                #print(model.losses)
                                loss += tf.math.add_n(model.losses)
                        gradients = tape.gradient(loss, model.trainable_variables)
                        optim.apply_gradients(zip(gradients, model.trainable_variables))
                        trainloss= loss.numpy()
                        if trainloss<0:
                            print('NEGATIVE')
                            print(trainloss)
                            print(labels, predictions)
                            for item in model.losses:
                                print(item)
                        return loss.numpy()

                    # Function to run the validation step.
                    def run_val_step(graph_inputs, labels):
                        predictions = model(graph_inputs, training=False)
                        loss = loss_fn(labels, predictions)
                        # Update the metric.
                        epoch_loss_metric.update_state(loss)

                    best_epoch_loss = float("inf")

                    # The custom training loop.
                    step=0
                    current_epoch=0
                    patience=10
                    epochs_no_improve=0
                    print('For Trial num '+str(trial.trial_id)+'_'+str(te)+'_'+str(va))
                    for batch in loader_tr:

                        if step==0:
                            print(f"Epoch: {current_epoch}")
                        step += 1

                        train_loss=run_train_step(*batch)

                        if step == loader_tr.steps_per_epoch:
                            inputs, target = loader_va.__next__()
                            run_val_step(inputs, target)
                            epoch_loss = float(epoch_loss_metric.result().numpy())

                            epoch_loss_metric.reset_states()
                            print('train loss= '+str(train_loss))
                            print(f"Epoch loss: {epoch_loss}") #this is val loss
                            if np.isnan(epoch_loss):
                                break
                            elif best_epoch_loss<=epoch_loss:
                                epochs_no_improve+=1
                            else:
                                best_epoch_loss = min(best_epoch_loss, epoch_loss)
                                model.save_weights(checkpoint_path)
                                epochs_no_improve=0
                            step= 0
                            current_epoch+=1

                            if epochs_no_improve==patience:
                                break
                    val_score_manager.append(best_epoch_loss)

        return np.mean(val_score_manager)

tuner= LoeoBayesianOptimizer(
    objective='val_loss',
    max_trials=300,
    seed=input_args.random_seed,
    project_name=filename)

sys.stdout = open(filename+'/out.txt', 'w')
data= MyDataset(input_args.datadir,input_args.filename, input_args.radius_angstroms, input_args.num_nbrs, input_args.task)

tuner.search(data, epochs=250)

print(tuner.results_summary(5))
