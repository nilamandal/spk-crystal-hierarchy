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
parser.add_argument('--file-out', dest='file_out',
                    help='output file name', default='bayes_history')
parser.add_argument('--path-out', dest='path',
                    help='output path', default='./bayes_history')
parser.add_argument('--num-atoms', dest='num_atoms', type=int,
                    help='Maximum number of nodes', default=200)
parser.add_argument('--num-nbrs', dest='num_nbrs', type=int,
                    help='num neighbors per atom', default=12)
parser.add_argument('--num-classes', dest='num_classes', type=int,
                    help='Number of label classes', default=1)
parser.add_argument('--radius-angstroms', dest='radius_angstroms', type=int,
                    help='search radius for neighbors', default=10)
parser.add_argument('--random-seed', dest='random_seed', type=int,
                    help='random seed for numpy', default=2)
parser.add_argument('--optim', default='SGD', type=str, metavar='SGD',
                        help='choose an optimizer, SGD or Adam, (default: SGD)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run (default: 30)')
parser.add_argument('--task', choices=['r', 'c'],
                    default='r', help='complete a regression or '
                        'classification task (default: regression)')
parser.add_argument('--dataset', choices=['prashun', 'mp'],
                    default='prashun')

args = parser.parse_args(sys.argv[1:])

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





class HNetHyperModel(keras_tuner.HyperModel):
    def build(self, hp):
        embedding_size= 52#hp.Int('embedding_size', min_value=64, max_value=128, step=1)


        dr1=hp.Float("drop_rate1", min_value=0, max_value=1.0, step=0.01)
        dr2= hp.Float("drop_rate2", min_value=0, max_value=1.0, step=0.01)


        column_lambda_hp= tf.cast(hp.Float("column_lambda", min_value=0, max_value=4, step=0.1), tf.float32)
        column_lambda= 10**column_lambda_hp
        entropy_lambda_hp=tf.cast(hp.Float("entropy_lambda", min_value=0, max_value=4, step=0.1), tf.float32)
        entropy_lambda= 10**entropy_lambda_hp

        model= HNetConcat('r', 1, embedding_size, dr1, dr2, entropy_lambda, column_lambda)
        self.params=['r', 1, embedding_size, dr1, dr2, entropy_lambda, column_lambda]

        return model



    def fit(self, hp, model, *args, **kwargs):
        #print(model)
        learning_rate_hp= hp.Float("lr_exp", min_value=1, max_value=4, step=0.01)
        lr= 10**(-1*learning_rate_hp)
        optim=Adam(lr)
        loss_fn= MeanSquaredError()
        batch_size = 8 #hp.Int("batch_size", 16, 64, step=4, default=16)
        num_epochs= kwargs['epochs']
        callbacks= kwargs['callbacks']
        te=33

        va=51


        data_tr, data_va, data_te, data_ex= split_for_prashuns_data(data, te, va)
        loader_tr = DisjointLoader(PartitionedData(data_tr), batch_size=batch_size, epochs=num_epochs)
        loader_va = DisjointLoader(PartitionedData(data_va), batch_size=len(data_va))
        model.compile(optimizer=optim, loss=loss_fn)


        epoch_loss_metric = keras.metrics.Mean()

        def run_train_step(graph_inputs, labels):
            with tf.GradientTape() as tape:
                #print('train')
                predictions = model(graph_inputs)
                #print(np.isnan(predictions))
                loss = loss_fn(labels, predictions)
                # Add any regularization losses.
                if model.losses:
                    loss += tf.math.add_n(model.losses)
            gradients = tape.gradient(loss, model.trainable_variables)
            optim.apply_gradients(zip(gradients, model.trainable_variables))

        # Function to run the validation step.
        #@tf.function
        def run_val_step(graph_inputs, labels):
            #print('val')
            predictions = model(graph_inputs, training=False)
            #print(np.isnan(predictions))
            loss = loss_fn(labels, predictions)
            # Update the metric.
            epoch_loss_metric.update_state(loss)

        # Assign the model to the callbacks.
        for callback in callbacks:
            callback.model = model
        # Record the best validation loss value
        best_epoch_loss = float("inf")

        # The custom training loop.
        step=0
        current_epoch=0
        patience=10
        epochs_no_improve=0
        for batch in loader_tr:
            #if step==0:
                #print(f"Epoch: {current_epoch}")
            step += 1

            run_train_step(*batch)

            if step == loader_tr.steps_per_epoch:
                inputs, target = loader_va.__next__()
                run_val_step(inputs, target)
                epoch_loss = float(epoch_loss_metric.result().numpy())

                for callback in callbacks:
                    # The "my_metric" is the objective passed to the tuner.
                    callback.on_epoch_end(current_epoch, logs={"val_loss": epoch_loss})
                epoch_loss_metric.reset_states()

                #print(f"Epoch loss: {epoch_loss}")
                if np.isnan(epoch_loss):
                    break
                elif best_epoch_loss<=epoch_loss:
                    epochs_no_improve+=1
                else:

                    best_epoch_loss = min(best_epoch_loss, epoch_loss)
                    step= 0
                    current_epoch+=1

                if epochs_no_improve==patience:
                    break


        return best_epoch_loss




filename='bayes_history5'#args.path+str(args.random_seed)

my_hyper_model= HNetHyperModel()
tuner= keras_tuner.tuners.BayesianOptimization(
    my_hyper_model,
    objective='val_loss',
    max_trials=10,
    seed=args.random_seed,
    project_name=filename)

data= MyDataset(args.datadir,args.filename, args.radius_angstroms, args.num_nbrs, args.task)
#test_element= 15
#val_element= 33

tuner.search(data, epochs=args.epochs)

print(tuner.results_summary(5))
