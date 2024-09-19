import pandas as pd
import os
import argparse
from spektral_essential_objects import MyDataset, HNetSingleJanossy
import sys
from spektral.data import DisjointLoader
from tensorflow.keras.callbacks import CallbackList, CSVLogger
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError

parser = argparse.ArgumentParser(description='crystal hierarchy arguments.')
parser.add_argument('--datadir', dest='datadir',
        help='Directory where dataset is located', default='/Users/nilamandal/desktop/Main_fol_Zintl')
parser.add_argument('--num-nbrs', dest='num_nbrs', type=int,
                    help='num neighbors per atom', default=12)
parser.add_argument('--radius-angstroms', dest='radius_angstroms', type=int,
                    help='search radius for neighbors', default=10)
parser.add_argument('--task', choices=['r', 'c'],
                    default='r', help='complete a regression or classification task (default: regression)')
args = parser.parse_args(sys.argv[1:])

def train_step(inputs, target, model, loss_fn, optimizer):
    with tf.GradientTape() as tape:
        predictions, s = model(inputs, training=True)
        loss = loss_fn(target, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    if args.task=='r':
        #mse = tf.reduce_mean((target-predictions)**2)
        return loss
    if args.task=='c':
        sca= tf.reduce_mean(sparse_categorical_accuracy(target, predictions))
        return loss, sca

def evaluate(loader, model, loss_fn, test=False):
    step = 0
    output=[]
    while step < loader.steps_per_epoch:
        step += 1
        inputs, target = loader.__next__()
        x, a, e, i = inputs
        pred, s = model(inputs, training=False)

        c_p, r_e= row_e_and_column_p(s, i)
        if args.task=='c':
            outs = (
                loss_fn(target, pred),
                tf.reduce_mean(sparse_categorical_accuracy(target, pred)),
                len(target),  # Keep track of batch size
            )
        elif args.task=='r':
            mse = tf.reduce_mean((target-pred)**2)
            rmse= np.sqrt(mse)
            mae= tf.reduce_mean(np.abs(target-pred))
            outs = (
                loss_fn(target, pred),
                rmse,
                mae)

        return
        # output.append(outs)
        # if step == loader.steps_per_epoch:
        #     output = np.array(output)
        #     return np.average(output[:, :-1], 0, weights=output[:, -1])

def train_model():
    print('BEGUN INDIVIDUAL TRAINING')

    checkpoint_path='./goodmodel.ckpt'

    epochs = 10
    if epochs<1000:
        print('WARNING: CURRENTLY RUNNING IN DEBUG MODE WITH '+str(epochs)+' EPOCHS')

    embedding_size= 16
    batch_size= 16
    entropy_lambda= 16
    softmax_beta= 16
    lr= 0.0001

    # Load data and train model code here...
    train_df = pd.read_csv(os.path.join(args.datadir,'train_no_metals.csv'))
    train_df = train_df.head(20)
    train_data= MyDataset(train_df, args.datadir, args.radius_angstroms, args.num_nbrs, args.task)
    load_tr= DisjointLoader(train_data, batch_size=batch_size, epochs=epochs)
    load_tr_eval_copy= DisjointLoader(train_data, batch_size=len(train_data))
    #
    val_df = pd.read_csv(os.path.join(args.datadir,'val_no_metals.csv'))
    val_df = val_df.head(20)
    val_data= MyDataset(val_df, args.datadir, args.radius_angstroms, args.num_nbrs, args.task)
    load_va= DisjointLoader(val_data, batch_size=len(val_data))

    csv_log = CSVLogger("./callback_results.csv")
    model= HNetSingleJanossy('r', 1, beta=softmax_beta, return_s=True)

    all_callbacks= CallbackList([csv_log], add_history=True, model=model)
    # #
    optim=Adam(lr)
    loss_fn= MeanSquaredError()
    #
    train_metric=[]
    val_metric_list=[]
    early_stop_counter= 0
    patience= 100
    epoch = step = 0
    #
    best_val_mse = np.inf

    logs = {}
    all_callbacks.on_train_begin(logs=logs)
    for batch in load_tr:
        if step==0:
            all_callbacks.on_epoch_begin(epoch, logs=logs)
        step += 1
    #
        all_callbacks.on_train_batch_begin(step)
        loss, metric = train_step(*batch, model, loss_fn, optim)
        all_callbacks.on_train_batch_end(step, logs)
    #
        if tf.math.is_nan(loss):
            all_callbacks.on_train_end(logs)
            if epoch>1:
                gen_plots(train_metric, val_metric_list)
            return {"score": np.inf}

        if step == load_tr.steps_per_epoch:
            step = 0
            loss_str="Loss: {}".format(loss / load_tr.steps_per_epoch)

            tr_loss, tr_mse, tr_rmse, tr_mae, tr_ce, tr_re= evaluate(load_tr_eval, model, loss_fn)
    #             val_loss, val_mse, val_rmse, val_mae, val_ce, val_re = evaluate(load_va, model, loss_fn)
    #             val_metric_list.append(val_loss)
    #             train_metric.append(tr_loss)
    #             total_val_loss= val_mse
    #
    #             if epoch>0:
    #                 if total_val_loss<best_val_loss:
    #                     early_stop_counter=0
    #                     model.save_weights(checkpoint_path)
    #                     best_val_loss= total_val_loss
    #                     best_model_mse= val_mse
    #                 else:
    #                     early_stop_counter+=1
    #
    #             all_callbacks.on_epoch_end(epoch, {'train_mse':tr_mse, 'train_rmse':tr_rmse, 'train_mae':tr_mae, 'val_mse':val_mse, 'val_rmse:':val_rmse, 'val_mae':val_mae, 'train_row_penalty':tr_re, 'train_column_penalty':tr_ce, 'val_row_penalty':val_re, 'val_column_penalty':val_ce, 'val_total':total_val_loss})
    #
    #             if early_stop_counter==patience:
    #                 all_callbacks.on_train_end(logs)
    #                 gen_plots(train_metric, val_metric_list)
    #                 return {"score": best_model_mse}
    #             else:
    #                 epoch+=1
    # all_callbacks.on_train_end(logs)
    # gen_plots(train_metric, val_metric_list)
    #
    # return {"score": best_model_mse}

if __name__ == "__main__":
    train_model()
