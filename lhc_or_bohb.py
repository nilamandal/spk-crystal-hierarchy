import tensorflow as tf
import os
import sys
import argparse
from spektral_essential_objects import MyDataset, SparseEdgepool
from spektral.data import DisjointLoader
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
import numpy as np
from tensorflow.keras.metrics import sparse_categorical_accuracy #, mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import CallbackList, CSVLogger
from tensorflow.keras import backend as K
import json
import resource
from scipy.stats import qmc
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd

parser = argparse.ArgumentParser(description='crystal hierarchy arguments.')
parser.add_argument('--datadir', dest='datadir',
        help='Directory where dataset is located', default='../Main_fol_Zintl')

parser.add_argument('--task', choices=['r', 'c'],
                    default='r', help='complete a regression or classification task (default: regression)')
args = parser.parse_args(sys.argv[1:])


def evaluate(loader, model, loss_fn, log=False):
    step = 0
    output=[]
    while step < loader.steps_per_epoch:
        step += 1
        inputs, target = loader.__next__()
        x, a, e, i = inputs
        pred, s = model(inputs, training=False)
        #if log:
        #    df= pd.DataFrame(data=[target, pred], columns=['target', 'pred'])
        if args.task=='c':
            outs = (
                loss_fn(target, pred),
                tf.reduce_mean(sparse_categorical_accuracy(target, pred)),
                len(target),  # Keep track of batch size
            )
        elif args.task=='r':
            mse = mean_squared_error(target, pred)
            rmse= np.sqrt(mse)
            mae= mean_absolute_error(target, pred)
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

def train_step(inputs, target, model, loss_fn, optimizer):
    with tf.GradientTape() as tape:
        predictions, s = model(inputs, training=True)
        loss = loss_fn(target, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    if args.task=='r':
        mse = tf.reduce_mean((target-predictions)**2)

        return loss, mse
    if args.task=='c':
        sca= tf.reduce_mean(sparse_categorical_accuracy(target, predictions))
        return loss, sca

def train_model(config):
    print('BEGUN INDIVIDUAL TRAINING')
    write_output_path='./'+config['idx']+'/'
    if not os.path.exists(write_output_path):
      os.makedirs(write_output_path)

    checkpoint_path=write_output_path+'goodmodel.ckpt'

    epochs = 1000
    if epochs<1000:
        print('WARNING: CURRENTLY RUNNING IN DEBUG MODE WITH '+str(epochs)+' EPOCHS')

    # Load data and train model code here...
    train_df = pd.read_csv(os.path.join(args.datadir,'train_no_metals.csv'))
    #train_df = train_df.head(20)
    train_data= MyDataset(train_df, args.datadir, 8, int(config['num_nbrs']), args.task)
    load_tr= DisjointLoader(train_data, batch_size=int(config['batch_size']), epochs=epochs)
    load_tr_eval= DisjointLoader(train_data, batch_size=len(train_data))

    val_df = pd.read_csv(os.path.join(args.datadir,'val_no_metals.csv'))

    val_data= MyDataset(val_df, args.datadir, 8, int(config['num_nbrs']), args.task)
    load_va= DisjointLoader(val_data, batch_size=len(val_data))
    print('loaded data')
    csv_log = CSVLogger(write_output_path+"callback_results.csv")

    model= SparseEdgepool('r', 1, embedding_size=int(config['embedding_size']), cgcnn_num=int(config['cgcnn_num']), cgcnn_num2=int(config['cgcnn_num2']), softmax_beta=config['softmax_beta'], return_s=True)

    all_callbacks= CallbackList([csv_log], add_history=True, model=model)
    #
    optim=Adam(config['lr'])
    loss_fn= MeanSquaredError()

    train_metric=[]
    val_metric_list=[]
    early_stop_counter= 0
    patience= 1000
    epoch = step = 0

    best_model_mse = np.inf
    logs = {}
    all_callbacks.on_train_begin(logs=logs)
    for batch in load_tr:
            #print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            # peak memory usage (kilobytes on Linux, bytes on OS X)
            if step==0:
                all_callbacks.on_epoch_begin(epoch, logs=logs)
            step += 1

            all_callbacks.on_train_batch_begin(step)
            loss, metric = train_step(*batch, model, loss_fn, optim)
            all_callbacks.on_train_batch_end(step, logs)

            if tf.math.is_nan(loss):
                all_callbacks.on_train_end(logs)
                if epoch>1:
                    gen_plots(train_metric, val_metric_list,write_output_path)
                with open(write_output_path+'params.json','w') as outfile:
                   config['score']=best_model_mse
                   json_object= json.dumps(config, indent=4)
                   outfile.write(json_object)

                return {"score": best_model_mse}

            if step == load_tr.steps_per_epoch:
                step = 0
                loss_str="Loss: {}".format(loss / load_tr.steps_per_epoch)

                tr_loss, tr_mse, tr_rmse, tr_mae = evaluate(load_tr_eval, model, loss_fn)
                val_loss, val_mse, val_rmse, val_mae = evaluate(load_va, model, loss_fn)
                val_metric_list.append(val_loss)
                train_metric.append(tr_loss)

                #model.save_weights(write_output_path+str(epoch)+'/model.ckpt')
                all_callbacks.on_epoch_end(epoch, {'train_mse':tr_mse, 'train_rmse':tr_rmse, 'train_mae':tr_mae, 'val_mse':val_mse, 'val_rmse:':val_rmse, 'val_mae':val_mae})

                if epoch>0:
                    if val_mse<best_model_mse:
                        early_stop_counter=0
                        model.save_weights(write_output_path+str(epoch)+'/model.ckpt')
                        best_model_mse= val_mse
                    else:
                        early_stop_counter+=1


                if early_stop_counter==patience:
                    all_callbacks.on_train_end(logs)
                    gen_plots(train_metric, val_metric_list, write_output_path)
                    with open(write_output_path+'params.json','w') as outfile:
                       config['score']=best_model_mse
                       json_object= json.dumps(config, indent=4)
                       outfile.write(json_object)
                    return {"score": best_model_mse}
                else:
                    epoch+=1
    all_callbacks.on_train_end(logs)
    gen_plots(train_metric, val_metric_list, write_output_path)
    with open(write_output_path+'params.json','w') as outfile:
      config['score']=best_model_mse
      json_object= json.dumps(config, indent=4)
      outfile.write(json_object)
    return {"score": best_model_mse}


def gen_plots(train_metric, val_metric, savepath):
    plt.switch_backend('Agg')

    plt.figure()

    epochs=list(range(len(train_metric)))
    min_train= 'train min='+str(np.round(np.min(train_metric), decimals=3))+','
    min_val= 'val min='+str(np.round(np.min(val_metric), decimals=3))
    plt.plot(epochs, np.log(train_metric), label='training loss')
    plt.plot(epochs, np.log(val_metric), label='val loss')

    figtitle=savepath+'result.png'

    plt.xlabel('epochs')
    plt.legend()
    plt.ylabel('log of mean square error ')
    plt.savefig(figtitle)


if __name__ == "__main__":
      NUM_MODELS = 100
      trial_space = {
            'embedding_size': tune.choice([4,8,16,32,64]),
            'cgcnn_num': tune.choice([1,2,3]),
            'cgcnn_num2': tune.choice([1,2,3]),
            'num_nbrs': tune.choice([1,2,3,4,5,6,7,8,9,10,11,12]),
            'batch_size': tune.choice([4,8,16,32,64]),
            'softmax_beta': tune.loguniform(1, 1e8),
            'lr': tune.loguniform(1e-8, 1e-1)
      }
      bohb_hyperband = HyperBandForBOHB(
        time_attr="training_iteration",
        max_t=10,
        reduction_factor=4,
        stop_last_trials=False,
      )
      bohb = BOHBRepeater(metric='score', mode='min', repeat=1, max_concurrent=10)
      train_model_object = tune.with_resources(train_model, {"cpu": 1})
      tuner = tune.Tuner(train_model_object, tune_config=tune.TuneConfig(
        search_alg=bohb,
        scheduler=bohb_hyperband,
        metric='score',
        mode='min',
        num_samples=NUM_MODELS), param_space=trial_space)
      print('CREATED all TUNING OBJECTS')
      results = tuner.fit()
      print(results)
      #num_complete_models= 0
      #sampler = qmc.LatinHypercube(d=7, seed=15)
      #sample= sampler.random(n=NUM_MODELS)

      #l_bounds=[1,0,0,0,1,0,-8]
      #u_bounds=[8,5,5,12,8,8,-1]
      #scaled_sample= qmc.scale(sample, l_bounds, u_bounds)
      #i=0

      #for s in scaled_sample:
       #  config={}
        # config['embedding_size']= 2**np.ceil(s[0])
        # config['cgcnn_num']= np.ceil(s[1])
        # config['cgcnn_num2']= np.ceil(s[2])
        # config['num_nbrs']= np.ceil(s[3])
        # config['batch_size']= 2**np.ceil(s[4])
        # config['softmax_beta']= 10**s[5]
        # config['lr']= 10**s[6]
        # config['idx']= 'lhc_15_noearlystop/'+str(i)
        # print(config)
        # try:
        #     train_model(config)
        # except:
        #     pass
        # i+=1
