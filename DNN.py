# install dependencies for MultilabelStratifiedKFold
#!pip install iterative-stratification

# import libraries
import sys
sys.path.append('../input/iterative-stratification/iterative-stratification-master')
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

import warnings
warnings.filterwarnings("ignore")

import math
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.layers as L
import tensorflow.keras.models as M
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras.losses import BinaryCrossentropy
import tensorflow_addons as tfa
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss
from sklearn import preprocessing
from tqdm.notebook import tqdm

#load data
train_features = pd.read_csv('../input/lish-moa/train_features.csv')
train_targets = pd.read_csv('../input/lish-moa/train_targets_scored.csv')
test_features = pd.read_csv('../input/lish-moa/test_features.csv')

data = train_features.append(test_features)

ss = pd.read_csv('../input/lish-moa/sample_submission.csv')

#preprocess the data
def preprocess(df):
    df = df.copy()
    #df.loc[:, 'cp_type'] = df.loc[:, 'cp_type'].map({'trt_cp': 0, 'ctl_vehicle': 1})
    df.drop(['cp_type'], axis=1, inplace=True)
    df.loc[:, 'cp_dose'] = df.loc[:, 'cp_dose'].map({'D1': 0, 'D2': 1})
    df.loc[:, 'cp_time'] = df.loc[:, 'cp_time'].map({24: 0, 48: 1, 72: 2})    
    df = pd.get_dummies(df, columns=['cp_time','cp_dose'])
    del df['sig_id']
    return df

train = preprocess(train_features)
test = preprocess(test_features)

del train_targets['sig_id']

#scale the train and test data
scaler = preprocessing.MinMaxScaler()
scaler.fit(train.append(test))

train_trans = scaler.transform(train)
test_trans = scaler.transform(test)

train = pd.DataFrame(train_trans, columns=train.columns)
test = pd.DataFrame(test_trans, columns=test.columns)

# define the log loss function
somthing_rate = 1e-15
P_MIN = somthing_rate
P_MAX = 1 - P_MIN

def loss_fn(yt, yp):
    yp = np.clip(yp, P_MIN, P_MAX)
    return log_loss(yt, yp, labels=[0,1])

#create the keras model
def create_model(num_columns, actv='relu'):
    
    #input layer
    model = tf.keras.Sequential([tf.keras.layers.Input(num_columns)])
                
    #first hidden layer, L1
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tfa.layers.WeightNormalization(tf.keras.layers.Dense(1024, kernel_initializer='he_normal', activation=actv)))
    
    #second hidden layer, L2
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tfa.layers.WeightNormalization(tf.keras.layers.Dense(1024, kernel_initializer='he_normal', activation=actv))) 

    #third hidden layer, L3
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.AlphaDropout(0.2))
    model.add(tfa.layers.WeightNormalization(tf.keras.layers.Dense(512, kernel_initializer='lecun_normal', activation='selu')))

    #output layer
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tfa.layers.WeightNormalization(tf.keras.layers.Dense(206, activation="sigmoid")))
    
    #compile the model
    model.compile(optimizer=tfa.optimizers.AdamW(lr = 1e-3, weight_decay = 1e-5, clipvalue = 756), 
                  loss=BinaryCrossentropy(label_smoothing=somthing_rate),
                  )
    return model

#select all features as top features
top_feats = [i for i in range(train.shape[1])]
print("Top feats length:",len(top_feats))

#create model object
mod = create_model(len(top_feats))
mod.summary()

#define evaluation metric
def metric(y_true, y_pred):
    metrics = []
    for _target in train_targets.columns:
        metrics.append(loss_fn(y_true.loc[:, _target], y_pred.loc[:, _target].astype(float)))
    return np.mean(metrics)

#set the seeds for cross validation
N_STARTS = 7
#S_STARTS = int(N_STARTS/2) 

res_relu = train_targets.copy()
res_relu.loc[:, train_targets.columns] = 0

ss_relu = ss.copy()
ss_relu.loc[:, train_targets.columns] = 0

ss_dict = {}

historys = dict()

tf.random.set_seed(42)
for seed in range(N_STARTS):
    for n, (tr, te) in enumerate(MultilabelStratifiedKFold(n_splits=7, random_state=seed, shuffle=True).split(train_targets, train_targets)):

        model = create_model(len(top_feats), actv='relu')

        #define checkpoint path
        checkpoint_path = f'repeat:{seed}_Fold:{n}.hdf5'

        #callbacks
        #adaptive learning rate
        reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.2, min_lr=1e-6, patience=4, verbose=1, mode='auto')
        #model checkpoint
        cb_checkpt = ModelCheckpoint(checkpoint_path, monitor = 'val_loss', verbose = 1, save_best_only = True,
                                     save_weights_only = True, mode = 'auto')
        #early stopping
        early = EarlyStopping(monitor="val_loss", mode="min", restore_best_weights=True, patience= 10, verbose = 1)
        
        #model history
        history = model.fit(train.values[tr][:, top_feats],
                  train_targets.values[tr],
                  validation_data=(train.values[te][:, top_feats], train_targets.values[te]),
                  epochs=60, batch_size=128,
                  callbacks=[reduce_lr_loss, cb_checkpt, early], verbose=2
                 )
        
        historys[f'history_{seed+1}'] = history
        
        #load model weights
        model.load_weights(checkpoint_path)
        
        #make predictions on validation and testing sets
        test_predict = model.predict(test.values[:, top_feats])
        val_predict = model.predict(train.values[te][:, top_feats])

        ss_relu.loc[:, train_targets.columns] += test_predict
        res_relu.loc[te, train_targets.columns] += val_predict

ss_relu.loc[:, train_targets.columns] /= ((n+1) * S_STARTS)
res_relu.loc[:, train_targets.columns] /= S_STARTS

#show model loss in plot
for k,v in historys.items():
    loss = []
    val_loss = []
    loss.append(v.history['loss'][:40])
    val_loss.append(v.history['val_loss'][:40])
    
import matplotlib.pyplot as plt
fig=plt.figure(figsize = (10, 6))
plt.plot(np.mean(loss, axis=0),marker='o', linestyle='--', color='tab:olive', alpha =0.6)
plt.plot(np.mean(val_loss, axis=0),marker='o', linestyle='--', color='tab:cyan', alpha =0.6)
plt.yscale('log')
plt.yticks(ticks=[1,1e-1,1e-2], fontsize=12, family='serif')
plt.xticks(fontsize=12, family='serif')
plt.xlabel('Epochs',fontsize=15, family='serif')
plt.ylabel('Average log loss',fontsize=15, family='serif')
plt.legend(['Training','Validation'])
fig.savefig('loss.pdf',format='pdf')

# %% [code]
ss_relu.to_csv('submission_relu.csv', index=False)
#ss_elu.to_csv('submission_elu.csv', index=False)