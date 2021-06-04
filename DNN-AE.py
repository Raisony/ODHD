#!/usr/bin/env python
# coding: utf-8

import copy
import math
import time
import keras
import scipy
import random
import sklearn
import numpy as np
import pandas as pd
import scipy.io as sio
import keras.backend as K

from keras import layers
from pathlib import Path
from sklearn import metrics
from numpy import linalg as li
from keras.models import Model
from sklearn import preprocessing
from scipy.spatial import distance
from sklearn.externals import joblib
from os.path import dirname, join as pjoin
from tensorflow.keras.constraints import max_norm
from keras.layers.normalization import BatchNormalization
from keras.layers import Input, Conv1D, MaxPool1D, UpSampling1D, Dropout
from sklearn.metrics import mean_squared_log_error, mean_squared_error, mean_absolute_error
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, f1_score

def DataPreprocessing(dataset):
    mat_fname = pjoin('./', dataset)
    print('Loading', mat_fname)
    mat_contents = sio.loadmat(mat_fname)
    X, y = mat_contents['X'], mat_contents['y']
#     feature = X.shape[1]
    y = y.reshape(-1)
    inliers, outliers = np.where(y == 0)[0], np.where(y == 1)[0]#     1 = outliers, 0 = inliers
    return X, y, inliers, outliers
    
def standardization(X, MAX, MIN):
    t = MIN + (MAX - MIN)/2
    print(MAX, MIN, t)
    return (X - t)/np.abs((MAX - MIN)/2)

X_set, y_set, inner, outer = DataPreprocessing('satimage-2.mat')
np.random.seed  = 1


p = outer.tolist() + np.random.choice(inner, 3*len(outer), replace = 0).tolist()
position = [x for x in inner if x not in p]
Xtrain, Xtest, ytrain, ytest = X_set[position], X_set[p], y_set[position],  y_set[p]
Xtrain = standardization(Xtrain, Xtrain.max(), Xtrain.min())
Xtest  = standardization(Xtest,Xtest.max(), Xtest.min())

input_shape = Xtrain.shape[1]
input = keras.Input(shape=(input_shape,))

K.clear_session()
x = layers.Dense(512, activation='sigmoid')(input)
x = layers.Dense(256, activation='sigmoid')(x)
x = layers.Dense(128, activation='sigmoid')(x)
x = layers.Dense(64, activation='sigmoid')(x)
x = layers.Dense(64, activation='sigmoid')(x)
x = layers.Dense(128, activation='sigmoid')(x)
x = layers.Dense(256, activation='sigmoid')(x)
x = layers.Dense(512, activation='sigmoid')(x)
decoded = layers.Dense(input_shape, activation='sigmoid')(x)

autoencoder = keras.Model(input, decoded)
autoencoder.summary()

autoencoder.compile(metrics=['accuracy'], loss=['MeanSquaredError'], optimizer='adam')
history = autoencoder.fit(Xtrain, Xtrain,
                    epochs=30,
                    batch_size=64,
                    shuffle=True,
                    validation_data=(Xtest, Xtest),
                    verbose=1,
                    ).history

test_x_predictions = autoencoder.predict(Xtest)
mse = []
for i in range(Xtest.shape[0]):
    mse.append(mean_squared_error(Xtest[i], test_x_predictions[i]))
mse = np.array(mse)

# print(mse, mse.shape)
error_df = pd.DataFrame({'Reconstruction_error': mse, 'True_class': ytest})
error_meas = np.array(error_df['Reconstruction_error'])
threshold_fixed = (np.mean(error_meas) - 0.25*np.std(error_meas))
groups = error_df.groupby('True_class')
pred_y = [1 if e > threshold_fixed else 0 for e in error_df.Reconstruction_error.values]
error_df['pred'] = pred_y


y_true = error_df['True_class']
y_pred_pos = error_df['pred']

acc = accuracy_score(error_df['True_class'], error_df['pred'])
f1 = f1_score(y_true, y_pred_pos)
ap = average_precision_score(y_true, y_pred_pos)
roc_auc = roc_auc_score(y_true, y_pred_pos)
      
print('DNN-AE')
print('ACC = ', acc)
print('AP = ', ap)
print('F1-score = ', f1)
print('ROC AUC = ', roc_auc)


