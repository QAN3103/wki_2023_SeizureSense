# -*- coding: utf-8 -*-
"""
Beispiel Code und  Spielwiese

"""


import csv
import matplotlib.pyplot as plt
import numpy as np
import os
from wettbewerb import load_references, get_3montages
import mne
from scipy import signal as sig
import ruptures as rpt
import json

## unsere imports:
from split import split_file, load_folder
import pre_process
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from imblearn.over_sampling import RandomOverSampler


### if __name__ == '__main__':  # bei multiprocessing auf Windows notwendig


###--------------------------------------
# split 
# load
# segmentation
# pre prosses (filter + wavelet + downsampling)
# train model


###---------------------------------------- 

training_folder  = "../shared_data/training"



#------------ SPLIT--------------------
#--------------------------------------
# create a number of subsets (equally distributed and without separation of patients), 
# split each subset into train,test and validation 
# save as csv files in subfolders
# return path to subsets
number_subsets = 10
destination_folder = 'split/'
reference_file = 'shared_data/training/REFERENCE.csv'
subsets = split_file(reference_file, number_subsets, destination_folder) # List with path names to subsets

#------------LOAD ------------------
#----------------------------------
target_sampling_rate = 173.61 # get the same sampling rate for all signals
segment_duration = int(5*target_sampling_rate)
features_train, features_val, _ = pre_process.preprocess_all_subsets(subsets, target_sampling_rate, segment_duration)

# save dataframes for faster training later on -> optional
destination_folder = 'wavelet/'
create_wavelet_csv(features_train, features_val, features_test, destination_folder)


#---------TRAIN---------------------
#-----------------------------------

# Divide the dataFrame into X and y data
X_train = df_train.iloc[:, :-1]
y_train = df_train.iloc[:, -1]

X_val = df_val.iloc[:, :-1]
y_val = df_val.iloc[:, -1]



# First Oversample because of unbalanced set
rus = RandomOverSampler(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = rus.fit_resample(X_train, y_train)
X_resampled_val, y_resampled_val = rus.fit_resample(X_val, y_val)


scaler = MinMaxScaler(feature_range=(0, 1))
# Fit the scaler on the training data and transform it
features_train_scaled = pd.DataFrame(scaler.fit_transform(X_resampled), columns = X_resampled.columns)
# Transform the validation data using the same scaler
features_val_scaled= pd.DataFrame(scaler.transform(X_resampled_val), columns = X_resampled_val.columns)

X_train = features_train_scaled
y_train = y_resampled
X_val = features_val_scaled
y_val = y_resampled_val


### ----------- Create the ANN ----------------- ###
def make_model(input_shape):
    
    input_layer = keras.layers.Input(input_shape)
    dense1 = keras.layers.Dense(512,)(input_layer)
    leakyRelU1 = keras.layers.LeakyReLU(alpha=0.01)(dense1)
    
    dense2 = keras.layers.Dense(256)(leakyRelU1)
    leakyRelU2 = keras.layers.LeakyReLU(alpha=0.01)(dense2)

    dense3 = keras.layers.Dense(128)(leakyRelU2)
    leakyRelU3 = keras.layers.LeakyReLU(alpha=0.01)(dense3)
    
    
    dense4 = keras.layers.Dense(32)(leakyRelU3)
    leakyRelU4 = keras.layers.LeakyReLU(alpha=0.01)(dense4)
    dropout3 = keras.layers.Dropout(0.2)(leakyRelU4)

    output_layer = keras.layers.Dense(1, activation="sigmoid")(dropout3)

    return keras.models.Model(inputs=input_layer, outputs=output_layer)

model= make_model((X_train.shape[1]))


class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        # Directly calculate F1 Score
        return 2 * ((p * r) / (p + r + tf.keras.backend.epsilon()))

    def reset_state(self):
        # Resets all of the metric state variables
        self.precision.reset_state()
        self.recall.reset_state()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
    loss= 'binary_crossentropy',
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        F1Score(name='f1_score')
    ])
###------- The Training ----------###

callbacks = [
    keras.callbacks.ModelCheckpoint(
        "model_ANN_wavelet.h5", save_best_only=True, monitor="val_loss", verbose=1,
    ),

    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.1, patience=20, min_lr=0.00000001
    ),
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=1),
]
model.fit(
    X_train,
    y_train,
    batch_size= 64,
    epochs= 250 ,
    shuffle = True ,
    callbacks=callbacks,
    validation_data =(X_val, y_val),
    verbose=1)

