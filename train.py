"""
Seizure Detection Using ANN

This script is intended for the preprocessing of EEG data and the creation of an Artificial Neural Network (ANN) model using Keras for the purpose of seizure detection. The dataset undergoes filtering, downsampling, montage calculation, segmentation, and oversampling to address data imbalance concerns. Additionally, feature scaling using MinMaxScaler is applied. The ANN architecture comprises multiple fully connected layers with LeakyReLU activation functions and dropout for regularization. To mitigate overfitting, the model training incorporates early stopping.

Authors:
- Jana Taube
- Ayman Kabawa
- Quỳnh Anh Nguyễn

Functions:
- make_model(input_shape): Constructs the ANN model with specified architecture.
- main(): Orchestrates data preprocessing, model construction, and the training process.

Classes:
- F1Score: A custom TensorFlow metric class for computing the F1 score as a measure of model performance,
  particularly useful for imbalanced datasets.

Requirements:
- Python 3.x
- Libraries: TensorFlow, Keras, Scikit-learn, Imbalanced-learn, Pandas, NumPy, MNE, Matplotlib, Ruptures
- Custom modules: split, pre_process, wettbewerb

"""

import csv
import numpy as np
import os
from wettbewerb import get_3montages
import mne
from scipy import signal as sig
import ruptures as rpt
import json

## our imports:
from split import split_file
import pre_process
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import RandomOverSampler
import pandas as pd
import tensorflow as tf
from tensorflow import keras


class F1Score(tf.keras.metrics.Metric):
    """
    Custom F1 Score metric class for model evaluation.
    
    Extends TensorFlow's Metric class to calculate the F1 score, which is a harmonic mean of precision and recall.
    Useful for evaluating performance on imbalanced datasets.
    """
    
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
        
def make_model(input_shape):
    """
    Constructs the ANN model architecture using Keras.
    
    Parameters:
    - input_shape: Shape of the input data (number of features).
    
    Returns:
    - A defined Keras model with specified layers and activation functions.
    """
    
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


def main():
    """
    Main function to execute the preprocessing, model construction, and training process.
    """
    target_sampling_rate = 173.61 # get the same sampling rate for all signals
    segment_length = 5 # time in s for length of segment
    segment_duration = int(segment_length*target_sampling_rate) # samples 
    
    # because the data is too large to load at once on the memory we have to divide it into subsets
    number_subsets = 10
    destination_folder = 'split_test_final/'
    reference_file = 'shared_data/training/REFERENCE.csv'
    subsets = split_file(reference_file, number_subsets, destination_folder) # List with path names to subsets
    
    # filter, downsample, calculate montages, and segment the data and label it 
    df_train, df_val, _ = pre_process.preprocess_all_subsets(subsets, target_sampling_rate, segment_duration)# _ is for df_test(to save momeory while training)
    
    # optinal: save the pre-proccessed data
    # destination_folder = 'wavelet/'
    # create_wavelet_csv(df_train, df_val, df_test, destination_folder)
    
    # split in train and val subsets
    X_train = df_train.iloc[:, :-1]
    y_train = df_train.iloc[:, -1]

    X_val = df_val.iloc[:, :-1]
    y_val = df_val.iloc[:, -1]
    
    #resample the data to be balanced(unbalanced training dataset)
    rus = RandomOverSampler(sampling_strategy='auto', random_state=42)
    X_train, y_train = rus.fit_resample(X_train, y_train)
    X_val, y_val = rus.fit_resample(X_val, y_val)
    
    # scale the data 
    scaler = MinMaxScaler(feature_range=(0, 1))
    # Fit the scaler on the training data and transform it
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns = X_train.columns)
    # Transform the validation data using the same scaler
    X_val= pd.DataFrame(scaler.transform(X_val), columns = X_val.columns)
    
    model= make_model((X_train.shape[1]))
    # compile the model with Adam optimizer and binary crossentropy loss function
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
        loss= 'binary_crossentropy',
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            F1Score(name='f1_score')])
    
    callbacks = [
        #save the model after improvment on the val data set
        keras.callbacks.ModelCheckpoint(
            "model.h5", save_best_only=True, monitor="val_loss", verbose=1
        ),
        # reduce the learning rate after 20 epochs of no improvment 
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.1, patience=20, min_lr=0.00000001
        ),
        # Earlystopping, if the model did not improve after 50 epochs to avoid overfitting
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=1)]
    # train the model
    model.fit(
        X_train,
        y_train,
        batch_size= 64,
        epochs= 250 ,
        shuffle = True ,
        callbacks=callbacks,
        validation_data =(X_val, y_val),
        verbose=1)










