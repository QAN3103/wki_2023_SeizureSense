# -*- coding: utf-8 -*-
"""

Skript testet das vortrainierte Modell


@author:  Maurice Rohr, Dirk Schweickard
"""


import csv
import matplotlib.pyplot as plt
import numpy as np
import os
from wettbewerb import load_references, get_3montages
import mne
from scipy import signal as sig
import ruptures as rpt
from random import randint
import numpy as np
import pre_process_test_ANN as pre
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import joblib
from typing import List
import pandas as pd
from typing import List, Tuple, Dict, Any

from tensorflow.keras.models import load_model, Model, Sequential
import tensorflow as tf 
from tensorflow.keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,EarlyStopping,TensorBoard, History, LambdaCallback
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv1D, GaussianNoise,Input,concatenate,MaxPooling1D,Conv2D, MaxPooling2D, Dense, Flatten, Dropout,Lambda, LSTM,GRU, Input,Reshape, Bidirectional, Attention, RepeatVector, TimeDistributed,Activation ,InputLayer, BatchNormalization
from tensorflow.keras.models import load_model, Model, Sequential
from keras import regularizers
import keras 


###Signatur der Methode (Parameter und Anzahl return-Werte) darf nicht verändert werden
def predict_labels(channels : List[str], data : np.ndarray, fs : float, reference_system: str, model_name : str='model.json') -> Dict[str,Any]:
    '''
    Parameters
    ----------
    channels : List[str]
        Namen der übergebenen Kanäle
    data : ndarray
        EEG-Signale der angegebenen Kanäle
    fs : float
        Sampling-Frequenz der Signale.
    reference_system :  str
        Welches Referenzsystem wurde benutzt, "Bezugselektrode", nicht garantiert korrekt!
    model_name : str
        Name eures Models,das ihr beispielsweise bei Abgabe genannt habt. 
        Kann verwendet werden um korrektes Model aus Ordner zu laden
    Returns
    -------
    prediction : Dict[str,Any]
        enthält Vorhersage, ob Anfall vorhanden und wenn ja wo (Onset+Offset)
    '''

#------------------------------------------------------------------------------
# Euer Code ab hier  
#######
    # Initialisiere Return (Ergebnisse)
    seizure_present = True # gibt an ob ein Anfall vorliegt
    seizure_confidence = 0.5 # gibt die Unsicherheit des Modells an (optional)
    onset = 4.2   # gibt den Beginn des Anfalls an (in Sekunden)
    onset_confidence = 0.99 # gibt die Unsicherheit bezüglich des Beginns an (optional)
    offset = 999999  # gibt das Ende des Anfalls an (optional)
    offset_confidence = 0   # gibt die Unsicherheit bezüglich des Endes an (optional)
    
    target_sampling_rate = 173.61
    segment_duration =  868

    if data.shape[1] <= segment_duration:
        seizure_present = False
        onset = 0
    else: 
        segmented_data_input = pre.data_preprocess(data,fs ,channels,target_sampling_rate, segment_duration)

        if segmented_data_input.shape[0] < 1:
            seizure_present = False
            onset = 0
        else:
            # Initialize MinMaxScaler
            scaler = MinMaxScaler(feature_range=(0, 1))
        # Fit and transform the data
        # Scale the datasets
            X_train = pre.reshape_and_scale(segmented_data_input, scaler, fit=True)

            model = keras.models.load_model('time_model.h5')
        #calculate the probability that seizure occurs on each segment
            predictions = model.predict(X_train)
        
            percentile = 0.9
        # Calculate threshold based on the desired percentile of the prediction scores
            threshold = np.percentile(predictions, percentile)
            #median_threshold = np.median(predictions)
            #predictions = [1 if x >= median_threshold else 0 for x in predictions]

        # Apply adaptive threshold
            predictions = [1 if x >= threshold else 0 for x in predictions]
            seizure_segment = []

            # Variables to keep track of the counting
            max_count = 0  # Maximum count of consecutive 1s
            current_count = 0  # Current count of consecutive 1s
            start_index = None  # Start index of the current group of 1s
            max_start_index = None  # Start index of the largest group of 1s

            for i in range(len(predictions)):
                if predictions[i] == 1:
                    current_count += 1
                    if current_count == 1:  # This is the start of a new group of 1s
                        start_index = i
                else:
                    if current_count > max_count:  # Found a larger group of 1s
                        max_count = current_count
                        max_start_index = start_index
                        current_count = 0  # Reset count for the next group

            # Check the last group of 1s in case it's the largest
            if current_count > max_count:
                max_count = current_count
                max_start_index = start_index

            # Append the start index of the largest group of 1s, if such a group was found and it contains more than 1 element
            if max_start_index is not None and max_count >= 2:
                seizure_segment.append(max_start_index)
            
            if not seizure_segment:
                seizure_present = False
                onset = 0
            else:
                seizure_present = True
                #after_seizure = predictions [seizure_segment[0]:]
            
                downsampling_factor = fs / target_sampling_rate #calculate downsampling factor
            
                onset_index_downsampled = seizure_segment[0] #*downsampling_factor
                onset_index_original = onset_index_downsampled * downsampling_factor +1 # in sec
            
                onset = (onset_index_original*segment_duration) / fs
    
#------------------------------------------------------------------------------  
    prediction = {"seizure_present":seizure_present,"seizure_confidence":seizure_confidence,
                   "onset":onset,"onset_confidence":onset_confidence,"offset":offset,
                   "offset_confidence":offset_confidence}
  
    return prediction # Dictionary mit prediction - Muss unverändert bleiben!
                               
                               
        
