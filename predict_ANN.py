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
import Segment_no_labels as sg
import vorfilter
import pre_process_test as pre
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
def predict_labels(channels : List[str], data : np.ndarray, fs : float, reference_system: str, model_name : str='model.json') Dict[str,Any]:
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
        #Pre-process data by using notch filter, bandpass filter, resampling, and then segment it
        segmented_data_input = pre.data_preprocess(data,fs ,channels,target_sampling_rate, segment_duration)
        
        #Feature Extraction using Wavelets
        #montage1 = []
        #montage2 = []
        #montage3 = []

        #for i in range (len(segmented_data_input)):
        #    montage1.append(pre.wavelet_features(segmented_data_input[i][0],1))
        #    montage2.append(pre.wavelet_features(segmented_data_input[i][0],2))
        #    montage3.append(pre.wavelet_features(segmented_data_input[i][0],3))
 
        #df_test = pd.concat((pd.DataFrame(montage1), pd.DataFrame(montage2), pd.DataFrame(montage3)), axis = 1)
        
        data_entries_test = []
        for i, item in enumerate(segmented_data_input):
            features = {}
            
            item = np.transpose (item)
            for montage_idx, channel in enumerate(item, start=1):
            
                channel_features = pre.wavelet_features(channel, montage_idx)
                features.update(channel_features)
    
            data_entries_test.append(features)

        df_test = pd.DataFrame(data_entries_test)
        df_test.replace([np.inf, -np.inf], 0, inplace=True)
        df_test = df_test.fillna(0)
        if df_test.empty:
            seizure_present = False
            onset = 0
        else:
            #Data Scaling
            scaler = MinMaxScaler(feature_range=(0, 1))
            #Fit the scaler on the training data and transform it
            X_test = pd.DataFrame(scaler.fit_transform(df_test), columns = df_test.columns)

            model = keras.models.load_model('wavelet_868_173.6_best.h5')
            #calculate the probability that seizure occurs on each segment
            predictions = model.predict(X_test)
        
            threshold = 0.55
            #predictions = [1 if x >= threshold else 0 for x in predictions]
            
            predictions = custom_prediction_logic(predictions, threshold, consecutive_ones= 3, number_of_zeros=1)
    
            #seizure diagnose, calculate onset/offset
            #Seizure_present = True when seizure occurs on 3 consecutive segments
    
            seizure_segment = []
            for i in range(len(predictions)-2):
                if (predictions[i] == 1 and predictions[i+1] == 1 and predictions[i+2] ==1):
                    seizure_segment.append(i)
                else:
                    pass

            if not seizure_segment:
                seizure_present = False
                onset = 0
            else:
                seizure_present = True
                #after_seizure = predictions [seizure_segment[0]:]
            
                downsampling_factor = fs / target_sampling_rate #calculate downsampling factor
            
                onset_index_downsampled = seizure_segment[0] #*downsampling_factor
                onset_index_orginal = onset_index_downsampled * downsampling_factor +1  # in sec
            
                onset = (onset_index_orginal*segment_duration) / fs 
               
            
                #offset_index = np.where (after_seizure == 0)[0]
                #offset_index_downsampled = int(offset_index / downsampling_factor) # Adjust Offset Indices
                #offset_time_downsampled = offset_index_downsampled / target_sampling_rate #Convert New Indices Back to Seconds

#------------------------------------------------------------------------------  
    prediction = {"seizure_present":seizure_present,"seizure_confidence":seizure_confidence,
                   "onset":onset,"onset_confidence":onset_confidence,"offset":offset,
                   "offset_confidence":offset_confidence}
  
    return prediction # Dictionary mit prediction - Muss unverändert bleiben!
                               
                               
        
# Calculate Downsampling Factor
    #downsampling_factor = original_sampling_frequency / target_sampling_rate
    # Convert New Indices Back to Seconds
    #onset_time_downsampled = onset_index_downsampled / target_sampling_rate
    #offset_time_downsampled = offset_index_downsampled / target_sampling_rate
