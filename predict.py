# -*- coding: utf-8 -*-
"""

Skript testet das vortrainierte Modell


@author:  Maurice Rohr, Dirk Schweickard
"""


import numpy as np
import json
import os
from typing import List, Tuple, Dict, Any
from wettbewerb import get_3montages

# Pakete aus dem Vorlesungsbeispiel
import mne
from scipy import signal as sig
import ruptures as rpt
import Segment_no_labels as sg
from sklearn.preprocessing import MinMaxScaler
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
 
    segmented_data_input_train = sg.segment_all_data(data, channels, fs, segment_duration = 1000)


    # Initialize MinMaxScaler
    Scaler = MinMaxScaler()
    # Fit and transform the data
    X_scaler = Scaler.fit(segmented_data_input_train.reshape(-1, 3))

    X_scaled = X_scaler.transform(segmented_data_input_train.reshape(-1, 3))

    # Reshape the scaled data back to the original shape
    X_scaled = X_scaled.reshape(segmented_data_input_train.shape[0], segmented_data_input_train.shape[2] ,segmented_data_input_train.shape[1])

    model = keras.models.load_model("best_model_overlap.h5")
    #calculate the probability that seizure occurs on each segment
    predictions = model.predict(X_scaled)
    samples_per_seg = 1000 # number of samples pro segment
    segment_duration = samples_per_seg#/fs
    
    #set a threshold of 0.26. Only when equal or higher does seizure occur. Sezure_present = True when seizure occur over 3 segments
    
    seizure_prediction = predictions >= 0.26
    seizure_present = []
    seizure_segment = []
    for i in range(seizure_prediction.shape[0]-2):
        if (seizure_prediction[i][0] == True and seizure_prediction[i+1][0] == True and seizure_prediction [i+2][0]==True):
            seizure_present.append (True)
            seizure_segment.append(i)
        else:
            seizure_present.append(False)
    
    #if seizure occur, onset = first segment index * segment duration
    #offset = onset + distance to the next segment without seizure
    not_seizure = []
    if seizure_present:
        onset = segment_duration*seizure_segment[0]
        rest_segment = seizure_prediction[seizure_segment[0]:]
        not_seizure = np.where(rest_segment==False)[0]
        offset = onset + segment_duration * (not_seizure[0]-1)
        
        
    
#------------------------------------------------------------------------------  
    prediction = {"seizure_present":seizure_present,"seizure_confidence":seizure_confidence,
                   "onset":onset,"onset_confidence":onset_confidence,"offset":offset,
                   "offset_confidence":offset_confidence}
  
    return prediction # Dictionary mit prediction - Muss unverändert bleiben!
                               
                               
        
