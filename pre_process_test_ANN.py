"""
EEG Data Preprocessing and Feature Extraction Pipeline for testing and submission 

This script is designed to preprocess EEG (electroencephalogram) data for seizure detection. It incorporates various signal processing techniques, including filtering, segmentation, and feature extraction using wavelet transforms, to prepare the data for classification.

Authors:
- Ayman Kabawa
- Quỳnh Anh Nguyễn
- Jana Taube

Functions:
- `bandpass`: Filters the signal within a specific frequency band.
- `apply_notch_filter`: Removes a specific frequency from the signal.
- `segmentation_train`: Splits the signal into segments based on seizure events.
- `data_preprocess`: Applies notch and bandpass filters, and performs segmentation.
- `wavelet_features`: Extracts wavelet-based features from each segment.
- `reshape_and_scale`: Scaling the data using scaler as input parameter

Requirements:
- Python 3.x
- Libraries: Scikit-learn, Pandas, NumPy, Scipy, Pywt, typing
- Custom modules: wettbewerb

"""

from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
from tensorflow.keras.models import load_model, Model, Sequential
#import keras

from wettbewerb import get_3montages
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal
from pywt import wavedec
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import keras 
import tensorflow as tf 
from scipy import signal
from sklearn.metrics import roc_curve, roc_auc_score, auc

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, f1_score
from imblearn.under_sampling import RandomUnderSampler
import random
from typing import List
from wettbewerb import get_3montages, get_6montages
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, freqz, iirnotch
import scipy
from scipy.signal import find_peaks


def bandpass(data: np.ndarray, edges: List[float], sample_rate: float, order: int = 8 ) -> np.ndarray:
    """
    Applies a Butterworth bandpass filter to the input data.

    This function designs a second-order Butterworth bandpass filter and applies it to the input data using
    forward and backward filtering.

    Parameters:
    - data (np.ndarray): Input data to be filtered.
    - edges (List[float]): A list containing the low and high frequencies defining the bandpass filter.
    - sample_rate (float): The sampling rate of the input data.
    - order (int, optional): The order of the Butterworth filter. Default is 8.

    Returns:
    np.ndarray: Filtered data after applying the bandpass filter.
    """

    # Design a second-order Butterworth bandpass filter
    sos = scipy.signal.butter(order, edges, 'bandpass', fs=sample_rate, output='sos')

    # Apply the filter to the input data using forward and backward filtering
    filtered_data = scipy.signal.sosfiltfilt(sos, data)

    return filtered_data

def apply_notch_filter(data, notch_freq, fs, quality_factor=30):
    """
    Applies a notch filter to the input data.

    Parameters:
        - data: Input data to be filtered.
        - notch_freq: Frequency of the notch filter.
        - fs: Sampling rate of the input data.
        - quality_factor: Quality factor of the notch filter. Defaults to 30.

    Returns:
        - y (np.ndarray): Filtered data.

    Note:
        This function uses an Infinite Impulse Response (IIR) notch filter.
    """
    
    # Design an IIR notch filter
    b, a = iirnotch(notch_freq, quality_factor, fs)

    # Apply the filter to the input data using linear filtering
    y = lfilter(b, a, data)

    return y

def segmentation_train(data_montage, segment_duration=None):
    """
    Segments EEG data based on seizure onset and offset timings "without overlapping".

    Parameters:
    - data_montage (np.ndarray): NumPy array containing EEG signals (montages x samples)
    - segment_duration (float, optional): Duration of each segment in seconds (default: 30 seconds)
    
    Returns:
    - segmented_data (list of np.ndarray): List of NumPy arrays containing segmented EEG data
    """
    segmented_data = []  # List to store segmented data
   
    # Segment the data with labels
    for i in range(0, data_montage.shape[1] - segment_duration, segment_duration):
        segment_start = i  # Start index of the segment
        segment_end = i + segment_duration  # End index of the segment

        segment = data_montage[:, segment_start:segment_end]  # Extract the segment from EEG data

        segmented_data.append(segment)  # Append segmented data

    return segmented_data


def data_preprocess(data, sampling_f,channels, target_sampling_rate, segment_duration):
    """
    Preprocesses EEG data by applying notch and bandpass filters, resampling,
    and segmenting the data.

    Parameters:
        - data (np.ndarray): Raw EEG data matrix.
        - sampling_f (float): Sampling frequency of the raw data.
        - channels (list): List of channel names in the data.
        - target_sampling_rate (float): Desired sampling rate after resampling.
        - segment_duration (int): Duration of each segment for segmentation.

    Returns:
        - segmented_data_input_all (np.ndarray): Preprocessed and segmented EEG data.
    """
    # Frequency to be notched out    
    notch_freq = 50  # Frequency to be notched out
    
    # Define bandpass filter parameters
    lowcut = 0.5
    highcut = 70
    
    # List to store segmented input data
    segmented_data_input_all = []

    # List to store resampled channels
    resampled_channels = [] 

    # Define all possible EEG channels
    all_channels = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz']
    resampled_data_list = []
    
    resampled_channels = [] 
    for channel_name in all_channels:
        # Check if the channel is present in the data
        if channel_name in channels:
            channel_index = channels.index(channel_name)
            channel_data = data[channel_index, :]
                
        else:
            # If the channel is not present, fill with zeros
            channel_data = np.zeros_like(data[0, :])
            
        # Apply notch filter
        eeg_data_notched = apply_notch_filter(channel_data, 50, sampling_f)
        eeg_data_notched = np.nan_to_num(eeg_data_notched)
        # Apply bandpass filter
        eeg_data_bandpassed = bandpass(eeg_data_notched, [0.5, 70], sampling_f)
        eeg_data_bandpassed = np.nan_to_num(eeg_data_bandpassed)
        # Resample
        resampled_channel_data = signal.resample(eeg_data_bandpassed, int(data.shape[1] * target_sampling_rate / sampling_f))
        resampled_channels.append(resampled_channel_data)
        
    # Convert the list of resampled channels to a NumPy array
    resampled_channels = np.array(resampled_channels)
      
    # Get montages (assuming these functions exist)    
    new_montage, data_montage, is_missing = get_3montages(channels, resampled_channels)
        
    # Perform segmentation on EEG data
    segmented_data_input = segmentation_train(data_montage,segment_duration)
        
    # Reshape the segmented input data for concatenation
    segmented_data_input_reshape = np.array(segmented_data_input).reshape(len(segmented_data_input),segment_duration, len(data_montage))
    # Append segmented data to respective lists
    segmented_data_input_all.append(segmented_data_input_reshape)
                                                                            
    #Concatenate all segmented data along the first axis to create single arrays
    segmented_data_input_all = np.concatenate(segmented_data_input_all, axis=0)
    
    return segmented_data_input_all



def wavelet_features(data, montage_num):
    """
    Extracts wavelet features from EEG data using the Discrete Wavelet Transform (DWT) with function db4 level 5.

    Parameters:
        - data (np.ndarray): EEG data for feature extraction.
        - montage_num (int): Montage number to be included in feature names.

    Returns:
        - features (dict): Dictionary containing wavelet features.
    """
    # Perform Discrete Wavelet Transform (DWT) with 'db4' wavelet and 5 decomposition levels
    coeffs = wavedec(data, 'db4', level=5)
    
    # Dictionary to store wavelet features
    features = {}
    for idx, coeff in enumerate(coeffs):
        # Skip D1 and D2 coefficients
        
        # Calculate energy, entropy, mean, standard deviation, mean absolute value (mav), and zero crossing rate (zcr)
        energy = np.sum(np.square(coeff))
        entropy = np.sum(np.square(coeff) * np.log(np.square(coeff) + 1e-10))  # Avoid log(0)
        mean = np.mean(coeff)
        std = np.std(coeff)
        mav = np.mean(np.abs(coeff))
        zcr = len(find_peaks(coeff)[0]) / len(coeff)
        
        # Label for the coefficient (cA for idx=0, cD_idx for idx>0)
        coeff_label = 'cA' if idx == 0 else f'cD_{idx}'
        
        # Add features to the dictionary with appropriate labels
        features[f'Montage_{montage_num}_{coeff_label}_Energy'] = energy
        features[f'Montage_{montage_num}_{coeff_label}_Entropy'] = entropy
        features[f'Montage_{montage_num}_{coeff_label}_Mean'] = mean
        features[f'Montage_{montage_num}_{coeff_label}_Std'] = std
        features[f'Montage_{montage_num}_{coeff_label}_zcr'] = zcr
        features[f'Montage_{montage_num}_{coeff_label}mav'] = mav

    return features

def reshape_and_scale(data, scaler, fit=False):
    """
    Reshapes and scales EEG data using a MinMaxScaler.

    Parameters:
        - data (np.ndarray): EEG data to be reshaped and scaled.
        - scaler (MinMaxScaler): Scaler object for scaling the data.
        - fit (bool, optional): If True, fit the scaler to the data (default: False).

    Returns:
        - scaled_data (np.ndarray): Reshaped and scaled EEG data.
    """
    # Reshape the data to a 2D array for scaling
    data_reshaped = data.reshape(-1, data.shape[2])
    
    # Fit the scaler to the data if required, and transform the data
    if fit:
        return scaler.fit_transform(data_reshaped).reshape(data.shape)
    else:
        return scaler.transform(data_reshaped).reshape(data.shape)



