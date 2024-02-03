"""
EEG Data Preprocessing and Feature Extraction Pipeline for training

This script is designed to preprocess EEG (electroencephalogram) data for seizure detection. It incorporates various signal processing techniques, including filtering, segmentation, and feature extraction using wavelet transforms, to prepare the data for machine learning model training.

Authors:
- Ayman Kabawa
- Quỳnh Anh Nguyễn
- Jana Taube

Functions:
- `bandpass`: Filters the signal within a specific frequency band.
- `apply_notch_filter`: Removes a specific frequency from the signal.
- `segmentation`: Splits the signal into segments based on seizure events.
- `data_preprocess`: Applies notch and bandpass filters, and performs segmentation.
- `wavelet_features`: Extracts wavelet-based features from each segment.
- `create_feature_dataframe`: Aggregates wavelet features into a DataFrame.
- `preprocess_all_subsets`: Processes multiple data subsets and compiles them into training, validation, and test sets.
- `create_wavelet_csv`: Saves the processed features into CSV files for later use.

Requirements:
- Python 3.x
- Libraries: Scikit-learn, Pandas, NumPy, Matplotlib, Scipy, Pywt
- Custom modules: split, wettbewerb

"""

from typing import List, Tuple
from split import delete_folder_contents, split_file, load_folder
from wettbewerb import get_3montages, get_6montages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, freqz, iirnotch
import scipy
# from Channel_Detection import choose_6montages_bipolar
from scipy.signal import find_peaks
from pywt import wavedec
import os

def bandpass(data: np.ndarray, edges: List[float], sample_rate: float, order: int = 8 ) -> np.ndarray:
    """
    Apply a Butterworth bandpass filter to input data.

    Parameters:
    - data (np.ndarray): Input time-domain data to be filtered.
    - edges (List[float]): List containing the lower and upper frequency cutoffs of the passband.
    - sample_rate (float): Sampling rate of the input data.
    - order (int, optional): Order of the Butterworth filter. Default is 8.

    Returns:
    - filtered_data (np.ndarray): Filtered data in the time domain.
    """
    # Design the Butterworth bandpass filter using scipy.signal.butter
    sos = scipy.signal.butter(order, edges, 'bandpass', fs=sample_rate, output='sos')
    # Apply the filter to the input data using sosfiltfilt for zero-phase filtering
    filtered_data = scipy.signal.sosfiltfilt(sos, data)
    return filtered_data

def apply_notch_filter(data, notch_freq, fs, quality_factor=30):
    """
    Applies a notch filter to EEG data using the Infinite Impulse Response (IIR) method.

    Parameters:
        - data (np.ndarray): Input EEG data to be filtered.
        - notch_freq (float): Frequency of the notch filter.
        - fs (float): Sampling rate of the EEG data.
        - quality_factor (float, optional): Quality factor of the notch filter (default: 30).

    Returns:
        - filtered_data (np.ndarray): EEG data after applying the notch filter.
    """
    
    b, a = iirnotch(notch_freq, quality_factor, fs)
    y = lfilter(b, a, data)
    return y

def segmentation_train(data_montage, sampling_frequencies, target_sampling_rate, eeg_labels, segment_duration=30, index_record=None):
    """
    Segments EEG data based on seizure onset and offset timings or non-seizure conditions.

    Parameters:
        - data_montage (np.ndarray): EEG data matrix (montages x samples).
        - sampling_frequencies (float): Original sampling frequency of the EEG data.
        - target_sampling_rate (float): Desired sampling rate for segmentation.
        - eeg_labels (tuple): Tuple containing seizure_present, onset_index, and offset_index.
        - segment_duration (int, optional): Duration of each segment in seconds (default: 30).
        - index_record (int, optional): Index of the record (default: None).

    Returns:
        - segmented_data (list of np.ndarray): List of segmented EEG data.
        - labels (list of int): List of labels for each segment (0 for non-seizure, 1 for seizure).
    """
    
    original_sampling_frequency = sampling_frequencies
    segmented_data = []
    labels = []

    seizure_present, onset_index, offset_index = eeg_labels
    onset_index_original = int(onset_index * original_sampling_frequency)
    offset_index_original = int(offset_index * original_sampling_frequency)
    downsampling_factor = original_sampling_frequency / target_sampling_rate
    onset_index_downsampled = int(onset_index_original / downsampling_factor)
    offset_index_downsampled = int(offset_index_original / downsampling_factor)

    if seizure_present == 1:
        # Calculate the index range for 3 segments before and after seizure
        pre_seizure_start = max(0, onset_index_downsampled - 15 * segment_duration)
        post_seizure_end = min(data_montage.shape[1], offset_index_downsampled + 15 * segment_duration)

        # Adjust indices to ensure they are within the data range
        pre_seizure_start = max(pre_seizure_start, 0)
        post_seizure_end = min(post_seizure_end, data_montage.shape[1] - segment_duration)

        # Segment the data including pre and post seizure
        for i in range(pre_seizure_start, post_seizure_end, segment_duration):
            segment_start = i
            segment_end = i + segment_duration
            segment = data_montage[:, segment_start:segment_end]
            label = 1 if any(segment_start <= index <= segment_end for index in range(onset_index_downsampled, offset_index_downsampled)) else 0
            segmented_data.append(segment)
            labels.append(label)

    elif seizure_present == 0:
        # Segment only 15 consecutive segments in the middle
        middle_index = data_montage.shape[1] // 2
        start_index = max(0, middle_index - 30 * segment_duration)
        end_index = min(data_montage.shape[1], start_index + 60 * segment_duration)

        # Adjust indices to ensure they are within the data range
        start_index = max(start_index, 0)
        end_index = min(end_index, data_montage.shape[1] - segment_duration)

        for i in range(start_index, end_index, segment_duration):
            segment_start = i
            segment_end = i + segment_duration
            segment = data_montage[:, segment_start:segment_end]
            segmented_data.append(segment)
            labels.append(0)  # All labels are 0 as there is no seizure

    return segmented_data, labels


def segmentation(data_montage, sampling_frequencies, target_sampling_rate, eeg_labels, segment_duration=None, index_record=None):
    """
    Segments EEG data based on seizure onset and offset timings "without overlapping".

    Parameters:
    - data_montage (np.ndarray): NumPy array containing EEG signals (montages x samples).
    - sampling_frequencies (list): List of sampling frequencies for different EEG recordings.
    - target_sampling_rate (float): Desired sampling rate for segmentation.
    - eeg_labels (list): List of tuples (seizure_present, onset, offset) containing seizure event information.
    - segment_duration (float, optional): Duration of each segment in seconds (default: None).
    - index_record (int, optional): Index of the patient's record (default: None).

    Returns:
    - segmented_data (list of np.ndarray): List of segmented EEG data.
    - labels (list of int): List of labels for each segment (1 for seizure, 0 for non-seizure).
    """
    original_sampling_frequency = sampling_frequencies
    segmented_data = []  # List to store segmented data
    labels = []  # List to store corresponding labels
    
    seizure_present, onset_index, offset_index = eeg_labels
    onset_index_original = int(onset_index * original_sampling_frequency)
    offset_index_original = int(offset_index * original_sampling_frequency)
    downsampling_factor = original_sampling_frequency / target_sampling_rate
    onset_index_downsampled = int(onset_index_original / downsampling_factor)
    offset_index_downsampled = int(offset_index_original / downsampling_factor)
    
    
    if data_montage.shape[1] <= segment_duration:
        pass
    else:
        # Segment the data with labels
        for i in range(0, data_montage.shape[1] - segment_duration, segment_duration):
            segment_start = i
            segment_end = i + segment_duration
            segment = data_montage[:, segment_start:segment_end]
            label = 1 if any(segment_start <= index <= segment_end for index in range(onset_index_downsampled, offset_index_downsampled)) else 0
            segmented_data.append(segment)
            labels.append(label)

    return segmented_data, labels  # Return segmented data and labels


def data_preprocess(all_data, target_sampling_rate, segment_duration):
    """
    This function preprocesses EEG data by applying notch and bandpass filters, resampling it to the target sampling rate, and segmenting it into fixed-duration segments.

    Parameters:
    - all_data (object): Object containing EEG data, sampling frequencies, channels, and EEG labels.
    - target_sampling_rate (float): Desired sampling rate after resampling.
    - segment_duration (int): Duration of each segment for segmentation.

    Returns:
    - segmented_data_input_all (np.ndarray): Preprocessed and segmented EEG input data.
    - segmented_data_label_all (np.ndarray): Corresponding labels for the segmented data.
    """
    
    notch_freq = 50  # Frequency to be notched out
    lowcut = 0.5
    highcut = 70
    
    segmented_data_input_all = []  # List to store segmented input data
    segmented_data_label_all = []  # List to store segmented label data
    
    all_channels = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz']
    resampled_data_list = []
    
    for i, (data, sampling_f, channels, eeg_labels) in enumerate(zip(all_data.data[:], all_data.sampling_frequencies[:],all_data.channels[:] ,all_data.eeg_labels[:])):
        if sampling_f <= target_sampling_rate: 
            print("Sampling f is small")
            
        resampled_channels = [] 
        for i in range(len(channels)):
            channel_data = data[i, :]
            # Apply notch filter
            eeg_data_notched = apply_notch_filter(channel_data, notch_freq, sampling_f)
            # Apply bandpass filter
            eeg_data_bandpassed = bandpass(eeg_data_notched, [lowcut, highcut], sampling_f)
            # Resample
            resampled_channel_data = signal.resample(eeg_data_bandpassed, int(data.shape[1] * target_sampling_rate / sampling_f))
            
            resampled_channels.append(resampled_channel_data)
    
        resampled_channels = np.array(resampled_channels)
        
        montages,data_montage,montage_missing = get_3montages(channels, resampled_channels)
        #sorted_channels,data_montage = choose_6montages_bipolar(i)
        
        # Perform segmentation on EEG data
        segmented_data_input, segmented_data_label = segmentation(data_montage,
                                                                  sampling_f,target_sampling_rate, eeg_labels,
                                                                  segment_duration, index_record=i)
        
        # Reshape the segmented input data for concatenation
        segmented_data_input_reshape = np.array(segmented_data_input).reshape(len(segmented_data_input),
                                                                              segment_duration,
                                                                              len(data_montage))
        
        # Append segmented data to respective lists
        segmented_data_input_all.append(segmented_data_input_reshape)
        segmented_data_label_all.append(segmented_data_label)
                                                                            
    #Concatenate all segmented data along the first axis to create single arrays
    segmented_data_input_all = np.concatenate(segmented_data_input_all, axis=0)
    segmented_data_label_all = np.concatenate(segmented_data_label_all, axis=0)
        
    return segmented_data_input_all, segmented_data_label_all



def wavelet_features(data, montage_num):
    """
    This function applies the Discrete Wavelet Transform (DWT) with the function 'db4' level 5 to EEG data and calculates various statistics 
    on the wavelet coefficients, including energy, entropy, mean, standard deviation, zero crossing rate (zcr), and mean absolute value (mav).
    
    Parameters:
    - data (np.ndarray): EEG data for feature extraction.
    - montage_num (int): Montage number to be included in feature names.

    Returns:
    - features (dict): Dictionary containing wavelet features.
    
    """
    coeffs = wavedec(data, 'db4', level=5)
    
    features = {}
    for idx, coeff in enumerate(coeffs):
        # Skip D1 and D2 coefficients
        #if idx == 1 or idx == 2 :
         #   continue
        energy = np.sum(np.square(coeff))
        entropy = np.sum(np.square(coeff) * np.log(np.square(coeff) + 1e-10))  # Avoid log(0)
        mean = np.mean(coeff)
        std = np.std(coeff)
        zcr = len(find_peaks(coeff)[0]) / len(coeff)
        mav = np.mean(np.abs(coeff))
        
        coeff_label = 'cA' if idx == 0 else f'cD_{idx}'
        features[f'Montage_{montage_num}_{coeff_label}_Energy'] = energy
        features[f'Montage_{montage_num}_{coeff_label}_Entropy'] = entropy
        features[f'Montage_{montage_num}_{coeff_label}_Mean'] = mean
        features[f'Montage_{montage_num}_{coeff_label}_Std'] = std
        features[f'Montage_{montage_num}_{coeff_label}_zcr'] = zcr
        features[f'Montage_{montage_num}_{coeff_label}_mav'] = mav

    return features


def create_feature_dataframe(segmented_data: np.ndarray, segmented_label: np.ndarray)->pd.DataFrame:
    '''
    Create a feature dataframe using wavelet features from segmented data.

    Parameters:
    - segmented_data (list of lists): The segmented data containing channels and montages.
    - segmented_label (list): The corresponding labels for each segmented data.

    Returns:
    - pandas.DataFrame: A DataFrame containing extracted wavelet features and labels.
    '''
    data_entries= []
    for i, (item, label) in enumerate(zip(segmented_data, segmented_label)):
        features = {}

        item = np.transpose(item)
        for montage_idx, channel in enumerate(item, start=1):
            channel_features = wavelet_features(channel, montage_idx)
            features.update(channel_features)
    
        features['label'] = label
        data_entries.append(features)

    df_features = pd.DataFrame(data_entries)
    return df_features


def preprocess_all_subsets(subsets: list, target_sampling_rate: float, segment_duration: int)-> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Preprocess data from multiple subsets using notch filter, band-pass filter, and wavelet transform.
    Extract features in the wavelet domain for each subset. Take subsets 1-8 for training, subset 9 for validation, and subset 10 for testing
    Save preprocessed time data in .npy format in the folder 'time_data' and extracted wavelet features in .csv format in the folder 'wavelet'
    
    Parameters:
    - subsets (list): List of subset names or paths containing EEG data for preprocessing.
    - target_sampling_rate (float): Sampling rate for resampling all data
    - segment_duration (float): Duration for each segment

    Returns:
    - df_features_train_stacked (pd.DataFrame): Stacked DataFrame containing training set features.
    - df_features_val_stacked (pd.DataFrame): Stacked DataFrame containing validation set features.
    - df_features_test_stacked (pd.DataFrame): Stacked DataFrame containing test set features.

    Each subset is processed as follows:
    1. Load data from the subset using the load_folder function.
    2. Preprocess the loaded data using notch filter, band-pass filter, and segmentation.
    3. Saves the time data in .nyp files
    4. Extract features (energy, entropy, mean, std, zero-crossing rate, MAV) for each segmented data.
    5. Save the features in separate DataFrames for training, validation, and testing sets.
    6. Stack the features from all subsets into three main DataFrames for training, validation, and testing.

    """
    new_folder_path = 'time_data'
    # Use os.makedirs() to create the folder and its parent directories if they don't exist
    os.makedirs(new_folder_path, exist_ok=True)
    new_folder_path = 'wavelet'
    # Use os.makedirs() to create the folder and its parent directories if they don't exist
    os.makedirs(new_folder_path, exist_ok=True)
    
    # Iterate over each subset for preprocessing
    for i, subset in enumerate(subsets):
        train, test, val = load_folder(subset)

        # Preprocess the data (notch filter, band-pass filter, segmentation)
        segmented_data_train, segmented_label_train = data_preprocess(train, target_sampling_rate, segment_duration)
        segmented_data_val, segmented_label_val = data_preprocess(val, target_sampling_rate, segment_duration)
        segmented_data_test, segmented_label_test = data_preprocess(test, target_sampling_rate, segment_duration)

        concatenated_array_X = np.concatenate([segmented_data_train, segmented_data_val, segmented_data_test], axis=0)
        concatenated_array_y = np.concatenate([segmented_label_train, segmented_label_val, segmented_label_test], axis=0)
        np.save(f'time_data/subset{i}_X.npy', concatenated_array_X)
        np.save(f'time_data/subset{i}_y.npy', concatenated_array_y)
    
    # List of file names for y_train
    file_names_y_train = [f'time_data/subset{i}_y.npy' for i in range(1, 9)]

    # Load and concatenate arrays for y_train
    y_train = np.concatenate([np.load(file_name) for file_name in file_names_y_train], axis=0)

    y_val = np.load('time_data/subset9_y.npy')
    y_test = np.load('time_data/subset9_y.npy')
    
    # List of file names for y_train
    file_names_x_train = [f'time_data/subset{i}_X.npy' for i in range(1, 9)]

    # Load and concatenate arrays for y_train
    X_train = np.concatenate([np.load(file_name) for file_name in file_names_x_train], axis=0)

    X_val = np.load('time_data/subset9_X.npy')
    X_test = np.load('time_data/subset10_X.npy')
    
    # create wavelet_train.csv
    data_entries = []
    for i, (item, label) in enumerate(zip(X_train, y_train)):
        features = {}
    
        item = np.transpose(item)
    
        for montage_idx, channel in enumerate(item, start=1):
            channel_features = wavelet_features(channel, montage_idx)
            features.update(channel_features)
    
        features['label'] = label
        data_entries.append(features)
        

    df= pd.DataFrame(data_entries) 
    df.to_csv(f'wavelet/wavelet_train.csv', index=False)
    
    # create wavelet_val.csv
    data_entries = []
    for i, (item, label) in enumerate(zip(X_val, y_val)):
        features = {}
    
        item = np.transpose(item)
    
        for montage_idx, channel in enumerate(item, start=1):
            channel_features = wavelet_features(channel, montage_idx)
            features.update(channel_features)
    
        features['label'] = label
        data_entries.append(features)
        

    df= pd.DataFrame(data_entries) 
    df.to_csv(f'wavelet/wavelet_val.csv', index=False)
    
    # create wavelet_test.csv
    data_entries = []
    for i, (item, label) in enumerate(zip(X_test, y_test)):
        features = {}
    
        item = np.transpose(item)
    
        for montage_idx, channel in enumerate(item, start=1):
            channel_features = wavelet_features(channel, montage_idx)
            features.update(channel_features)
    
        features['label'] = label
        data_entries.append(features)
        

    df= pd.DataFrame(data_entries) 
    df.to_csv(f'wavelet/wavelet_test.csv', index=False)
    
    
    df_train = pd.read_csv('wavelet/wavelet_train.csv', index = False)
    df_val = pd.read_csv('wavelet/wavelet_val.csv', index = False)
    df_test = pd.read_csv('wavelet/wavelet_test.csv', index = False)
    
    return df_train, df_val, df_test


    