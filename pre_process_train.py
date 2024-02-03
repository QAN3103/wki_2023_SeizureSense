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
from Channel_Detection import choose_6montages_bipolar
from scipy.signal import find_peaks
from pywt import wavedec
import os

def bandpass(data: np.ndarray, edges: List[float], sample_rate: float, order: int = 8 ) -> np.ndarray:
    sos = scipy.signal.butter(order, edges, 'bandpass', fs=sample_rate, output='sos')
    filtered_data = scipy.signal.sosfiltfilt(sos, data)
    return filtered_data

def notch_filter(data: np.ndarray, notch_freq: float, sample_rate: float, quality_factor: float = 30) -> np.ndarray:
    """
    Apply a notch (band-stop) filter to the data.

    Parameters:
    data (np.ndarray): The input signal.
    notch_freq (float): The center frequency to be notched out.
    sample_rate (float): The sampling rate of the data.
    quality_factor (float): Quality factor for the notch filter, which determines the bandwidth around the notch_freq.

    Returns:
    np.ndarray: The filtered data.
    """
    sos = scipy.signal.iirnotch(w0=notch_freq, Q=quality_factor, fs=sample_rate)
    filtered_data = scipy.signal.sosfiltfilt(sos, data)
    return filtered_data

def apply_notch_filter(data, notch_freq, fs, quality_factor=30):
    b, a = iirnotch(notch_freq, quality_factor, fs)
    y = lfilter(b, a, data)
    return y


def segmentation_train(data_montage, sampling_frequencies, target_sampling_rate, eeg_labels, segment_duration=30, index_record=None):
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


def segmentation_test(data_montage, sampling_frequencies, target_sampling_rate, eeg_labels, segment_duration=None, index_record=None):
    """
    Segments EEG data based on seizure onset and offset timings "without overlapping".

    Parameters:
    - data_montage: NumPy array containing EEG signals (montages x samples)
    - channels: List of channel information corresponding to EEG data
    - sampling_frequencies: List of sampling frequencies for different EEG recordings
    - eeg_labels: List of tuples (seizure_present, onset, offset) containing seizure event information
    - segment_duration: Duration of each segment in seconds (default: 30 seconds)
    - index_record: Index of the patient's record

    Returns:
    - segmented_data: List of NumPy arrays containing segmented EEG data
    - labels: List of labels where 1 denotes a seizure segment and 0 denotes a non-seizure segment
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

    # Print information about the segmentation
    # print(f"Patient index {index_record}")
    # print(f"Number  of segments: {len(segmented_data)}")
    # print(f"Segments with seizure: {labels.count(1)}")
    # print(f"Length of the record in sec: {data_montage.shape[1]}")
    # print(f"Label: {eeg_labels[index_record]}")

    return segmented_data, labels  # Return segmented data and labels


def data_preprocess_test(all_data, target_sampling_rate, segment_duration):
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
        segmented_data_input, segmented_data_label = segmentation_test(data_montage,
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


def data_preprocess(all_data, target_sampling_rate, segment_duration):
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
        segmented_data_input, segmented_data_label = segmentation_train(data_montage,
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
    coeffs = wavedec(data, 'db4', level=5)
    if np.any(np.isnan(coeffs)) or np.any(np.isinf(coeffs)):
        print('One of the coefficients is NaN or infinity')
    
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
        segmented_data (list of lists): The segmented data containing channels and montages.
        segmented_label (list): The corresponding labels for each segmented data.

    Returns:
        pandas.DataFrame: A DataFrame containing extracted wavelet features and labels.
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
    Extract features in the wavelet domain for each subset.
    
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
    3. Extract features (energy, entropy, mean, std, zero-crossing rate, MAV) for each segmented data.
    4. Save the features in separate DataFrames for training, validation, and testing sets.
    5. Stack the features from all subsets into three main DataFrames for training, validation, and testing.

    """
    # Initialize empty DataFrames to store features for training, validation, and testing sets
    df_features_train_stacked = pd.DataFrame()
    df_features_val_stacked = pd.DataFrame()
    df_features_test_stacked = pd.DataFrame()

    # Iterate over each subset for preprocessing
    for i, subset in enumerate(subsets):
        
        # Load data from the subset
        train, test, val = load_folder(subset)
        
        # Preprocess the data (notch filter, band-pass filter, segmentation)
        segmented_data_train, segmented_label_train = data_preprocess(train, target_sampling_rate, segment_duration)
        segmented_data_val, segmented_label_val = data_preprocess(val, target_sampling_rate, segment_duration)
        segmented_data_test, segmented_label_test = data_preprocess(test, target_sampling_rate, segment_duration)
    
        # Extract features and create DataFrames
        df_features_train = create_feature_dataframe(segmented_data_train, segmented_label_train)
        df_features_val = create_feature_dataframe(segmented_data_val, segmented_label_val)
        df_features_test = create_feature_dataframe(segmented_data_test, segmented_label_test)
    
        # Concatenate features to the stacked DataFrames
        df_features_train_stacked = pd.concat([df_features_train_stacked, df_features_train], axis = 0, ignore_index=True)
        df_features_val_stacked = pd.concat([df_features_val_stacked, df_features_val], axis = 0, ignore_index=True)
        df_features_test_stacked = pd.concat([df_features_test_stacked, df_features_test], axis = 0, ignore_index=True)

    return df_features_train_stacked, df_features_val_stacked, df_features_test_stacked


def create_wavelet_csv(df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame, destination_folder: str)-> None:
    """
    Create CSV files for wavelet-transformed features from training, validation, and test sets.
    
    Parameters:
    - df_train (pd.DataFrame): DataFrame containing wavelet-transformed features for the training set.
    - df_val (pd.DataFrame): DataFrame containing wavelet-transformed features for the validation set.
    - df_test (pd.DataFrame): DataFrame containing wavelet-transformed features for the test set.
    - destination_folder (str): Path to the folder where the CSV files will be saved.

    This function creates a folder if it does not exist, deletes the contents of the folder if it already exists,
    and then saves three separate CSV files for the training, validation, and test sets in the specified destination folder.
    
    """
    # Create the output folder if it does not exist
    os.makedirs(destination_folder, exist_ok=True)
    
    # delete folder with previous subsets
    delete_folder_contents(destination_folder)
    
    # create csv for train
    name_file = os.path.join(destination_folder, 'train_wavelet.csv')
    df_train.to_csv(name_file, index = False)
    print(f'\n The file has been saved as {name_file} in the folder {destination_folder}')
    
    # create csv for val
    name_file = os.path.join(destination_folder, 'val_wavelet.csv')
    df_val.to_csv(name_file, index = False)
    print(f'\n The file has been saved as {name_file} in the folder {destination_folder}')
    
    # create csv for test
    name_file = os.path.join(destination_folder, 'test_wavelet.csv')
    df_test.to_csv(name_file, index = False)
    print(f'\n The file has been saved as {name_file} in the folder {destination_folder}')