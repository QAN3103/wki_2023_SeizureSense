import csv
import matplotlib.pyplot as plt
import numpy as np
import os
from wettbewerb import load_references, get_3montages, get_6montages
import mne
from scipy import signal as sig
import ruptures as rpt
import json


def segmentation(data_montage, channels, sampling_frequencies, segment_duration=None, index_record=None):
    """
    Segments EEG data based on seizure onset and offset timings "without overlapping".

    Parameters:
    - data_montage: NumPy array containing EEG signals (montages x samples)
    - channels: List of channel information corresponding to EEG data
    - sampling_frequencies: List of sampling frequencies for different EEG recordings
    - segment_duration: Duration of each segment in seconds (default: 30 seconds)
    - index_record: Index of the patient's record

    Returns:
    - segmented_data: List of NumPy arrays containing segmented EEG data
    """
    sampling_frequencies = sampling_frequencies[index_record]
    segmented_data = []  # List to store segmented data

    
    # Segment the data 
    for i in range(0, data_montage.shape[1] - segment_duration, segment_duration):
        segment_start = i  # Start index of the segment
        segment_end = i + segment_duration  # End index of the segment

        segment = data_montage[:, segment_start:segment_end]  # Extract the segment from EEG data
        
        #label = 1 if onset_index <= segment_start <= offset_index and seizure_present == 1 else 0  # Set label based on seizure event
        segmented_data.append(segment)  # Append segmented data
    
    # Print information about the segmentation
    #print(f"Patient index {index_record}")
    #print(f"Number of segments: {len(segmented_data)}")
    #print(f"Segments with seizure: {labels.count(1)}")
    #print(f"Length of the record in sec: {data_montage.shape[1]}")
    #print(f"Label: {eeg_labels[index_record]}")

    return segmented_data  # Return segmented data and labels

import numpy as np

def segment_all_data(data, channels, sampling_frequencies, segment_duration):
    """
    Segments EEG data for all patients.

    Args:
    - data: List of EEG data for different patients.
    - channels: List of channel information.
    - sampling_frequencies: List of sampling frequencies.
    - segment_duration: Duration for segmentation.

    Returns:
    - segmented_data_input_all_array: Numpy array of segmented input data.
    """
    
    segmented_data_input_all = []  # List to store segmented input data

    # Iterate through the data and perform segmentation
    for i in range(len(data)):
        # Obtain necessary data for segmentation
        new_montage, data_montage, is_missing = get_6montages(channels[i], data[i])
        # Perform segmentation on EEG data
        segmented_data_input = segmentation(data_montage, channels, sampling_frequencies, 
                                                                segment_duration=segment_duration, index_record=i)
        # Reshape the segmented input data for concatenation
        segmented_data_input_reshape = np.array(segmented_data_input).reshape(len(segmented_data_input),
                                                                              len(data_montage),
                                                                              segment_duration)
        # Append segmented data to respective lists
        segmented_data_input_all.append(segmented_data_input_reshape)
        
       
    # Concatenate all segmented data along the first axis to create single arrays
    segmented_data_input_all = np.concatenate(segmented_data_input_all, axis=0)

    return segmented_data_input_all


def count_segment_1(segmented_data_label_all_array):
    """
    Counts the total number of elements and occurrences of '1' in a 2D array of lists.

    Parameters:
    - segmented_data_label_all_array (list): A 2D array containing sublists.

    Returns:
    - None
        Prints the total number of elements in the array and the occurrences of '1'.
    """
    total_elements = 0
    count_of_ones = 0

    # Iterate through the lists and count all elements and occurrences of '1'
    for sublist in segmented_data_label_all_array:
        total_elements += len(sublist)
        count_of_ones += np.count_nonzero(sublist)
        
    print(f"Total segments in the array: {total_elements}")
    print(f"Occurrences of '1': {count_of_ones}")

    return None

def count_segment(segmented_data_label_all_array):
    # Count occurrences of '1' in the entire array
    count_of_ones = np.count_nonzero(segmented_data_label_all_array == 1)

    # Get the total number of elements in the array
    total_elements = segmented_data_label_all_array.size

    print(f"Total segments in the array: {total_elements}")
    print(f"Count of '1's in the array: {count_of_ones}")


def count_patients(patient_ids):
    unique_patients = set()

    for patient_id in patient_ids:
        unique_id = patient_id.split('_')[0]
        unique_patients.add(unique_id)

    return len(unique_patients)
