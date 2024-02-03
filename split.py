"""
EEG Data Processing and Analysis Module

This module provides functions and classes for processing and analyzing EEG (Electroencephalogram) data. It includes
functionalities for splitting datasets into subsets, dividing data into train, test, and validation sets, loading EEG data from
files, and more.

Authors:
- Jana Taube
- Ayman Kabawa
- Quỳnh Anh Nguyễn
- Dirk Schweickard
- Maurice Rohr

Functions:
- create_subsets: Creates subsets of data based on unique patient identifiers.
- write_csv: Writes a DataFrame to a CSV file.
- split_data: Splits data into train, test, and validation sets.
- equal_split: Attempts to split data into sets with roughly equal seizure distributions.
- load_references_self: Loads EEG data from specified folder and reference file.
- load_folder: Loads train, test, and val data from a specified folder.
- delete_folder_contents: Deletes the contents of a specified folder.
- load_folder_as_one: Combines train, test, and val data from a folder into a single dataset.

Classes:
- EEGDataset: Represents an EEG dataset.
- EEGStruct: Represents the structure of EEG data.

Usage:
This module is intended to be used as part of an EEG data analysis pipeline, facilitating the preprocessing, loading, and
splitting of EEG datasets for further analysis or machine learning tasks.

Requirements:
- Python 3.x
- Libraries: Scikit-learn,Pandas, NumPy, Scipy

Please note: This module assumes the presence of specific file structures and formats as part of its operation. Ensure that
your data is correctly formatted and that necessary files are in place before using these utilities.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from typing import List, Tuple, Dict, Any
import csv
import scipy.io as sio
import shutil


def create_subsets(data: pd.DataFrame, number_subsets: int) -> List:
    """
    Create subsets of CSV data based on unique patient identifiers.

    Parameters:
    - data (pd.DataFrame): The input data in DataFrame format.
    - number_subsets (int): The number of subsets to create.

    Returns:
    - subsets: List of subsets
    """
     
    # Step 1: Extract unique patients identified by the letters before the first "_"
    patients = data[0].apply(lambda x: x.split('_')[0]).unique()
    
    # Calculate the total number of entries for each patient
    patient_entry_counts = data.groupby(data[0].apply(lambda x: x.split('_')[0])).size()
    
    # Step 2: Sort patients based on the number of entries
    sorted_patients = patient_entry_counts.sort_values(ascending=False).index
    
    # Step 3: Distribute patients sequentially to subsets
    subsets_patients = [sorted_patients[i::number_subsets] for i in range(number_subsets)]
    
    # Step 4: Create and save subsets
    subsets = []
    for i, subset_patients in enumerate(subsets_patients):
        # Filter data based on the selected patient sets
        subset = data[data[0].apply(lambda x: x.split('_')[0]).isin(subset_patients)]
        subsets.append(subset)
        print(f'Subset_{i+1} has {len(subset)} entries.')
    print(f'\n \n \n')
    
    return subsets


def write_csv(df: pd.DataFrame, folder: str, name: str)->None:
    '''
     Write a pandas DataFrame to a CSV file.

    Parameters:
    - df (pd.DataFrame): The DataFrame to be saved.
    - folder (str): The folder where the CSV file will be saved.
    - name (str): The name of the CSV file (including the ".csv" extension).

    Returns:
    None
    '''
    path = os.path.join(folder, name)
    df.to_csv(path, index = False, header = False)
    print(f'\n The file has been saved as {name} in the folder {folder}')
    

def split_data(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray]:
    """
    Splits EEG data into train, test, and validation sets based on unique patients.

    Parameters:
    - data (pandas.DataFrame): EEG data with columns including 'patient_id', 'session', 'x' (seizure indicator), and other EEG data columns.

    Returns:
    - train_data (pandas.DataFrame): Training set based on a split of unique patients.
    - test_data (pandas.DataFrame): Testing set based on a split of unique patients.
    - val_data (pandas.DataFrame): Validation set based on a split of unique patients.
    - split_percentage (numpy.ndarray): An array containing the percentage of seizures in each set (train, test, val).
    - patients_split (list): A list containing the split of patients into train, test, and validation sets.

    """
    
    # Step 1: Extract unique patients identified by the letters before the first "_"
    patients = data[0].apply(lambda x: x.split('_')[0]).unique()
    
    # Step 2: Split patients into train, test, and validation sets
    train_patients, test_and_val_patients = train_test_split(patients, test_size=0.4, stratify=None, shuffle = True)
    test_patients, val_patients = train_test_split(test_and_val_patients, test_size=0.5, stratify=None, shuffle = True )
    patients_split = [train_patients, test_patients, val_patients]
    
    # Step 3: Filter data based on the selected patient sets
    train_data = data[data[0].apply(lambda x: x.split('_')[0]).isin(train_patients)]
    test_data = data[data[0].apply(lambda x: x.split('_')[0]).isin(test_patients)]
    val_data = data[data[0].apply(lambda x: x.split('_')[0]).isin(val_patients)]
    
    # Number of seizures in whole set:
    num_seizures = (data[1] == 1).sum()
    # Number of seizures in each subset
    seizures_train = (train_data[1]==1).sum()
    seizures_test = (test_data[1]==1).sum()
    seizures_val = (val_data[1]==1).sum()

    # Percentages in each subset:
    train_perc = (seizures_train/num_seizures)
    test_perc = (seizures_test/num_seizures)
    val_perc = (seizures_val/num_seizures)
    split_percentage = [train_perc, test_perc, val_perc]
    split_percentage  = np.array([round(x, 2) for x in split_percentage]) # formated
    
    return train_data, test_data, val_data, split_percentage, patients_split


def equal_split(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list, np.ndarray]:
    """
    Splits EEG data into train, test, and validation sets until the percentage of seizures is roughly equally distributed,
    with the train set having the highest percentage of seizures.

    Parameters:
    - data (pandas.DataFrame): EEG data with columns including 'patient_id', 'session', 'x' (seizure indicator), and other EEG data columns.

    Returns:
    - train_data (pandas.DataFrame): Training set with roughly equal distribution of seizures.
    - test_data (pandas.DataFrame): Testing set.
    - val_data (pandas.DataFrame): Validation set.
    - split_percentage (list): A list containing the percentage of seizures in each set.
    - patients_split (numpy.ndarray): An array representing the distribution of patients in each set.

    """
        
    train_data, test_data, val_data, split_percentage, patients_split = split_data(data)
    counter = 1
    threshold = 0.05
    increase_threshold = 0
         
    # Check if the percentage of seizures in each set is roughly equally sized and train having the higest percentage
    while (np.std(split_percentage) > threshold) or (split_percentage[0]< split_percentage[1]) or (split_percentage[0]< split_percentage[2]):
        train_data, test_data, val_data, split_percentage, patients_split = split_data(data)
        counter= counter + 1
        if counter > 5:
            threshold = threshold + 0.05
            counter = 0
            increase_threshold += 1
        
    print(f'Split wurde in {counter} Schritten durgeführt. \n Dabei wurde der Threshold {increase_threshold} Mal um 0.05 erhöht')
    
    return train_data, test_data, val_data, split_percentage, patients_split



def load_references_self(folder: str = '../shared_data/training_mini', reference_file: str = '../shared_data/training_mini/REFERENCE.csv') -> Tuple[List[str], List[List[str]],
                                                          List[np.ndarray],  List[float],
                                                          List[str], List[Tuple[bool,float,float]]]:
    """
    Load EEG dataset information from a specified folder and reference file.

    This function reads EEG data files and their metadata, organizing them into structured lists for further processing or analysis.
    The EEG data and metadata include identifiers, channel configurations, raw data arrays, sampling frequencies, reference systems,
    and labels indicating seizure presence along with seizure start and end times.

    Parameters:
    - folder (str, optional): The directory path where the EEG data files are stored. Defaults to '../training'.
    - reference_file (str, optional): The path to the CSV file containing metadata for the EEG recordings. This file should list the identifiers, seizure presence,
    and seizure timing information. Defaults to '../training/REFERENCE.csv'.

    Returns:
    - ids (List[str]): A list containing the identifiers for each EEG recording.
    - channels (List[List[str]]): A list where each element is a list of channel names available in the corresponding EEG recording.
    - data (List[np.ndarray]): A list of NumPy arrays, each containing the raw EEG data for a recording.
    - sampling_frequencies (List[float]): A list of sampling frequencies (in Hz) for each EEG recording.
    - reference_systems (List[str]): A list of reference system identifiers used in the EEG recordings (e.g., "LE", "AR", "Sz").   
    - eeg_labels (List[Tuple[bool, float, float]]): A list of tuples for each EEG recording, where each tuple contains a boolean indicating the presence of a seizure,
    the start time of the seizure (if any), and the end time of the seizure (if any).
    """
    
    # initialize Lists ids, channels, data, sampling_frequencies, refernece_systems, and eeg_labels
    ids: List[str] = []
    channels: List[List[str]] = []
    data: List[np.ndarray] = []
    sampling_frequencies: List[float] = []
    reference_systems: List[str] = []
    eeg_labels: List[Tuple[bool,float,float]] = []
    
    # create a dataset from a folder and populate lists with data
    dataset = EEGDataset(folder, reference_file)
    for item in dataset:
        ids.append(item[0])
        channels.append(item[1])
        data.append(item[2])
        sampling_frequencies.append(item[3])
        reference_systems.append(item[4])
        eeg_labels.append(item[5])
        
    # show how much data was loaded
    print("{}\t Dateien wurden geladen.".format(len(ids)))
    return ids, channels, data, sampling_frequencies, reference_systems, eeg_labels


class EEGDataset:
    def __init__(self,folder:str, reference_file:str) -> None:
        """
        This class represents an EEG dataset.
        Main structure from Dirk Scheickard and Maurice Rohr. Changes for own purpose. 

        Usage:
            Create a new dataset (without loading all data) with
            dataset = EEGDataset("../training/")
            len(dataset) # returns the dataset size
            dataset[0] # returns the first element from the dataset consisting of (id, channels, data, sampling_frequency, reference_system, eeg_label)
            it = iter(dataset) # returns an iterator on the dataset,
            next(it) # returns the next element until all data is fetched once
            for item in dataset: # iterates over the entire dataset once
            (id, channels, data, sampling_frequency, reference_system, eeg_label) = item
        

        Args:
        folder (str): Folder where the dataset consisting of .mat files and a REFERENCE.csv file is located.
        
        """
        assert isinstance(folder, str), "Parameter folder muss ein string sein aber {} gegeben".format(type(folder))
        assert os.path.exists(folder), 'Parameter folder existiert nicht!'
        assert os.path.exists(reference_file), 'Reference datei existiert nicht!'
        # Initialize lists for channel and id
        self._folder = folder
        self._ids: List[str] = []
        self._eeg_labels: List[Tuple[bool,float,float]] = []
        # load reference data
        with open(reference_file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            
            # iterate over each row
            for row in csv_reader:
                self._ids.append(row[0])
                self._eeg_labels.append((int(row[1]),float(row[2]),float(row[3])))
    
    def __len__(self):
        return len(self._ids)
    
    def __getitem__(self,idx) -> Tuple[str, List[str],
                                    np.ndarray,  float,
                                    str, Tuple[bool,float,float]]:
        #load Matlab-file
        eeg_data = sio.loadmat(os.path.join(self._folder, self._ids[idx] + '.mat'),simplify_cells=True)
        ch_names = eeg_data.get('channels')
        channels = [x.strip(' ') for x in ch_names] 
        data = eeg_data.get('data')
        sampling_frequency = eeg_data.get('fs')
        reference_system = eeg_data.get('reference_system')
        return (self._ids[idx],channels,data,sampling_frequency,reference_system,self._eeg_labels[idx])
    
    def get_labels(self):
        return self._eeg_labels
    
        
class EEGStruct:
    """
    A class representing the structure of EEG (Electroencephalogram) data.

    Attributes:
    - ids (List): List of identifiers for each data sample.
    - channels (List): List of EEG channels used in the data.
    - data (List): List of EEG data samples.
    - sampling_frequencies (List): List of corresponding sampling frequencies for each data sample.
    - reference_systems (List): List of reference systems used in the EEG recordings.
    - eeg_labels (List): List of labels associated with the EEG data (seizure/no-seizure).
    """
    def __init__(self, ids, channels, data, sampling_frequencies, reference_systems, eeg_labels):
        self.ids = ids
        self.channels = channels
        self.data = data
        self.sampling_frequencies = sampling_frequencies
        self.reference_systems = reference_systems
        self.eeg_labels = eeg_labels



        
def split_file(reference: str, number_subsets: int, folder: str = 'split/')-> List:
    """
   Splits a reference CSV file into a specified number of subsets and further divides each subset into training, testing, and validation data.
    
    Parameters:
    - reference (str): Path to the reference CSV file (e.g., 'shared_data/training/REFERENCE.csv').
    - number_subsets (int): Number of subsets to create from the reference data.
    - folder (str): Path to the folder where the subsets will be saved. ( default = split/ ) 

    Returns:
    paths_subsets (List): contains the paths of the subsets
    
    Example: paths_subsets = split_file('shared_data/training/REFERENCE.csv', 10, 'split/')
    """
    
    # List for names of the subsets
    paths_subsets = []
    
    # Create the output folder if it does not exist
    os.makedirs(folder, exist_ok=True)
    
    # delete folder with previous subsets
    delete_folder_contents(folder)
        
    # create datafram out of reference.csv file
    data = pd.read_csv(reference, sep = ',', header = None)
    
    # Step 1: Split the csv into n sets using the function "create_subsets"
    subsets = create_subsets(data, number_subsets)
    
    # Step 2: Split a subset into train, test and val data
    for i, subset in enumerate(subsets):
        # split subset into train, test and val data
        train_data, test_data, val_data, split_percentage, patients_split = equal_split(subsets[i])
        print(f"Percentage of seizures in Subset {i+1}: {split_percentage}")
        
        # Create a folder for the current subset
        subset_folder = os.path.join(folder, f'subset{i+1}')
        os.makedirs(subset_folder, exist_ok=True)
        paths_subsets.append(subset_folder)
       
        write_csv(train_data, subset_folder, 'train.csv')
        write_csv(test_data, subset_folder, 'test.csv')
        write_csv(val_data, subset_folder, 'val.csv')
    
    return paths_subsets


def load_folder(folder: str)->Tuple[EEGStruct, EEGStruct, EEGStruct]:
    """
    Load EEG data from a specified folder and return three structures (train, test, and val). 
    The data is assumed to be stored in three separate CSV files: 'train.csv', 'test.csv', and 'val.csv'
    within the specified folder.

    Parameters:
    - folder (str): The path to the folder containing the EEG data files.

    Returns:
    Tuple[EEGStruct, EEGStruct, EEGStruct]: A tuple of three EEGStruct instances representing the
    training, testing, and validation datasets, respectively.

    Each EEGStruct instance contains the following attributes:
    - ids: List of identifiers for each data sample.
    - channels: List of EEG channels used in the data.
    - data: List of EEG data samples.
    - sampling_frequencies: List of corresponding sampling frequencies for each data sample.
    - reference_systems: List of reference systems used in the EEG recordings.
    - eeg_labels: List of labels associated with the EEG data.

    Example:
    ```
    folder_path = 'split/subset1'
    train, test, val = load_folder(folder_path)
    ```

    Note:
    - The actual loading of EEG data is performed by the function 'load_references_self,' 
    - The EEG data is structured using the 'EEGStruct' class
    """

    path_train = os.path.join(folder, 'train.csv')
    path_test = os.path.join(folder, 'test.csv') 
    path_val = os.path.join(folder, 'val.csv')
    folder_data = 'shared_data/training'
    
    # TRAIN DATA:
    ids_train, channels_train, data_train, sampling_frequencies_train, reference_systems_train, eeg_labels_train = load_references_self(folder_data, path_train)
    train = EEGStruct(ids_train, channels_train, data_train, sampling_frequencies_train, reference_systems_train, eeg_labels_train)
    
    # TEST DATA:
    ids_test, channels_test, data_test, sampling_frequencies_test, reference_systems_test, eeg_labels_test = load_references_self(folder_data, path_test)
    test = EEGStruct(ids_test, channels_test, data_test, sampling_frequencies_test, reference_systems_test, eeg_labels_test)
    
    # VAL DATA: 
    ids_val, channels_val, data_val, sampling_frequencies_val, reference_systems_val, eeg_labels_val = load_references_self(folder_data, path_val)
    val = EEGStruct(ids_val, channels_val, data_val, sampling_frequencies_val, reference_systems_val, eeg_labels_val)
    
    return train, test, val



def delete_folder_contents(folder_path: str)-> None:
    '''
    Checks if a folder exists and, if it does, deletes the contents, including files and subfolders, within it.
    
    Parameters:
    - folder_path (str): Path to a folder that wants to be deleted.
    
    Returns:
    - None
    
    '''
    try:
        # Check if the folder exists
        if os.path.exists(folder_path):
            # Iterate over the files and subfolders in the folder
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)

                # Check if it's a file or a subfolder
                if os.path.isfile(file_path):
                    # Delete the file
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    # Delete the subfolder and its contents recursively
                    shutil.rmtree(file_path)

            print(f"Contents of {folder_path} deleted successfully.")
        else:
            print(f"The folder {folder_path} does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")

        
def load_folder_as_one(folder: str)->EEGStruct:
    """
    Load EEG data from a specified folder and return one EEGStruct. 
    The data is assumed to be stored in three separate CSV files: 'train.csv', 'test.csv', and 'val.csv' within the specified folder.

    Parameters:
    - folder (str): The path to the folder containing the EEG data files.

    Returns:
    - subset (EEGStruct): A EEGStruct instances representing the training, testing, and validation dataset in the specified folder.

    EEGStruct instance contains the following attributes:
    - ids: List of identifiers for each data sample.
    - channels: List of EEG channels used in the data.
    - data: List of EEG data samples.
    - sampling_frequencies: List of corresponding sampling frequencies for each data sample.
    - reference_systems: List of reference systems used in the EEG recordings.
    - eeg_labels: List of labels associated with the EEG data.

    Example:
    ```
    folder_path = 'split/subset1'
    train, test, val = load_folder(folder_path)
    ```

    Note:
    - The actual loading of EEG data is performed by the function 'load_references_self,' 
    - The EEG data is structured using the 'EEGStruct' class
    """
    path_train = os.path.join(folder, 'train.csv')
    path_test = os.path.join(folder, 'test.csv') 
    path_val = os.path.join(folder, 'val.csv')
    train_subset= pd.read_csv(path_train, header = None)
    test_subset= pd.read_csv(path_test, header = None)
    val_subset= pd.read_csv(path_val, header = None)
    subset_stacked = pd.concat([train_subset, test_subset, val_subset], axis = 0,ignore_index=True)
    path_subset = os.path.join(folder, 'REFERENCE.csv')
    subset_stacked.to_csv(path_subset, index = False, header = None)
    folder_data = 'shared_data/training'
    
    
    # DATA:
    ids, channels, data, sampling_frequencies, reference_systems, eeg_labels = load_references_self(folder_data, path_subset)
    subset = EEGStruct(ids, channels, data, sampling_frequencies, reference_systems, eeg_labels)
    
    return subset