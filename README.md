# Wettbewerb KI in der Medizin - Team SeizureSense
Project by Ayman Kabawa, Quỳnh Anh Nguyễn and Jana Taube

## Short Overview
This project contains the necessary code to train an artificial neuronal network (ANN) for detecting epileptic seizures in EEG data. The data is first preprocessed using a notch filter, bandpass filter, resampled, combined to a montage, and segmented. Then features are extraced from the wavelet domain. 

!Important: The code in the file train.py needs a large amout of data. For the code to run properly the function load_data() can be run to create the required .csv files. Alternatively the files can be uploaded and have to be stored in a folder called 'wavelet'.


## Road Map
The code consists of two main steps. 1) Creating the ANN model and 2) predicting seizures using the model

### 1) Creating the ANN model
#### 1. Split data into smaller subsets (split.py)
The primary functions within the 'split.py' file revolve around two key functions: split_file(reference, number_subsets, folder) and load_folder(folder). 
##### split_file(reference, number_subsets, folder)
Given the size of the training data, loading it in its entirety becomes impossible. Consequently, the split_file function strategically divides the original REFERENCE.csv file into subsets, further segmenting each subset into train.csv, val.csv, and test.csv. This division follows specific conditions:
- To avoid data leakage, all sessions containing to a single patient are confined within the same subset, without any dispersion across the training, validation, and test sets.
- The percentage of sessions featuring seizures is equitably distributed among the train, validation, and test sets in each subset.
- If an equal percentage is not possible, the train set should have the highest percentage of seizures.
##### load_folder(folder)
Load EEG data from a specified folder and return three structures (train, test, and val). Each EEGStruct instance contains the following attributes:
- List of identifiers for each data sample.
- List of EEG channels used in the data.
- List of EEG data samples.
- List of corresponding sampling frequencies for each data sample.
- List of reference systems used in the EEG recordings.
- List of labels associated with the EEG data.

#### 2. Preprocess the data (pre_process.py)
The 'pre_process.py' file loads the data, performs pre processing, and saves the extracted features for training the neuronal network as .csv files.

##### preprocess_all_subsets(subsets, target_sampling_rate, segment_duration)
The preprocessing is done for each subset seperately and combined in the end. The data is processed through a notch filter (50 Hz), a band-pass filter (0.5 Hz, 70 Hz), and resampled to have the same samling frequency of 173.61 Hz. Afterwards the get_3montages() function calculates the 3 montages (Fp1-F3, Fp2-F4, C3-P3) from the given channels to the same reference electrode, before all data is segmented into 5 second epoches. Each montage is then transformed into the wavelet domain for feature extraction. The following features are extracted for each coefficient from the wavelet domain: 
- energy, 
- entropy, 
- mean, 
- std, 
- zero-crossing rate, 
- MAV
The features for all three montages are saved in seperate DataFrames for training, validation, and testing.

##### create_wavelet_csv(features_train, features_val, features_test, destination_folder)
Once the features are extracted, the DataFrames are saved using the method create_wavelet_csv(). Later the features can easily be reloaded without having to perform the whole pre processing again, which is very time and storage consuming.

#### 3. Train the ANN (train.py)
For training the ANN the data first has to be oversampled, because the train set is highly imbalanced. Afterwards the data is scaled using the MinMaxScaler. 

The model architecture is constructed using the Keras API. Starting with an input layer size 108. Afterwards, a sequence of fully connected layers is employed, starting with a layer of 512 units, followed by Leaky Rectified Linear Unit (LeakyReLU) activation with an alpha parameter of 0.01. This process is repeated for two additional hidden layers with 256 and 128 units, respectively, each followed by LeakyReLU activation. The architecture culminates in a layer with 32 units and LeakyReLU activation. To mitigate overfitting, a dropout layer with a rate of 0.2 is inserted before the final output layer. The output layer consists of a single unit utilizing a sigmoid activation function for classify between seizure and no-seizure.

The ANN is trained with a batch size of 64 for 250 epoches. Early stopping is also implanted when validation loss doesn't improve for 50 epoches.
The model is saved as 'model.h5'

### 2) Predicting a seizure using the ANN
To predict seizures, the two files, 'pre_process_test_ANN.py' and 'predict.py,' are needed. The data undergoes preprocessing, following the same steps as during training, including notch filtering, bandpass filtering, resampling, 3 montage, wavelet transformation, feature extraction, and scaling. Once the seizures are predicted within each segment using the Artificial Neural Network (ANN) a seizure in a session is predicted under the following conditions:
- a 'possible seizure' in a session is classified when at least two consecutive segments with predicted seizure appear
- if there are several 'possible seizures', the one with the most consecutive ones is classified as 'true seizure'
- if there are several 'true seizures', the temporally first seizure is classified as 'true seizure'

The onset of the seizure is then predicted as the start of the initial segment of a 'true seizure'.


## Results
The results from previous training of the ANN are shown below. 

### Evaluation of seizure prediction:
{'performance_metric_WS23': 0.3018867924528302,
 'performance_metric_SS23': 72.05488055430664,
 'detection_error_onset': 38.44174186817525,
 'F1': 0.5955786736020806,
 'sensitivity': 0.9828326180257511,
 'PPV': 0.42723880597014924,
 'accuracy': 0.4324817518248175,
 'detection_error_offset': 60.0,
 'i_sensitivity': 0.41201716738197425,
 'i_PPV': 0.23821339950372208,
 'i_accuracy': 0.1897810218978102,
 'confusion_matrix': (229, 4, 307, 8)}


## Resources and Acknowledgemens
The data for training the ANN is provided in the folder "shared_data" and consists of EEG data in the form of .mat-files. The files have been contributed by the KISMED institute at TU Darmstadt. The function (load_files) was based on the original function (load_references) of KISMED Team. In addition, the original get_3montages function and the structure of EEGScript were also used. 

