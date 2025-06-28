## Algorithm:		MNE Seizure Detector
## Database:		CHB-MIT-EEG-Seizure database
## Current File:	train.py - To train the model on the dataset

##* Imports
import os
import mne

##* Data
print("Current directory: ", os.getcwd())

# all the files
DATA_PATH = "./MNE-seizure-detector/data/chb-mit-eeg-database-1.0.0"
print("Dataset is ready: ", os.path.isdir(DATA_PATH))
print("##########")

# summary of seizure recordings
#SUMMARY_FILE = os.path.join(DATA_PATH, "RECORDS-WITH-SEIZURES")
#with open(SUMMARY_FILE) as f:
#	print(f"Seizure records:\n{f.read()[:500]}.") # preview first 500 chars

# sample inspection (.edf)
sample_raw = mne.io.read_raw_edf(os.path.join(DATA_PATH, "./chb01/chb01_01.edf"), preload=False)
print("Sample Inspection: chb01/chb01_01.edf")
print(sample_raw.info)
