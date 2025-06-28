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
## Info:
## 8 non-empty values, bads [], chs: 23 EEG, custom_ref: false,
## highpass: 0, lowpass: 128 Hz,
## sfreq: 256 Hz
print("##########")

##* Preprocessing

def preprocess(raw):
	"""
	Preprocessing the raw `EDF` data > Returns processed `EDF` data.
	1. Standardize the naming of channels\n
	2. Filter signals\n
	3. No resample to 256Hz needed\n
	"""
	raw.rename_channels(lambda x: x.strip().upper()) # Std Naming to UPPERCASE
	raw.filter(0.50, 40, fir_design='firwin') # Filtering 0.5-40Hz
	raw.notch_filter(60.) # US line noise

	pp_raw = raw
	return pp_raw
