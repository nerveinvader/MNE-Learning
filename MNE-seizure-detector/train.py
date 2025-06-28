## Algorithm:		MNE Seizure Detector
## Database:		CHB-MIT-EEG-Seizure database
## Current File:	train.py - To train the model on the dataset

#%%
##* Imports
import os
import mne

#%%
##* Data
print("Current directory: ", os.getcwd())

# All the files
#DATA_PATH = "./MNE-seizure-detector/data/chb-mit-eeg-database-1.0.0" # data dir for run command
DATA_PATH = "./data/chb-mit-eeg-database-1.0.0" # data dir for interactive

print("Dataset is ready: ", os.path.isdir(DATA_PATH))
print("##########")

# Summary of seizure recordings
#SUMMARY_FILE = os.path.join(DATA_PATH, "RECORDS-WITH-SEIZURES")
#with open(SUMMARY_FILE) as f:
#	print(f"Seizure records:\n{f.read()[:500]}.") # preview first 500 chars

# Sample inspection (.edf)
sample_raw = mne.io.read_raw_edf(os.path.join(DATA_PATH, "./chb01/chb01_01.edf"), preload=True)
#print("Sample Inspection: chb01/chb01_01.edf")
#print(sample_raw.info)
## Info:
## 8 non-empty values, bads [], chs: 23 EEG, custom_ref: false,
## highpass: 0, lowpass: 128 Hz,
## sfreq: 256 Hz
print("##########")

#%%
##* Preprocessing

def preprocess(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
	"""
	Preprocessing the raw `mne.io.BaseRaw`.edf data > Returns processed `mne.io.BaseRaw`.edf data.
	1. Standardize the naming of channels\n
	2. Filter signals\n
	3. No resample to 256Hz needed\n
	"""
	pp_raw = raw.copy()

	# Renaming to UPPERCASE (Std)
	pp_raw.rename_channels(lambda x: x.strip().upper())

	# Filtering 0.5-40Hz (non-brain activity)
	# Signals above 40 will be contaminated by EMG/Muscle signals.
	# In a study, over 40Hz showed +0.7% detection improvement with +31% false alarms
	pp_raw.filter(0.50, 40, fir_design='firwin')

	# Filter exactly 60 +/- 0.5Hz
	# Analyzing the signals show a notch @ 60.Hz (the US noise)
	pp_raw.notch_filter(60.)

	return pp_raw

#%%
# Comparison (before-after preprocessing)
sample_raw.plot_psd(fmax=120, show=False)
preprocess(sample_raw).plot_psd(fmax=120)
print("##########")

#%%
# Seizure annotation parser (on CHB MIT DB format)
def get_seizure_labels(annotation_file: str) -> dict:
	"""
	Extract seizure start & end times.
	Looks for chb##-summary.txt file.
	Format of the .txt file:\n
	[
	File Name: chb##_##.edf,
	File Start Time: #:#:#,
	File End Time: #:#:#
	Seizure Start Time: # seconds
	Seizure End Time: # seconds
	]\n
	Returns {"edf_filename": [(start, end), ...]}
	"""
	seizure_dict: dict = {}
	current_file = None

	with open(annotation_file, 'r') as an:
		for line in an:
			line.strip()
			if line.startswith("File Name:"):
				current_file = line.split(":")[1].strip()
				seizure_dict[current_file] = []
			if line.startswith("Seizure Start Time:"):
				start = float(line.split()[3])
				next_line = next(an).strip()
				end = float(next_line.split()[3])
				seizure_dict[current_file].append((start, end))
	return seizure_dict


#%%
# Annotation example
print(get_seizure_labels(os.path.join(DATA_PATH, "./chb01/chb01-summary.txt")))

# %%
