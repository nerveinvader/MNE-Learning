## Algorithm:		MNE Seizure Detector
## Database:		CHB-MIT-EEG-Seizure database
## Current File:	train.py - To train the model on the dataset

#%%
##* Imports
import os
import pickle
import glob
import mne
import tqdm

#%%
##* Data
print("Current directory: ", os.getcwd())

# All the files
DATA_PATH = "./MNE-seizure-detector/data/chb-mit-eeg-database-1.0.0" # data dir for run command
#DATA_PATH = "./data/chb-mit-eeg-database-1.0.0" # data dir for interactive

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
	if not os.path.exists(annotation_file):
		# File not found Error
		raise FileNotFoundError(f"Summary File (.txt) not found at: {annotation_file}.")

	seizure_dict: dict = {}
	current_file = None
	seizure_active = False

	with open(annotation_file, 'r') as an:
		start = 0.0	## Start of the seizure
		end = 0.0	## End of the seizure
		for line in an:
			line.strip()
			if line.startswith("File Name:"):
				current_file = line.split(":")[1].strip()
				seizure_dict[current_file] = []
				seizure_active = False
			elif line.startswith("Seizure Start Time:"): # Start
				if current_file is None:
					raise ValueError("Seizure start encountered before file declaration.")
				try:
					start = float(line.split()[3])
					seizure_active = True
				except (IndexError, ValueError):
					print(f"Warning: Malformed seizure start line: {line}")
					continue
			elif line.startswith("Seizure End Time:") and seizure_active:
				try:
					#next_line = next(an).strip()
					end = float(line.split()[3])
					seizure_dict[current_file].append((start, end))
				except (IndexError, ValueError):
					print(f"Warning: Malformed seizure end line: {line}.")
	totatl_files = len(seizure_dict)
	seizure_files = sum(1 for v in seizure_dict.values() if v)
	seizure_count = sum(len(v) for v in seizure_dict.values())
	print(f"Processed {totatl_files} EEG files.")
	print(f"- Files with seizures: {seizure_files}")
	print(f"- Files without seizures: {totatl_files - seizure_files}.")
	print(f"- Seizures found: {seizure_count}.")
	return seizure_dict

#%%
# Annotation example
print(get_seizure_labels(os.path.join(DATA_PATH, "./chb01/chb01-summary.txt")))

# %%
# Serialization for Business
def save_labels_to_pickle(labels_dict: dict, output_path: str):
	"""
	Serialize labels dictionary for quick reloading.
	Write as binary format.
	"""
	with open(output_path, 'wb') as pkl_file:
		pickle.dump(labels_dict, pkl_file)
	print(f"Saved labels to {output_path}; (Size: {os.path.getsize(output_path)/1e6:.1f} MB)")

def load_labels_from_pickle(pickl_path: str) -> dict:
	"""
	Load serialized labels dictionary.
	Read binary format.
	"""
	if not os.path.exists(pickl_path):
		raise FileNotFoundError(f"Pickle file not found at: {pickl_path}.")
	with open(pickl_path, 'rb') as pckl_file:
		return pickle.load(pckl_file)

# %%
##* Example
if __name__ == "__main__":
	# Specific patient
	patient_id = "chb01"
	annotation_path = f"{DATA_PATH}/{patient_id}/{patient_id}-summary.txt"

	seizure_labels = get_seizure_labels(annotation_path)

	pickle_path = f"{DATA_PATH}/{patient_id}/{patient_id}-seizure-labels.pkl"
	save_labels_to_pickle(seizure_labels, pickle_path)

	labels = load_labels_from_pickle(pickle_path)
	print(f"Patient chb01: {len(labels)} files, {sum(len(v) for v in labels.values())} seizures.")

# %%
##* Multi Patient Processing
def process_single_patient(patient_dir, patient_id):
	"""
	Process a single patient using the pipeline above.
	Saves a `.pkl` file per patient. Doesn't return anything.
	"""
	ann_path = os.path.join(patient_dir, f"{patient_id}-summary.txt")
	pckl_path = os.path.join(patient_dir, f"{patient_id}-seizure-labels.pkl")
	if not os.path.exists(pckl_path):
		try:
			seizure_labels = get_seizure_labels(ann_path)
			save_labels_to_pickle(seizure_labels, pckl_path)
		except Exception as e:
			print(f"Error processing {patient_id}: {str(e)}.")
	else:
		pass

def process_all_patients(base_dir: str = DATA_PATH):
	"""
	Process all the patients.
	"""
	patient_dirs = sorted([d for d in glob.glob(os.path.join(base_dir, "chb*"))
						if os.path.isdir(d)])
	print(f"Found {len(patient_dirs)} patients.")
	# p is abbreviation for patient
	for p_dir in tqdm.tqdm(patient_dirs):
		p_id = os.path.basename(p_dir)
		process_single_patient(p_dir, p_id)

#%%
##* Verifying dataset / Business
def verify_dataset(base_dir: str = DATA_PATH) -> dict:
	"""
	Data validation. Critical for business side reports.
	Returns a `dictionary` as report.
	"""
	report = {
		"total_patients": 0,
		"processed_patients": 0,
		"total_seizures": 0,
		"missing_summaries": [],
	}
	for patient_dir in glob.glob(os.path.join(base_dir, "chb*")):
		if not os.path.exists(patient_dir):
			continue
		patinet_id = os.path.basename(patient_dir)
		report["total_patients"] += 1

		ann_path = os.path.join(patient_dir, f"{patient_id}-summary.txt")
		pckl_path = os.path.join(patient_dir, f"{patient_id}_seizure_labels.pkl")

		if not os.path.exists(ann_path):
			report["missing_summaries"].append(patient_id)
			continue

		if not os.path.exists(pckl_path):
			report["processed_patients"] += 1
			try:
				labels = load_labels_from_pickle(pckl_path)
				report["total_seizures"] += sum(len(v) for v in labels.values())
			except:
				os.remove(pckl_path)

	# Business-ready report
	print("\n=== DATASET INTEGRITY REPORT ===")
	print(f"Patients: {report['processed_patients']}/{report['total_patients']} processed")
	print(f"Total seizures: {report['total_seizures']}")

	if report["missing_summaries"]:
		print(f"\nWARNING: Missing summaries for {len(report['missing_summaries'])} patients:")
		print(", ".join(report["missing_summaries"]))

	return report

# %%
##* Example of Multi patient processing
process_all_patients()

# %%
##* Example of Business report
report = verify_dataset("./data/chb-mit-eeg-database-1.0.0")

# %%
