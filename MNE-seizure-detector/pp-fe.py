## Algorithm:		MNE Seizure Detector
## Database:		CHB-MIT-EEG-Seizure database
## Current File:	pp-fe.py - To extract features and preprocess the dataset

#%
##* Imports
import os
import pickle
import glob
import mne
import numpy as np
from scipy import signal, stats
import h5py
from collections import defaultdict

import tqdm
import pandas as pd

#%
##* Data
print("Current directory: ", os.getcwd())

# All the files
DATA_PATH = "./MNE-seizure-detector/data/chb-mit-eeg-database-1.0.0" # data dir for run command
DATA_PATH2 = "./data/chb-mit-eeg-database-1.0.0" # data dir for interactive

print("Dataset is ready: ", os.path.isdir(DATA_PATH))
print("##########")

# Summary of seizure recordings
# SUMMARY_FILE = os.path.join(DATA_PATH, "RECORDS-WITH-SEIZURES")
# with open(SUMMARY_FILE) as f:
#	print(f"Seizure records:\n{f.read()[:500]}.") # preview first 500 chars

# Sample inspection (.edf)
# sample_raw = mne.io.read_raw_edf(os.path.join(DATA_PATH, "./chb01/chb01_01.edf"), preload=True)
# print("Sample Inspection: chb01/chb01_01.edf")
# print(sample_raw.info)
## Info:
## 8 non-empty values, bads [], chs: 23 EEG, custom_ref: false,
## highpass: 0, lowpass: 128 Hz,
## sfreq: 256 Hz
print("##########")

#%
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
	pp_raw.filter(0.50, 40, fir_design='firwin', verbose=0)

	# Filter exactly 60 +/- 0.5Hz
	# Analyzing the signals show a notch @ 60.Hz (the US noise)
	pp_raw.notch_filter(60., verbose=0)

	return pp_raw

#% # Comparison - non essential
# Comparison (before-after preprocessing)
# sample_raw.plot_psd(fmax=120, show=False)
# preprocess(sample_raw).plot_psd(fmax=120)
# print("##########")

#%
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

#% # Annotation example - non essential
# Annotation example
#print(get_seizure_labels(os.path.join(DATA_PATH, "./chb01/chb01-summary.txt")))

#%
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

#% # Example of one patient data processing - non essential
##* Example
# if __name__ == "__main__":
# 	# Specific patient
# 	patient_id = "chb01"
# 	annotation_path = f"{DATA_PATH}/{patient_id}/{patient_id}-summary.txt"

# 	seizure_labels = get_seizure_labels(annotation_path)

# 	pickle_path = f"{DATA_PATH}/{patient_id}/{patient_id}-seizure-labels.pkl"
# 	save_labels_to_pickle(seizure_labels, pickle_path)

# 	labels = load_labels_from_pickle(pickle_path)
# 	print(f"Patient chb01: {len(labels)} files, {sum(len(v) for v in labels.values())} seizures.")

#%
##* Multi Patient Processing to Save pickle file
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

#%
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
		if not os.path.isdir(patient_dir):
			continue

		patient_id = os.path.basename(patient_dir)
		report["total_patients"] += 1
		ann_path = os.path.join(patient_dir, f"{patient_id}-summary.txt")
		pckl_path = os.path.join(patient_dir, f"{patient_id}-seizure-labels.pkl")

		if not os.path.exists(ann_path):
			report["missing_summaries"].append(patient_id)
			continue

		if os.path.exists(pckl_path):
			report["processed_patients"] += 1
			try:
				labels = load_labels_from_pickle(pckl_path)
				#print(labels)	# Debug purpose
				report["total_seizures"] += sum(len(v) for v in labels.values())
			except:
				print("Remove", pckl_path)
				#os.remove(pckl_path)	# If corrupted detected.

	# Business-ready report
	print("\n=== DATASET INTEGRITY REPORT ===")
	print(f"Patients: {report['processed_patients']}/{report['total_patients']} processed")
	print(f"Total seizures: {report['total_seizures']}")

	if report["missing_summaries"]:
		print(f"\nWARNING: Missing summaries for {len(report['missing_summaries'])} patients:")
		print(", ".join(report["missing_summaries"]))

	return report

##* Example of Business report
# report = verify_dataset(DATA_PATH)
# report_df = pd.DataFrame([report])
# report_df.to_markdown("dataset_report.md") # MD file of report.

# %
##* Example of Multi patient processing
# process_all_patients()

# %
##* Segmentation, Assigning & Feature extraction
# Windowing system (segmentation)
def create_segments(raw: mne.io.Raw, seg_sec: float = 4, stride_sec: float = 2) -> np.ndarray:
	"""
	Split EEG into overlapping segments.
	Returns (n_segments, n_channels, n_sampless)
	"""
	sfreq = raw.info['sfreq']
	n_samples = int(seg_sec * sfreq)
	step = int(stride_sec * sfreq)
	data = raw.get_data()

	segments = []
	for start in range(0, data.shape[1] - n_samples, step):
		segments.append(data[:, start:start+n_samples])
	return np.stack(segments)

# Assign label to segments
def assign_seg_label(
		segments: np.ndarray,
		seizure_intervals,
		seg_sec, stride_sec) -> np.ndarray:
	"""
	Assign seizure labels to segments based on annotation times.
	"""
	# create label array [0 non, 1 seizure]
	labels = np.zeros(len(segments), dtype=np.int8)

	# calculate seg start time in sec
	seg_start = [i * stride_sec for i in range(len(segments))]
	seg_end = [start + seg_sec for start in seg_start]

	# mark segments
	for start_sec, end_sec in seizure_intervals:
		for i, (sg_start, sg_end) in enumerate(zip(seg_start, seg_end)):
			if (sg_start < end_sec) and (sg_end > start_sec):
				labels[i] = 1
	return labels

# Feature extract
def extract_features(segment: np.ndarray, sfreq: int) -> np.ndarray:
	"""
	Extract clinically validated features from an EEG segment.
	Returns 1D feature vector of (n_channels * n_features)
	"""
	features = []
	for chnl_data in segment:
		# Time-domain features
		features.append(np.mean(chnl_data))		# mean
		features.append(np.std(chnl_data))		# variance
		features.append(stats.skew(chnl_data))	# skewness
		# kurtosis ?

		# Freq-domain features
		freq, psd = signal.welch(chnl_data, sfreq, nperseg=256)	# power spectral density
		for band in [(0.5, 4), (4, 8), (8, 12), (12, 30), (30, 40)]:
			band_mask = (freq >= band[0]) & (freq <= band[1])
			features.append(np.mean(psd[band_mask]))

		# Non-linear features
		diff = np.diff(chnl_data)
		features.append(np.sum(np.abs(diff)))	# line length
		features.append(np.sum(diff * 2))		# energy
	return np.array(features)

#%
# Save buffered data to HDF5 datasets
def save_buffer_to_hdf5(buffer_features, buffer_labels, features_dset, labels_dset):
	"""
	Append buffered data to HDF5 datasets
	"""
	# convert to np.array
	f_array = np.array(buffer_features, dtype=np.float32)	# features
	l_array = np.array(buffer_labels, dtype=np.int8)		# labels

	# resize
	current_size = features_dset.shape[0]
	new_size = current_size + f_array.shape[0]
	features_dset.resize((new_size, f_array.shape[1]))
	labels_dset.resize((new_size,))

	# append data
	features_dset[current_size:new_size] = f_array
	labels_dset[current_size:new_size] = l_array

#%
##* Feature extraction process for all patients
# Config
SEG_SEC = 4.0		# standard segment size
STRIDE_SEC = 2.0	# 50% overlap
SAMPLE_RATE = 256	# standard sample rate
BUFFER_SIZE = 5000	# segements to load before writing on disk
# standard channels - 23 Ch
STD_CHANNELS = [
	'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
	'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8-0', 'P8-O2',
	'FZ-CZ', 'CZ-PZ', 'P7-T7', 'T7-FT9', 'FT9-FT10', 'FT10-T8', 'T8-P8-1'
]

def process_full_dataset(base_dir: str, output_file: str = "eeg_features.h5"):
	"""
	Full pipeline. Process all patients.
	Save features + labels on disk with `HDF5` format
	"""
	with h5py.File(output_file, 'w') as hf:
		# datasets
		features_dset = hf.create_dataset(
			'features', (0, 0),
			maxshape=(None, None), chunks=(1000, 230),
			compression='gzip')
		labels_dset = hf.create_dataset(
			'labels', (0,),
			maxshape=(None,), chunks=(1000,),
			compression='gzip')
		metadata = hf.create_group('metadata')

		# process each patient
		patient_dirs = [d for d in os.listdir(base_dir)
				  if d.startswith('chb') and os.path.isdir(os.path.join(base_dir, d))]
		# patient_dirs = [chb01,...,chb24]

		total_segments = 0
		buffer_features = []
		buffer_labels = []

		for patient_id in tqdm.tqdm(sorted(patient_dirs), desc="Processing patients:"):
			patient_path = os.path.join(base_dir, patient_id)
			# patient_path = "./MNE-seizure-detector/data/chb-mit-eeg-datab.../chb01,...,chb24"

			# load labels from pickle file
			labels_path = os.path.join(patient_path, f"{patient_id}-seizure-labels.pkl")
			if not os.path.exists(labels_path):
				print(f"Skipping {patient_id}: No labels found.")
				continue

			seizure_labels = load_labels_from_pickle(labels_path)

			# process edf file
			edf_files = [f for f in os.listdir(patient_path)
				if f.endswith('.edf') and not f.startswith('.')]

			for edf_file in tqdm.tqdm(edf_files, desc=f"Files in {patient_id}", leave=False):
				# skip files without annotation
				if edf_file not in seizure_labels:
					continue

				file_path = os.path.join(patient_path, edf_file)

				try:
					# load and preprocess
					raw = mne.io.read_raw_edf(file_path, preload=True)
					p_raw: mne.io.Raw = preprocess(raw)

					# std channels
					p_raw.pick_channels([ch.upper().strip() for ch in STD_CHANNELS])

					# create seg and labels
					segments = create_segments(p_raw, SEG_SEC, STRIDE_SEC)
					segment_labels = assign_seg_label(
						segments, seizure_labels[edf_file],
						SEG_SEC, STRIDE_SEC)

					# extract features
					for i, segment in enumerate(segments):
						features = extract_features(segment, SAMPLE_RATE)
						buffer_features.append(features)
						buffer_labels.append(segment_labels[i])

						# save when buffer fills up
						if len(buffer_features) >= BUFFER_SIZE:
							save_buffer_to_hdf5(
								buffer_features, buffer_labels,
								features_dset, labels_dset)
							buffer_features = []
							buffer_labels = []
					total_segments += len(segments)
				except Exception as e:
					print(f"Error processing {edf_file}: {str(e)}")
		if buffer_features:
			save_buffer_to_hdf5(
				buffer_features, buffer_labels,
				features_dset, labels_dset)

		# add metadata
		metadata.attrs['total_patients'] = len(patient_dirs)
		metadata.attrs['total_segments'] = total_segments
		metadata.attrs['segment_sec'] = SEG_SEC
		metadata.attrs['stride_sec'] = STRIDE_SEC
		metadata.attrs['channels'] = str(STD_CHANNELS)

	print(f"\nCompleted! Saved {total_segments} segments to {output_file}")

##* Verify processed h5 file
def verify_processing():
	"""
	Business-critical validation of processed data
	"""
    # 1. Check file size
	file_size = os.path.getsize('eeg_features.h5') / (1024 * 1024)  # in MB
	print(f"File size: {file_size:.1f} MB")

    # 2. Check content
	with h5py.File('eeg_features.h5', 'r') as hf:
		n_windows = hf['features'].shape[0]
		seizure_count = np.sum(hf['labels'][:])

		print(f"Total windows: {n_windows}")
		print(f"Seizure windows: {seizure_count} ({seizure_count/n_windows*100:.2f}%)")

		# 3. Sample validation
		sample_features = hf['features'][0]
		print(f"Sample feature range: [{np.min(sample_features):.4f}, {np.max(sample_features):.4f}]")

		# Business rule checks
		assert file_size > 100, "File too small - processing likely failed"
		assert n_windows > 50000, "Insufficient windows processed"
		assert 0.5 < seizure_count/n_windows < 10, f"Abnormal seizure prevalence {seizure_count/n_windows}"

	print("✅ Business validation passed")

##* Load seizure times from .edf.seizure files
def process_patient(base_dir, patient_id, output_file="eeg_features_next.h5"):
	patient_dir = os.path.join(base_dir, patient_id)
	labels_path = os.path.join(patient_dir, f"{patient_id}-seizure-labels.pkl")
	if not os.path.exists(labels_path):
		print(f"Skipping {patient_id}: No labels found")
		return

	seizure_labels = load_labels_from_pickle(labels_path)

	with h5py.File(output_file, 'a') as hf:
		if 'features' not in hf:
			hf.create_dataset('features', (0, 230), maxshape=(None, 230), chunks=True, compression='gzip')
			hf.create_dataset('labels', (0,), maxshape=(None,), chunks=True, compression='gzip')
			hf.attrs['segment_sec'] = SEG_SEC
			hf.attrs['stride_sec'] = STRIDE_SEC
		edf_files = [f for f in os.listdir(patient_dir)
			   if f.endswith('.edf') and f in seizure_labels]

		for edf_file in tqdm.tqdm(edf_files, desc=f"Processing {patient_id}"):
			file_path = os.path.join(patient_dir, edf_file)
			try:
				raw = mne.io.read_raw_edf(file_path, preload=True)
				raw = preprocess(raw)
				# p_raw.pick_channels([ch.upper().strip() for ch in STD_CHANNELS])
				raw = raw.pick([ch.upper().strip() for ch in STD_CHANNELS])

				segments = create_segments(raw, SEG_SEC, STRIDE_SEC)
				segment_labels = assign_seg_label(
					segments, seizure_labels[edf_file],
					SEG_SEC, STRIDE_SEC)
				for i, segment in enumerate(segments):
					features = extract_features(segment, SAMPLE_RATE)

					features_ds = hf['features']
					labels_ds = hf['labels']

					buffer_f = []
					buffer_l = []
					buffer_f.append(features)
					buffer_l.append(segment_labels[i])

					save_buffer_to_hdf5(buffer_f, buffer_l, features_ds, labels_ds)
					buffer_f = []
					buffer_l = []

					# new_size = features_ds.shape[0] + 1
					# features_ds.resize((new_size, 230))
					# labels_ds.resize((new_size,))

					# features_ds[-1] = features
					# labels_ds[-1] = segment_labels[i]
				n_seizure = np.sum(segment_labels)
				print(f"{edf_file}: {len(segments)} segments, {n_seizure} seizure")
			except Exception as e:
				print(f"Error processing {edf_file}: {str(e)}")
	print(f"Completed {patient_id}.")

#%
##* Run the pipeline
if __name__ == "__main__":
	# Full dataset
	# process_full_dataset(base_dir=DATA_PATH)

	# Verification
	# verify_processing()

	# with h5py.File('eeg_features.h5', 'r') as hf:
	# 	labels = hf['labels'][:]
	# 	print(f"Total seizure segments: {np.sum(labels)}")
	# 	print(f"Max label value: {np.max(labels)}")
	# 	print(f"Label value counts: {np.unique(labels, return_counts=True)}")

	process_patient(DATA_PATH, "chb10")

	with h5py.File("eeg_features_next.h5", 'r') as hf:
		labels = hf['labels'][:]
		print(f"Seizure prevalence: {np.mean(labels)*100:.2f}%")


	#! Debugs
	# patient_id = "chb01"
	# pickle_path = f"{DATA_PATH}/{patient_id}/{patient_id}-seizure-labels.pkl"
	# seizure_labels = load_labels_from_pickle(pickle_path)
	# print("Seizure intervals in chb01_03.edf: ", seizure_labels.get("chb01_03.edf", []))

	# test_intervals = [(10, 20)]
	# test_segment = np.zeros((100, 23, 1024))
	# test_labels = assign_seg_label(segments=test_segment, seizure_intervals=test_intervals, seg_sec=4, stride_sec=2)
	# print("Seizure segments should be ~5: ", np.sum(test_labels))
