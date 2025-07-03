## Algorithm: 		MNE Seizure Detector
## Database:		CHB-MIT-EEG-Seizure database
## Current File:	model.py - To train the model on the extracted features

import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

##* Load data
def load_data(h5_path: str = "eeg_features.h5") -> tuple:
	"""
	Load h5 file as data (extracted features).
	Returns a `tuple` [features, labels]
	"""
	with h5py.File(h5_path, 'r') as hf:
		features = hf['features'][:]
		labels = hf['labels'][:]
	return features, labels

##* Prepare model
def create_model(input_shape):

	model = tf.keras.Sequential([
		tf.keras.layers.Dense(128, activation='relu', input_shape=(input_shape,)),
		tf.keras.layer.Dropout(0.5),
		tf.keras.layers.BatchNormalization(),
		tf.keras.layers.Dense(64, activation='relu'),
		tf.keras.layers.Dense(1, activation='sigmoid')
	])

	model.compile(
		optimizer=tf.keras.optimizers.Adam(0.001),
		loss='binary_crossentropy',
		metrics=[
			tf.keras.metrics.AUC(name='prc', curve='PR'),
			tf.keras.metrics.Recall(name='sensitivity')
		]
	)
	return model

if __name__ == "__main__":
	print(tf.__version__)
	# X, y = load_data()
	# create_model(X.shape[1])
