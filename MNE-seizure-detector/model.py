## Algorithm: 		MNE Seizure Detector
## Database:		CHB-MIT-EEG-Seizure database
## Current File:	model.py - To train the model on the extracted features

import h5py
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
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

# Visualize a seizure window
def plot_seizure_window(h5_path, seizure_index):
    with h5py.File(h5_path, 'r') as hf:
        features = hf['features'][seizure_index]
        label = hf['labels'][seizure_index]

    # Reshape to (channels, time)
    window = features.reshape(23, -1)

    plt.figure(figsize=(12, 6))
    for i in range(23):
        plt.plot(window[i] + i*0.5)
    plt.title(f"Seizure Window (Label: {label})")
    plt.xlabel("Samples")
    plt.ylabel("Channels")
    plt.show()

##* Prepare model
def create_model(input_shape):
	"""
	Create a simple model to train on the data.
	"""
	model = tf.keras.Sequential([
		tf.keras.layers.Input(shape=(input_shape,)),
		tf.keras.layers.Dense(128, activation='relu'),
		tf.keras.layers.Dropout(0.5),
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

##* Train model
def train_model():
	# load data
	X, y = load_data()
	#* CHECKING FOR PROBLEMS
	print("Full dataset:")
	print(f"  Total samples: {len(y)}")
	print(f"  Seizure samples: {np.sum(y)}")
	print(f"  Prevalence: {np.mean(y)*100:.4f}%")

	seizure_indices = np.where(y == 1)[0]
	plot_seizure_window("eeg_features.h5", seizure_indices[0])

	# split data
	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=0.2, random_state=7, stratify=y
	)
	#* CHECKING FOR PROBLEMS
	print("\nTraining set:")
	print(f"  Samples: {len(y_train)}")
	print(f"  Seizure: {np.sum(y_train)}")
	print(f"  Prevalence: {np.mean(y_train)*100:.4f}%")

	print("\nTest set:")
	print(f"  Samples: {len(y_test)}")
	print(f"  Seizure: {np.sum(y_test)}")
	print(f"  Prevalence: {np.mean(y_test)*100:.4f}%")

	print("#########################################")
	# calculate weights
	class_weights = class_weight.compute_class_weight(
		'balanced', classes=np.unique(y_train), y=y_train
	)
	class_weights = {i: weight for i, weight in enumerate(class_weights)}
	print(f"Class weights: {class_weights}")
	# create and train
	model = create_model(X.shape[1])

	early_stop = tf.keras.callbacks.EarlyStopping(
		monitor='val_prc',
		patience=5,
		mode='max',
		restore_best_weights=True
	)
	history = model.fit(
		X_train, y_train,
		epochs=10,
		batch_size=1024,
		validation_split=0.1,
		class_weight=class_weights,
		callbacks=[early_stop],
		verbose=2
	)

	# evaluate
	print("\n=== Final Evaluation ===")
	results = model.evaluate(X_test, y_test)
	print(f"Test Loss: {results[0]:.4f}")
	print(f"Test PR-AUC: {results[1]:.4f}")
	print(f"Test Sensitivity: {results[2]:.4f}")

	# save model
	model.save("seizure_detector.keras")
	print("Model saved to seizure_detector.keras")


if __name__ == "__main__":
	print("TF Version:", tf.__version__)
	print("Keras access:", tf.keras.__version__)
	print()

	# Check feature distributions
	with h5py.File("eeg_features.h5", 'r') as hf:
		features = hf['features'][:]
		seizure_mask = hf['labels'][:] == 1

		print("Non-seizure feature stats:")
		print(pd.DataFrame(features[~seizure_mask]).describe().loc[['mean', 'std']])

		print("\nSeizure feature stats:")
		print(pd.DataFrame(features[seizure_mask]).describe().loc[['mean', 'std']])

	train_model()
