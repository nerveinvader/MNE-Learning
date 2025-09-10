# Decoding Preparatory Neural State Before Speech
Can we detect and decode EEG patterns before speaking or conceptualizing an idea?

## Problem
Individuals with autism, aphasia, locked-in syndrome, etc., struggle to initiate speech or sometimes conceptualize, even when they have a clear understanding of what they are going to say. If we can detect their intent, we could:
1. Develop an assistive device to initiate before struggling happens,
2. Help therapists identify cognitive readiness,
3. Develop a thought-to-prompt interface,
4. More importantly, understand how the brain translates thoughts to linguistic formulation.

## Gap
Many studies and papers investigate motor intent, emotions, steady-state, and imagined speech; However, attempts to decode preparatory neural state before speech or conceptualization are few.

## Goal
We are attempting to detect, analyze, and decode EEG patterns before initiation of speaking, explaining, and conceptualization.

## Method
We are going to use the dataset from the Nature paper "Thinking Out Loud", an open-access EEG dataset for inner speech recognition. The dataset contains three tasks per trial (each trial lasts ~5 secs): Intent, Talking, and Relaxing. For more information on their data collection, follow this link: [LINK]

## Tasks
We are focusing on the "intent" part of the dataset, while feeding all three phases of each trial to the model. Preprocessing methods that we are using in this project are:
1. Filtering noise and non-EEG signals,
2. Three windows (mne.Epochs) for Intent, Speech, and Resting phases,
3. Filtering baseline frequency from each window (None for Rest phase),
4. Training various models on these inputs,
5. Compare the results and evaluations of each models,
6. Extract features if the results are not viable or significant.
