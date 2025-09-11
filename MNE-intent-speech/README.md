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
We are going to use the dataset from the Nature paper "Thinking Out Loud", an open-access EEG dataset for inner speech recognition. The dataset contains three tasks per trial (each trial lasts ~5 secs): Intent, Talking, and Relaxing. For more information on their data collection, follow this link: [https://openneuro.org/datasets/ds003626/versions/2.1.2]

## Tasks
We are focusing on the "intent" part of the dataset, while feeding all three phases of each trial to the model. Preprocessing methods that we are using in this project are:
1. Filtering noise and non-EEG signals,
2. Three windows (mne.Epochs) for Intent, Speech, and Resting phases,
3. Filtering baseline frequency from each window (None for Rest phase),
4. Training various models on these inputs,
5. Compare the results and evaluations of each model,
6. Extract features if the results are not viable or significant.

## Model
EEGNet

## Dataset
Inner Speech, link: [https://openneuro.org/datasets/ds003626/versions/2.1.2]..
Files:
Total Trials = 5640
Conditions = 0=Pronounced / 1=Inner / 2=Visualized
Trial/Condition = 1128 / 2236 / 227
Classes = Arriba/U, Abajo/D, Derecha/R, Izquierda/L
Subjects = 10
sub-01 to 10 folders/ses-01 to 03 folders:
- sub-01_ses-01_task-innerspeech_eeg.bdf
- - Raw EEG file (EEG + EOG + EMG channels)
- - Channels = 128 + 8 (24 bits resolution)
- - Sampling rate = 1024 Hz -> 256Hz Final
derivatives/sub-01 to 10 folders/ses-01 to 03 folders:
- sub-01_ses-01_eeg-epo.fif
- - Epochs of EEG data (segmented trials) (-500ms to 4000ms)
- - 200 events, 50 per class (U/D/R/L)
- - Shape = [Trials x 128 ch x 1154]
- - 1154 is number of samples (4.5s) x 256Hz
- sub-01_ses-01_exg-epo.fif
- - Same as eeg-epo.fif
- - Useful for detecting muscle movement for Pronounced speech
- - Shape = [Trials x 8 ch x 1154]
- sub-01_ses-01_baseline-epo.fif
- - Baseline/Rest periods
- - Shape = [1 x 136 ch x 3841]
- - 3841 is seconds of recording times 256Hz (final sampling rate)
- sub-01_ses-01_events.dat
- - Event markers
- - Format = four col matrix: [1, 2, 3, 4]
- - - Each row is one Trial
- - - Col 1, 2 = sample #, class 0-4
- - - Col 3, 4 = condition 0-3, session number 1-3
- sub-01_ses-01_report.pkl
- - Age, Gender, Recording_time
- - Ans_R/_W = answer right/wrong
- - EMG_trials = position of contaminated trials
- - Power_EXG7/EXG8 = mean power for channel of contaminated trials
- - Baseline_EXG7/8_mean/_std = mean and std of the power channels

## Current ToDo
[] Loading and Preprocessing of the dataset
