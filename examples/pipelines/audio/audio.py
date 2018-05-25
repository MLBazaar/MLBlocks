#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Usage example for an Audio Pipeline on the urban sound dataset."""
import glob
import os
import subprocess
import tempfile
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from scipy.io.wavfile import read

from mlblocks.ml_pipeline.ml_pipeline import MLPipeline


def segment(f, window_size = 2000, percent_overlap = 0.1):
    """
    params:
        window_size = size of window in milliseconds, float
        percent_overlap = amount of frame overlap for segments, float
    """
    samp_freq, data = read(f)
    num_channels = data.ndim
    num_samples = data.shape[0]
    start_intervals = []
    segments = []

    window = int(samp_freq * window_size/1000.0)
    offset = int((1-percent_overlap) * window)
    num_segments = num_samples // offset

    for i in range(num_segments):
        x = i * offset
        start_intervals.append(x)
        segment = data[x:x+window]
        segments.append(segment.reshape(len(segment), num_channels))
    return samp_freq, start_intervals, segments, num_channels


def load_and_segment(paths):
    # Loads and segments audio files given a list of paths
    segment_vector = []
    sample_freqs = []
    for path in paths:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as fd:
            subprocess.call(
                [
                    'ffmpeg',
                    '-y',
                    '-i', path,
                    '-vn',
                    '-ar', '44100',
                    '-ac', '2',
                    '-f', 'wav',
                    fd.name
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL)
            samp_freq, start_intervals, segments, num_channels = segment(fd.name)
            segment_vector.append(segments)
            sample_freqs.append(samp_freq)
    return segment_vector, sample_freqs

print("""
============================================
Testing Audio Pipeline
============================================
""")

# Data loading.
classes = ['street_music', 'siren', 'jackhammer', 'gun_shot', 'engine_idling', 'drilling', 'dog_bark', 'children_playing', 'car_horn', 'air_conditioner']

labels = []
all_filepaths = []
for label_class in classes:
    for filepath in glob.glob(os.path.join('data/UrbanSound/data', label_class, '*.wav')):
        all_filepaths.append(filepath)
        labels.append(label_class)

filepaths, filepaths_test, y, y_test = train_test_split(
    all_filepaths, labels, train_size=160, test_size=40)

audio_pipeline = MLPipeline.from_ml_json(['audio_featurizer', 'audio_padder', 'pca', 'random_forest_classifier'])

# Check that the hyperparameters are correct.
for hyperparam in audio_pipeline.get_tunable_hyperparams():
    print(hyperparam)

# Check that the steps are correct.
expected_steps = {'audio_featurizer', 'audio_padder', 'pca', 'rf_classifier'}
steps = set(audio_pipeline.steps_dict.keys())
assert expected_steps == steps

# Check that we can score properly.
print("\nFitting pipeline...")
X, sample_freqs = load_and_segment(filepaths)
fit_params = {('audio_featurizer', 'sample_freqs'): sample_freqs}
audio_pipeline.fit(X, y, fit_params=fit_params)
print("\nFit pipeline.")

print("\nScoring pipeline...")
X_test, sample_freqs_test = load_and_segment(filepaths_test)
predict_params = {('audio_featurizer', 'sample_freqs'): sample_freqs_test}
predicted_y_val = audio_pipeline.predict(X_test, predict_params)
score = f1_score(predicted_y_val, y_test, average='micro')
print("\nf1 micro score: %f" % score)
