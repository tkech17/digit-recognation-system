import os

import librosa
import numpy
import soundfile

DATA_DIR = 'recordings/'


def addNoisedAudios(noise_rate):
    for file in os.listdir(DATA_DIR):
        if file.endswith('.wav'):
            wave, sr = librosa.load(DATA_DIR + file, mono=True, sr=None)
            noise = numpy.random.normal(0, noise_rate, len(wave))
            modified = wave + noise
            soundfile.write(DATA_DIR + file[:-4] + '-noise-' + str(noise_rate) + '.wav', modified, sr)