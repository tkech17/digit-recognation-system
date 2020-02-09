import os

import librosa
import numpy
import soundfile

DATA_DIRS = ['noise_voices/', 'recordings/']
DEST_DIR = 'noise_voices/'


def addNoisedAudios(noise_rate, unique_identifier):
    for data_dir in DATA_DIRS:
        for file in os.listdir(data_dir):
            if file.endswith('.wav'):
                wave, sr = librosa.load(data_dir + file, mono=True, sr=None)
                noise = numpy.random.normal(0, noise_rate, len(wave))
                modified = wave + noise
                soundfile.write(DEST_DIR + file[:-4] + '-noise-' + str(noise_rate) + unique_identifier + '.wav', modified, sr)