from __future__ import division, print_function

import os
import shutil
from os import listdir
from os.path import isfile, join

import scipy.io.wavfile as wav
from matplotlib import pyplot as plt

SPECTOGRAMS_PATH = "spectograms/"


class VoiceData:
    def __init__(self, label, spectogram):
        self.label = label
        self.spectogram = spectogram


# take only .wav type
def __convertAutioIntoSpectogram(audio_path: str, dest_path: str):
    spectrogram_dimensions = (64, 64)
    noverlap = 16
    cmap = 'gray_r'

    sample_rate, samples = wav.read(audio_path)
    if len(samples.shape) != 1:
        samples = samples[:, 0]

    fig = plt.figure()
    fig.set_size_inches((spectrogram_dimensions[0] / fig.get_dpi(), spectrogram_dimensions[1] / fig.get_dpi()))
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.specgram(samples, cmap=cmap, Fs=2, noverlap=noverlap)
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())
    fig.savefig(dest_path, bbox_inches="tight", pad_inches=0)
    return dest_path


def voiceToSpectograms(voice_dirs):
    if os.path.exists(SPECTOGRAMS_PATH):
        removeImagesDirecotory()
    os.mkdir(SPECTOGRAMS_PATH)
    res = []
    for voice_dir in voice_dirs:
        file_names = [f for f in listdir(voice_dir) if isfile(join(voice_dir, f)) and f.endswith(".wav")]
        for file_name in file_names:
            number = file_name[0]
            audio_path = voice_dir + file_name
            try:
                res.append(VoiceData(number, __convertAutioIntoSpectogram(audio_path, SPECTOGRAMS_PATH + file_name[:-3] + "PNG")))
            except Exception:
                print("yle")
    return res


def removeImagesDirecotory():
    shutil.rmtree(SPECTOGRAMS_PATH)
