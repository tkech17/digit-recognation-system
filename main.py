import keras

from Trainer import trainModel
from utils.AudioFileMultiplier import addNoisedAudios
from utils.FileFormatHelper import voiceToSpectograms
from utils.dataSplitHelper import prepareData

VOICE_DIRECTORY = "recordings/"
NOISE_VOICE_DIRECTORY = "noise_voices/"


def addNoise():
    addNoisedAudios(0.01)


def prepare():
    return prepareData(voiceToSpectograms([NOISE_VOICE_DIRECTORY, VOICE_DIRECTORY]))


def train():
    X_train, X_test, y_train, y_test = prepare()
    trainModel(X_train, X_test, y_train, y_test)

    model = keras.models.load_model('model.h5')
    print(model.evaluate(X_test, y_test)[1])


if __name__ == '__main__':
    train()
    model = keras.models.load_model('model.h5')
