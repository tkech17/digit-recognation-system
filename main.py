import keras

from Trainer import trainModel
from utils.FileFormatHelper import voiceToSpectograms
from utils.dataSplitHelper import prepareData


def addNoise():
    pass
    # addNoisedAudios(0.01)
    # addNoisedAudios(0.01)


def prepare():
    return prepareData(voiceToSpectograms())


def train():
    X_train, X_test, y_train, y_test = prepare()
    trainModel(X_train, X_test, y_train, y_test)

    model = keras.models.load_model('model.h5')
    print(model.evaluate(X_test , y_test)[1])

if __name__ == '__main__':
    addNoise()
    train()
    model = keras.models.load_model('model.h5')
