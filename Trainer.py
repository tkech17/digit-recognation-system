from keras import optimizers
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.models import Sequential

from utils.FileFormatHelper import voiceToSpectograms
from utils.dataSplitHelper import prepareData

VOICE_DIRECTORY = "recordings/"
NOISE_VOICE_DIRECTORY = "noise_voices/"


def trainModel(X_train, X_test, y_train, y_test):
    data_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])

    model2 = getModel(data_shape)
    model2.summary()
    model2.fit(X_train, y_train, batch_size=64, epochs=100, verbose=1, validation_data=(X_test, y_test))
    model2.save('model.h5')


def getModel(data_shape):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same', input_shape=data_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(6, activation='softmax'))

    adam = optimizers.rmsprop(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    return model


def prepare():
    return prepareData(voiceToSpectograms([NOISE_VOICE_DIRECTORY, VOICE_DIRECTORY]))


def train():
    X_train, X_test, y_train, y_test = prepare()
    trainModel(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    train()
