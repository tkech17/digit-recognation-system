# %%

import numpy as np
from keras.preprocessing import image
from keras.utils.np_utils import to_categorical

from utils.FileFormatHelper import removeImagesDirecotory


def prepareData2(spectograms: list):
    trainset = []
    testset = []
    for index, spectogramData in enumerate(spectograms):
        trainset.append([image.img_to_array(image.load_img(spectogramData.spectogram)), spectogramData.label])

    X_test, X_train, y_test, y_train = getTraindAndTestSets(testset, trainset)
    X_test, X_train, y_test, y_train = convertTrainAndTestSetsIntoArrays(X_test, X_train, y_test, y_train)

    y_train = to_categorical(y_train)

    X_train /= 255
    X_test /= 255

    removeImagesDirecotory()

    return X_train, y_train, spectograms


def prepareData(spectograms: list):
    trainset = []
    testset = []
    for index, spectogramData in enumerate(spectograms):
        if index % 10 == 0:
            testset.append([image.img_to_array(image.load_img(spectogramData.spectogram)), spectogramData.label])
        else:
            trainset.append([image.img_to_array(image.load_img(spectogramData.spectogram)), spectogramData.label])

    X_test, X_train, y_test, y_train = getTraindAndTestSets(testset, trainset)
    X_test, X_train, y_test, y_train = convertTrainAndTestSetsIntoArrays(X_test, X_train, y_test, y_train)

    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    X_train /= 255
    X_test /= 255

    removeImagesDirecotory()

    return X_train, X_test, y_train, y_test


def convertTrainAndTestSetsIntoArrays(X_test, X_train, y_test, y_train):
    X_train = np.asanyarray(X_train)
    y_train = np.asanyarray(y_train)
    X_test = np.asanyarray(X_test)
    y_test = np.asanyarray(y_test)
    return X_test, X_train, y_test, y_train


def getTraindAndTestSets(testset, trainset):
    X_train = [item[0] for item in trainset]
    y_train = [item[1] for item in trainset]
    X_test = [item[0] for item in testset]
    y_test = [item[1] for item in testset]
    return X_test, X_train, y_test, y_train
