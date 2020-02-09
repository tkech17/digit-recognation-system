import keras
from xlwt import Workbook

from utils.FileFormatHelper import voiceToSpectograms
from utils.dataSplitHelper import prepareData2

VOICE_DIRECTORY = "recordings/"
NOISE_VOICE_DIRECTORY = "noise_voices/"


def prepareTestData():
    return prepareData2(voiceToSpectograms(["test_voices/"]))


if __name__ == '__main__':
    model = keras.models.load_model('model.h5')
    x_test, y_train, spectograms = prepareTestData()
    predd = model.predict(x_test)
    sm = 0.0

    wb = Workbook()
    sheet1 = wb.add_sheet('Sheet 1')

    for index, data in enumerate(predd):
        num = int(y_train[index][1] * 1 + y_train[index][2] * 2 + y_train[index][3] * 3 + y_train[index][4] * 4 +
                  y_train[index][5] * 5)
        print(data[num], num)
        sm += data[num]
        sheet1.write(index, 0, spectograms[index].spectogram)
        sheet1.write(index, 1, str(data[num]))

    wb.save('bla.xls')
    print(sm / len(predd))
