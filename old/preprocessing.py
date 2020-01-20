from Preprocessing.smoothing import smoothing
def preprocessing(SETTINGS):
    dataAll = smoothing(dataAll, SETTINGS.SMOOTHING_TECHNIQUE)
    dataAll.to_csv(SETTINGS.PATH_OUTPUT + '/after_smoothing.csv')