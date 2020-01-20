from setDiff import setDiff
from array2Dataframe import array2Dataframe

def get_pd(iFold, subfolds, features, labelsSegmentation, segmentation):
    testSet = iFold
    trainingSet = setDiff(list(subfolds), list([iFold]))
    testData = features[testSet]
    testLabels = labelsSegmentation[testSet]  # output testLabels is dataframe
    testSegmentation = segmentation[testSet]

    #  (4.1) Create training and test data sets
    features_selected = []
    for i in trainingSet:
        features_selected.append(features[i])
    trainingData = array2Dataframe(features_selected)

    labels_selected = []
    for i in trainingSet:
        labels_selected.append(labelsSegmentation[i])
    trainingLabels = array2Dataframe(labels_selected)

    return testData, testLabels, testSegmentation, trainingData, trainingLabels


def get_pi(iFold, subfolds, features, labelsSegmentation, segmentation):
    testSet = iFold
    testSet = subfolds[testSet]
    trainingSet = setDiff([item for sub in subfolds for item in sub], testSet)
    testData = [features[i] for i in testSet]
    testData = array2Dataframe(testData)

    #  (4.1) Create training and test data sets
    features_selected = []
    for i in trainingSet:
        features_selected.append(features[i])
    trainingData = array2Dataframe(features_selected)

    labels_selected = []
    for i in trainingSet:
        labels_selected.append(labelsSegmentation[i])
    trainingLabels = array2Dataframe(labels_selected)

    testLabels = [labelsSegmentation[i] for i in testSet]  # output testLabels is dataframe
    testLabels = array2Dataframe(testLabels)
    testSegmentation = [segmentation[i] for i in testSet]
    testSegmentation = array2Dataframe(testSegmentation)

    return testData, testLabels, testSegmentation, trainingData, trainingLabels



