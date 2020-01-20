import os, sys
import numpy as np
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Tools'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Tools/labels'))
from label2OneHotLabel import label2OneHotLabel
from sklearn.neighbors import KNeighborsClassifier
from fillinMatrix import fillinMatrix


def classification(trainData, trainLabels, testData, SETTINGS):
    """Train model with trainData and trainLabels, then predict testLabels given testData.
    Output one hot representation and probability

    Parameters
    ----------
        trainingData:               dataFrame
        trainLabels:                dataFrame
        testData:                   dataFrame
        SETTINGS:                   class

    Return
    ------
        oneHotDf:                   dataFrame
        probaDf:                    dataFrame

    """
    method = SETTINGS.CLASSIFIER
    nClass = SETTINGS.CLASSES
    classLabels = SETTINGS.CLASSLABELS
    verbose = SETTINGS.VERBOSE_LEVEL

    if verbose >= 2:
        print('  -> Classification ' + method)

    trainData = trainData.values
    trainLabels = trainLabels.values.ravel()
    trainLabelsUnqArr = np.unique(trainLabels)

    if method == 'NaiveBayes':

        pass

    elif method == 'knnVoting':

        classifier = KNeighborsClassifier(5)
        model = classifier.fit(trainData, trainLabels)

        testData = testData.values

        result = model.predict(testData)
        proba = model.predict_proba(testData)
        proba = fillinMatrix(proba, trainLabelsUnqArr-1, nClass)

        labelDf = pd.DataFrame(data=result, columns=['Label'])
        try:
            probaDf = pd.DataFrame(data=proba, columns=classLabels)
        except ValueError:

            if verbose >= 2:
                print ('Label:')
                print (labelDf)
                print ('proba:')
                print (proba)
                print (proba.shape)

            raise ValueError(
                "Function classification does not implement properly")

        oneHotDf = label2OneHotLabel(labelDf, classLabels)

    return oneHotDf, probaDf