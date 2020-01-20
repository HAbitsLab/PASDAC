import sys, os
import pandas as pd
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), 'Classification'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'Evaluation'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'Fusion'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'Plot'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'Preprocessing'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'Segmentation'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'Tools'))
from classification import classification
from lenArrayOfDataframe import lenArrayOfDataframe
from setDiff import setDiff
from array2Dataframe import array2Dataframe
from segmentsToTimeseries import segmentsToTimeseries
from decision import decision
from evaluation import evaluation
from standardizeData import standardizeData
from plotROC import plotROC
from plotConfusionMatrix import plotConfusionMatrix


def runEvaluation(features, fType, fDescr, segments, segmentation, labelsSegmentation, SETTINGS):

    """ Run Evaluation: create training and test data sets, standardize training and test data sets, train classifier, 
    classify, evaluate and plot.

    Parameters
    ----------
        features:               list of dataFrames
        fType:                  list
                                eg: ['Mean_acc_1_x', 'Variance_acc_1_x', 'Mean_acc_1_y', 'Variance_acc_1_y',
                                    ....
                                   'Mean_gyr_3_y', 'Variance_gyr_3_y']
        fDescr:                 list
                                eg: ['Mean', 'Variance']
        segments:               list of dataFrames
        segmentations:          list of dataFrames
        labelsSegmentations:    list of dataFrames

    Return
    ------
        confusion:              array
        scoreEval:              dataFrame

    """

    cvScores = np.zeros([lenArrayOfDataframe(labelsSegmentation), SETTINGS.CLASSES])
    cvProbas = np.zeros([lenArrayOfDataframe(labelsSegmentation), SETTINGS.CLASSES])
    cvFeatSelection = []
    offset = 0

    startIndex = 1
    startIndexSegments = 1

    if SETTINGS.EVALUATION == 'pd':
        subfolds = range(SETTINGS.FOLDS)
    elif SETTINGS.EVALUATION == 'pi':
        subfolds = SETTINGS.FOLDS
        SETTINGS.FOLDS = len(subfolds)

    # Training and Classification
    first_run = 1

    for iFold in range(SETTINGS.FOLDS):
        if SETTINGS.VERBOSE_LEVEL >= 2:
            print('FOLD '+str(iFold)+'/'+str(SETTINGS.FOLDS))
        else:
            pass

        testSet = iFold


        if SETTINGS.EVALUATION == 'pi':
            testSet = subfolds[testSet]
            trainingSet = setDiff([item for sub in subfolds for item in sub], testSet)
        elif SETTINGS.EVALUATION == 'pd':
            trainingSet = setDiff(list(subfolds), list([iFold]))

        #  (4.1) Create training and test data sets
        features_selected = []
        for i in trainingSet:
            features_selected.append(features[i])
        trainingData = array2Dataframe(features_selected)

        if SETTINGS.EVALUATION == 'pd':
            testData = features[testSet]
        elif SETTINGS.EVALUATION == 'pi':
            testData = [features[i] for i in testSet]
            testData = array2Dataframe(testData)


        labels_selected = []
        for i in trainingSet:
            labels_selected.append(labelsSegmentation[i])
        trainingLabels = array2Dataframe(labels_selected)

        if SETTINGS.EVALUATION == 'pd':
            testLabels = labelsSegmentation[testSet] # output testLabels is dataframe
            testSegmentation= segmentation[testSet]

        elif SETTINGS.EVALUATION == 'pi':
            testLabels = [labelsSegmentation[i] for i in testSet] # output testLabels is dataframe
            testLabels = array2Dataframe(testLabels)
            testSegmentation= [segmentation[i] for i in testSet]
            testSegmentation = array2Dataframe(testSegmentation)

        #  (4.2) Standardize training and test data sets
        # from N-d array/cell to 2d array/matrix
        trainingData, testData = standardizeData(trainingData, testData, SETTINGS.VERBOSE_LEVEL)

        #  (4.3) Classify
        # trainingData:   dataframe 
        # trainingLabels: dataframe 
        # testData:       dataframe 
        # scores is dataFrame with 'Label' as header, eg:['0', '1', '2', ...]            
        # probas is dataFrame with actvity name as header, eg:['0.7', '0.1', '0.2', ...] 
        scores, probas = classification(trainingData, trainingLabels, testData, SETTINGS)

        #  (4.4) Concatenate fold
        if SETTINGS.VERBOSE_LEVEL >= 2:
            print('  -> Concatenating fold')

        if first_run == 1:
            cvTestLabels = testLabels
            cvScores = scores
            cvProbas = probas
            cvTestSegmentation = testSegmentation
            first_run = 0
        else:
            cvTestLabels = pd.concat([cvTestLabels, testLabels]).reset_index(drop = True)
            cvScores = pd.concat([cvScores, scores]).reset_index(drop = True)
            cvProbas = pd.concat([cvProbas, probas]).reset_index(drop = True)
            cvTestSegmentation = pd.concat([cvTestSegmentation, testSegmentation+offset]).reset_index(drop = True)

        offset = max(cvTestSegmentation['End'].values)

    scores = cvScores
    probas = cvProbas

    # prediction one hot representation matrix,  point by point 
    # type: dataFrame
    scoresTimeseries = segmentsToTimeseries(cvTestSegmentation, scores, -float('Inf'))
    labelsTimeseries = segmentsToTimeseries(cvTestSegmentation, cvTestLabels, -float('Inf')) 

    if SETTINGS.SAVE == 1:
        cvTestSegmentation.to_csv("data/cvTestSegmentation.csv", index = None)
        scoresTimeseries.to_csv("data/scoresTimeseries.csv", index = None)
        # ground truth labels,  point by point 
        # type: Series
        cvTestLabels.to_csv("data/cvTestLabels.csv", index = None)
        # cvTestLabels starts from 1
        labelsTimeseries.to_csv("data/labelsTimeseries.csv", index = None)


    #  (4.5) Convert scores into prediction
    # prediction class result , point by point, 
    # type: Series
    # scoresTimeseries starts from 0
    # finalPredTimeseries starts from 1
    finalPredTimeseries = decision(scoresTimeseries)
    # finalPredTimeseries starts from 1
    finalPredTimeseries = finalPredTimeseries + 1

    if SETTINGS.SAVE == 1:
        finalPredTimeseries.to_csv("data/finalPredTimeseries.csv", index = None)


    #  (5) Evaluation
    confusion, scoreEval = evaluation(finalPredTimeseries, labelsTimeseries, scoresTimeseries, SETTINGS)


    #  (6) Plot ROC AUC curve & confusion matrix
    # cvTestLabels starts from 1, convert to starting from 0
    plotROC(cvTestLabels.values-1, probas.values, SETTINGS.CLASSES)
    plotConfusionMatrix(confusion)
        
    return confusion, scoreEval