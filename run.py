from Plot.plotConfusionMatrix import plotConfusionMatrix
from Plot.plotROC import plotROC
from Tools.class_settings import SETTING
from pipeline import *
import pandas as pd

##########################################################################################
# SET SETTINGS
##########################################################################################
setting = SETTING('Data2R', 'Output', '/feature')
setting.set_SAMPLINGRATE(32)  # sampling rate
setting.set_SUBJECT(1)
setting.set_SUBJECT_LIST([1, 2])
setting.set_SUBJECT_TOTAL(2)
setting.set_DATASET('gesture')
setting.set_SMOOTHING_TECHNIQUE(method='boxcar', winsize=30)
setting.set_FEATURE_TYPE('VerySimple')
setting.set_SAVE(0)
setting.set_PLOT(1)
setting.set_VERBOSE_LEVEL(0)
setting.set_FEATURE_SELECTION('none')
setting.set_FEATURE_SELECTION_OPTIONS(10)
setting.set_FUSION_TYPE('early')
setting.set_CLASSIFIER('knnVoting')
setting.set_CLASSIFIER_OPTIONS('knnVoting')
setting.set_EVALUATION('pi')

##########################################################################################
# DATA PREPARATION
##########################################################################################

# Loading the Dataset object
dataset = dataloading_step(setting)

# loop through participant
participants = dataset.get_participants()
for p in participants:

    sub_data = dataset.get_participant_raw_data(p)  # participants raw data
    nRepetitions = len(sub_data)  # getting the number of reps or days

    # temp variables to hold information extracted from each rep
    features = []
    segmentations = []
    labelsSegmentationRepetitions = []

    # loop reps or days and performing postprocessing, segmentation and feature extractions
    for rep in range(nRepetitions):
        ################### (1) Preprocessing ###################
        data = preprocessing_step(sub_data[rep], setting)

        ################### (2) Segmentation ###################
        labels = dataset.get_participant_raw_labels(p)
        segmentDf, labelsSegmentations = segmentation_step(data, labels[rep], setting)
        segmentations.append(segmentDf)
        labelsSegmentationRepetitions.append(labelsSegmentations)

        ################### (3) Feature extraction ###################
        feat, fType, fDescr = feature_extraction_step(data, segmentations[rep], setting)
        features.append(feat)

    # Saving segmented feature data and labels for each participants
    dataset.set_participant_segmentation_data(p, segmentations)
    dataset.set_participant_segmentation_labels(p, labelsSegmentationRepetitions)
    dataset.set_participant_segments_features(p, features)

# todo: what is this offset for
offset = 0
# creating the cross validation eval related metrics
cv = {
    "scores": [],
    "probas": [],
    "testLabels": [],
    "testSegmentation": []
}



##########################################################################################
# Model Training and Testing
##########################################################################################

################### (4) Preparing training and testing folds ###################
subfolds = prepare_folds_step(dataset, setting)

# loop through each fold
for iFold in range(setting.FOLDS):

    if setting.VERBOSE_LEVEL >= 2:
        print('FOLD ' + str(iFold) + '/' + str(setting.FOLDS))
        print('training on:', subfolds['train'][iFold])
        print('testing on:', subfolds['test'][iFold])
    else:
        pass

    ####################  (5) Getting data of this fold ####################
    trainingData, trainingLabels, testData, testLabels, testSegmentation = get_fold_data_step(iFold, dataset, setting)

    ####################  (5) Fold classification ####################
    # todo: make the headers of scores and probas the same.
    scores, probas = fold_classification_step(trainingData, trainingLabels, testData, setting)

    # saving eval scores this fold
    cv["testLabels"] += [testLabels]
    cv["scores"] += [scores]
    cv["probas"] += [probas]

    # todo: check
    if len(cv["testSegmentation"]) == 0:
        cv["testSegmentation"] += [testSegmentation]
    else:
        cv["testSegmentation"] += [testSegmentation + offset]

    offset_temp = max(testSegmentation['End'].values)

    if offset_temp > offset:
        offset = offset_temp

    if setting.VERBOSE_LEVEL >= 2:
        print('  -> End of fold ', iFold, " eval")

# converting cv metrics from a list of dataframes to to one dataframe
for k in cv.keys():
    cv[k] = pd.concat(cv[k]).reset_index(drop=True)

###################  (5) Evaluation ###################
confusion, scoreEval, scoresTimeseries, labelsTimeseries, finalPredTimeseries = evaluation_step(cv, setting)

if setting.VERBOSE_LEVEL >= 2:
    print(confusion)

#  (6) Plot ROC AUC curve & confusion matrix
# cvTestLabels starts from 1, convert to starting from 0
if setting.PLOT == 1:
    plotROC(cv["testLabels"].values - 1, cv["probas"].values, setting.CLASSES)
    plotConfusionMatrix(confusion)

if setting.SAVE == 1:
    cv["testSegmentation"].to_csv("data/cvTestSegmentation.csv", index=None)
    scoresTimeseries.to_csv("data/scoresTimeseries.csv", index=None)
    cv["testLabels"].to_csv("data/cvTestLabels.csv", index=None)
    labelsTimeseries.to_csv("data/labelsTimeseries.csv", index=None)
    finalPredTimeseries.to_csv("data/finalPredTimeseries.csv", index=None)
