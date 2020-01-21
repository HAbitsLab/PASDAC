from Classification.classification import classification
from Evaluation.evaluation import evaluation
from Fusion.decision import decision
from Preprocessing.standardizeData import standardizeData
# from Segmentation.segment import Segmentation
from Preprocessing.smoothing import smoothing
from Segmentation.assignLabels import assignLabels
from Features.featureExtraction import featureExtraction
from Segmentation.segmentsToTimeseries import segmentsToTimeseries
from dataloader import Bulling_dataloader
from Segmentation import segment

def dataloading_step(SETTINGS):
    """

    Parameters
    ----------
    SETTINGS : dict
        The dictionary should contain the following keys. It can be changed based on the dataset
        SETTINGS.PATH_DATA : str
            Name of the folder that contains the data
        SETTINGS.DATASET : str
            Name of the subfolder that contains the data
        SETTINGS.SUBJECT_LIST : list
            List of the subjects to be included from the dataset
        SETTINGS.SENSORS_AVAILABLE : list
            List of the sensors that are included in the
        SETTINGS.VERBOSE_LEVEL : bool
            to print or not to print progress messages


    Returns
    -------
    dataset : Dataset
        Object that contains the data

    """
    dataset = Bulling_dataloader(SETTINGS)
    return dataset


def preprocessing_step(data, SETTINGS):
    """ Signal prepocessing step

    Parameters
    ----------
    data : DataFrame




    """
    data = smoothing(data, SETTINGS.SMOOTHING_TECHNIQUE)
    data.to_csv(SETTINGS.PATH_OUTPUT + '/after_smoothing.csv')
    return data


def segmentation_step(data, labels, SETTINGS):
    """ Segmenting the raw data

    Parameters
    ----------
    data : DataFrame
        raw data
    labels : DataFrame
        raw labels containing the columns "Start", "End", "Label"
    SETTINGS : Dict
        SETTINGS.SEGMENTATION_TECHNIQUE : dict
            method : (str) the name of the segmentation technique implemented in the segmentation class
            ... Any other settings related to the segmentation technique
        SETTINGS.SAMPLINGRATE : int
            the sampling rate of the system
        SETTINGS.VERBOSE_LEVEL: bool
            to print or not to print progress messages

    Returns
    -------
    segmentDf : DataFrame
        index of segmented data containing the columns "Start" and "End"
    labelsSegmentations : DataFrame
        labels of the segments columns should contain at least a column "Label"

    """
    segmentationFunction = SETTINGS.SEGMENTATION_TECHNIQUE['method']
    segmentationMethod = getattr(segment, segmentationFunction)
    print("Segmentation {}".format(segmentationFunction))
    print("\n")
    try:
        segmentationMethod = getattr(segment, segmentationFunction)
    except AttributeError:
        raise NotImplementedError(
            "Function `{}` is not implemented in Segmentation.segment".format(segmentationFunction))
    segmentDf = segmentationMethod(data.values, SETTINGS)
    labelsSegmentations = assignLabels(labels, segmentDf)

    return segmentDf, labelsSegmentations

#todo: change feature extraction to be a class like segments
def feature_extraction_step(data, segmentation, SETTINGS):
    """ Extracting Features from each segment

    Parameters
    ----------
    data : DataFrame
        raw data of the sensors value
    segmentation : DataFrame
        start and end of each segment
    SETTINGS : dict
        SETTINGS.FEATURE_TYPE : str
            the name of the feature extraction method
        SETTINGS.VERBOSE_LEVEL: bool
            to print or not to print progress messages

    Returns
    -------
    feat : DataFrame
        extracted feature from each segment
    fDescr : list
        name of the functions used for extracting feature
    """

    feat, fDescr = featureExtraction(data, segmentation, SETTINGS.FEATURE_TYPE, SETTINGS.VERBOSE_LEVEL)
    return feat, fDescr


def prepare_folds_step(Dataset,SETTINGS):
    """

    """
    subfolds = Dataset.set_eval_type(fold_type=SETTINGS.EVALUATION)
    SETTINGS.FOLDS = len(subfolds["train"])

    return subfolds

def get_fold_data_step(iFold,Dataset,SETTINGS):
    """

    """
    # getting the fold data
    trainingData, trainingLabels, testData, testLabels, testSegmentation = Dataset.get_fold_data(iFold)

    # Standardize training and test data sets
    trainingData, testData = standardizeData(trainingData, testData, SETTINGS.VERBOSE_LEVEL)

    return trainingData, trainingLabels, testData, testLabels, testSegmentation


# todo: split the model building and testing steps
def fold_classification_step(trainingData, trainingLabels, testData,SETTINGS):
    """

    """
    scores, probas = classification(trainingData, trainingLabels, testData, SETTINGS)

    return scores, probas

def evaluation_step(cv,SETTINGS):
    """

    """
    # prediction one hot representation matrix,  point by point
    scoresTimeseries = segmentsToTimeseries(cv["testSegmentation"], cv["scores"], -float('Inf'))
    labelsTimeseries = segmentsToTimeseries(cv["testSegmentation"], cv["testLabels"], -float('Inf'))

    #  (4.5) Convert scores into prediction
    finalPredTimeseries = decision(scoresTimeseries)
    finalPredTimeseries = finalPredTimeseries + 1

    confusion, scoreEval = evaluation(finalPredTimeseries, labelsTimeseries, scoresTimeseries, SETTINGS)

    return confusion, scoreEval, scoresTimeseries, labelsTimeseries, finalPredTimeseries



