from Segmentation.segment import Segmentation
from Preprocessing.splitIntoRepetitions import splitIntoRepetitions
from Preprocessing.smoothing import smoothing
from Segmentation.assignLabels import assignLabels
from Features.featureExtraction import featureExtraction
import matplotlib.pyplot as plt


def runPreprocess(dataAll, labelsAll, segmentsAll, SETTINGS):

    """ Run preprocess, including smoothing, splitting data, labels and segments into repetitions,
    doing segmentation and feature extraction for each repetition

    Parameters
    ----------
        dataAll:                dataFrame
        labelsAll:              dataFrame
        segmentsAll:            dataFrame
        SETTINGS:               struct

    Return
    ------
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
        
    """
    # (0) Smoothing
    dataAll.to_csv(SETTINGS.PATH_OUTPUT+'/before_smoothing.csv')
    subDataThreshForPlot = 1000
    if SETTINGS.VERBOSE_LEVEL >= 2:
            plt.title('Before smoothing')
            if (dataAll.shape[0] > subDataThreshForPlot):
                plt.plot(dataAll.iloc[0:subDataThreshForPlot])
            else:
                plt.plot(dataAll)
            plt.show()
    dataAll = smoothing(dataAll,SETTINGS.SMOOTHING_TECHNIQUE)
    if SETTINGS.VERBOSE_LEVEL >= 2:
        plt.title('After smoothing')
        if (dataAll.shape[0] > subDataThreshForPlot):
            plt.plot(dataAll.iloc[0:subDataThreshForPlot])
        else:
            plt.plot(dataAll)
        plt.show()
    dataAll.to_csv(SETTINGS.PATH_OUTPUT+'/after_smoothing.csv')

    # (1) Preprocessing
    # Split data set into repetitions
    # data, labels, segments are arrays containing dataframes
    data, labels, segments, nRepetitions = splitIntoRepetitions(dataAll, labelsAll, segmentsAll) 

    features = []
    segmentations = []
    labelsSegmentationRepetitions = []

    # Loop over repetitions
    for i in range(nRepetitions):
        
        if SETTINGS.VERBOSE_LEVEL >= 2:
            print('REPETITION '+str(i)+'/'+str(nRepetitions))
        else:
            pass

        # (2) Segmentation

        segmentationClass = Segmentation()
        segmentationFunction = SETTINGS.SEGMENTATION_TECHNIQUE['method']
        print("Segmentation {}".format(segmentationFunction))
        print ("\n")

        try:
            segmentationMethod = getattr(segmentationClass, segmentationFunction)
        except AttributeError:
            raise NotImplementedError(
                "Class `{}` does not implement `{}`".format(segmentationClass.__class__.__name__,
                                                            segmentationFunction))
        segmentDf = segmentationMethod(data[i].values, SETTINGS)
        segmentations.append(segmentDf)
        labelsSegmentations = assignLabels(segments[i], segmentations[i])
        labelsSegmentationRepetitions.append(labelsSegmentations)

        # (3) Feature extraction
        feat, fType, fDescr = featureExtraction(data[i], segmentations[i], SETTINGS.FEATURE_TYPE, SETTINGS.VERBOSE_LEVEL)
        features.append(feat)
    
    labelsSegmentations = labelsSegmentationRepetitions
    
    return features, fType, fDescr, segments, segmentations, labelsSegmentations 