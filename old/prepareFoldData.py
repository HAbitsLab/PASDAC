import sys, os
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), 'Tools/labels'))
from label2segments import label2segments
from old.runPreprocess import runPreprocess


def prepareFoldData(SETTINGS):

    """Prepare fold data for training and classification.

    Parameters
    ----------
        SETTINGS:               struct

    return:
    -------
        features:               list of dataFrames
        fType:                  list
        fDescr:                 list
        segments:               list of dataFrames
        segmentations:          list of dataFrames
        labelsSegmentations:    list of dataFrames
        SETTINGS:               struct 
        
    """

    if SETTINGS.EVALUATION == 'pd':
        # define data file: STUDY/subject/device/sensor/hour(day)
        # define label file: STUDY/subject/device/sensor/hour(day)

        dataFileName = SETTINGS.PATH_DATA+'/subject'+str(SETTINGS.SUBJECT)+'_'+SETTINGS.DATASET+'_data.csv'
        labelsFileName = SETTINGS.PATH_DATA+'/subject'+str(SETTINGS.SUBJECT)+'_'+SETTINGS.DATASET+'_labels.csv'
        
        print(dataFileName)
        print(labelsFileName)

        if SETTINGS.VERBOSE_LEVEL >= 2:
            print('Loading dataset ', dataFileName)
            print('Loading dataset ', labelsFileName)

        # import data file
        if os.path.isfile(dataFileName):
            data = pd.read_csv(dataFileName, names = SETTINGS.SENSORS_AVAILABLE)
        else:
            print('prepareFoldData:fileDoesNotExist, '+dataFileName+' does not exist in the file system.')

        # import label file
        if os.path.isfile(labelsFileName):
            labels = pd.read_csv(labelsFileName, names = ['Label']).astype(int)
        else:
            print('prepareFoldData:fileDoesNotExist, '+labelsFileName+' does not exist in the file system.')

        # features, segments, segmentation, labelsSegmentation are lists of dataFrames
        features, fType, fDescr, segments, segmentation, labelsSegmentation = runPreprocess(data, labels, label2segments(labels), SETTINGS)

        return features, fType, fDescr, segments, segmentation, labelsSegmentation, SETTINGS 


    elif SETTINGS.EVALUATION == 'pi':

        subjectsFeatures = []
        subjectsSegments = []
        subjectsSegmentation = []
        subjectsLabelsSegmentation = []
        subfolds = []

        for iSubject in range(SETTINGS.SUBJECT_TOTAL):

            dataFileName = SETTINGS.PATH_DATA+'/subject'+str(iSubject+1)+'_'+SETTINGS.DATASET+'_data.csv'
            labelsFileName = SETTINGS.PATH_DATA+'/subject'+str(iSubject+1)+'_'+SETTINGS.DATASET+'_labels.csv'

            print(dataFileName)
            print(labelsFileName)

            if SETTINGS.VERBOSE_LEVEL >= 2:
                print('Loading dataset ', dataFileName)
                print('Loading dataset ', labelsFileName)

            # import data file
            if os.path.isfile(dataFileName):
                data = pd.read_csv(dataFileName, names = SETTINGS.SENSORS_AVAILABLE)
            else:
                print('prepareFoldData:fileDoesNotExist, '+dataFileName+' does not exist in the file system.')

            # import label file
            if os.path.isfile(labelsFileName):
                labels = pd.read_csv(labelsFileName, names = ['Label']).astype(int)
            else:
                print('prepareFoldData:fileDoesNotExist, '+labelsFileName+' does not exist in the file system.')

            features, fType, fDescr, segments, segmentation, labelsSegmentation = runPreprocess(data, labels, label2segments(labels), SETTINGS)
            subjectsFeatures.append(features)
            subjectsSegments.append(segments)
            subjectsSegmentation.append(segmentation)
            subjectsLabelsSegmentation.append(labelsSegmentation)

            subfolds.append(list(range(len(subjectsFeatures[iSubject]))))

            # unify segmentation offset somwhere
            for s in range(1,len(subjectsSegments[iSubject])):
                subjectsSegments[iSubject][s].Start = subjectsSegments[iSubject][s-1]['End'].iloc[-1] + subjectsSegments[iSubject][s].Start
                subjectsSegments[iSubject][s].End = subjectsSegments[iSubject][s-1]['End'].iloc[-1] + subjectsSegments[iSubject][s].End
            
            for s in range(1,len(subjectsSegmentation[iSubject])):
                subjectsSegmentation[iSubject][s].Start = subjectsSegmentation[iSubject][s-1]['End'].iloc[-1] + subjectsSegmentation[iSubject][s].Start
                subjectsSegmentation[iSubject][s].End = subjectsSegmentation[iSubject][s-1]['End'].iloc[-1] + subjectsSegmentation[iSubject][s].End

        features = [item for sub in subjectsFeatures for item in sub]
        segments = [item for sub in subjectsSegments for item in sub]
        segmentation = [item for sub in subjectsSegmentation for item in sub]
        labelsSegmentation = [item for sub in subjectsLabelsSegmentation for item in sub]


        for s in range(1,len(subfolds)): # update subfold indices
            for t in range(len(subfolds[s])):
                subfolds[s][t] = subfolds[s-1][-1] + subfolds[s][t] + 1

        SETTINGS.FOLDS = subfolds

        return features, fType, fDescr, segments, segmentation, labelsSegmentation, SETTINGS
