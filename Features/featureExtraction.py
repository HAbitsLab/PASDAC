import pandas as pd
from .calculateFeaturesSimple import calculateFeaturesSimple
from .calculateFeaturesVerySimple import calculateFeaturesVerySimple

def featureExtraction(data, segmentations, featureType, verbose):

    """ Extract features

    Parameters
    ----------
        data:                   dataFrame
        segmentations:          dataFrame
        featureType:            string
        verbose:                int

    Return
    ------
        featuresDf:             dataFrames
        fType:                  list
                                eg: ['Mean_acc_1_x', 'Variance_acc_1_x', 'Mean_acc_1_y', 'Variance_acc_1_y',
                                    ....
                                   'Mean_gyr_3_y', 'Variance_gyr_3_y']
        fDescr:                 list
                                eg: ['Mean', 'Variance']
    """

    if verbose >= 2:
        print('  -> Feature extraction ', featureType)

    if featureType == 'Simple':
        # get description and size to init (speed)
        fDescr = ['Mean', 'Variance', 'ZCR', 'MCR']
        sensorList = list(data.keys())
        fType = []
        

        for s in sensorList:
            for f in fDescr:
                fType.extend([f + "_" + s])

        allfeats = []

        for i in range(len(segmentations)):
            start = segmentations["Start"].iloc[i]
            end = segmentations["End"].iloc[i]
            features = []
            
            for s in sensorList:
                f = calculateFeaturesSimple(data[s].iloc[start:end].as_matrix())
                features.extend(f)

            allfeats.append(features)

        featuresDf = pd.DataFrame(data = allfeats, columns = fType)
        

    elif featureType == 'VerySimple':
        fDescr = ['Mean', 'Variance']
        sensorList = list(data.keys())
        fType = []
        

        for s in sensorList:
            for f in fDescr:
                fType.extend([f + "_" + s])

        allfeats = []

        for i in range(len(segmentations)):
            start = segmentations["Start"].iloc[i]
            end = segmentations["End"].iloc[i]
            features = []
            
            for s in sensorList:
                f = calculateFeaturesVerySimple(data[s].iloc[start:end].values)
                features.extend(f)

            allfeats.append(features)

        featuresDf = pd.DataFrame(data = allfeats, columns = fType)

    
    return featuresDf, fDescr


if __name__ == "__main__":

    # inFile = 'data/segments.csv'
    dataFile = 'data/subject1_gesture_data.csv'
    segFile = 'data/segmentation.csv'

    data = pd.read_csv(dataFile, names = ['acc_1_x', 'acc_1_y', 'acc_1_z','gyr_1_x', 'gyr_1_y', 'acc_2_x', 'acc_2_y', 'acc_2_z', 'gyr_2_x', 'gyr_2_y','acc_3_x', 'acc_3_y', 'acc_3_z', 'gyr_3_x', 'gyr_3_y'])
    segmentation = pd.read_csv(segFile, names = ['Start','End'])
    
    print(featureExtraction(data, segmentation, 'Simple', 2))
