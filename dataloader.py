import sys, os
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), 'Tools/labels'))
from Preprocessing.splitIntoRepetitions import splitIntoRepetitions
from label2segments import label2segments
from Dataset.Dataset import Dataset

def Bulling_dataloader(SETTINGS):

    Bulling = Dataset()

    for iSubject in SETTINGS.SUBJECT_LIST:

        dataFileName = SETTINGS.PATH_DATA + '/subject' + str(iSubject) + '_' + SETTINGS.DATASET + '_data.csv'
        labelsFileName = SETTINGS.PATH_DATA + '/subject' + str(
            iSubject) + '_' + SETTINGS.DATASET + '_labels.csv'

        print(dataFileName)
        print(labelsFileName)

        if SETTINGS.VERBOSE_LEVEL >= 2:
            print('Loading dataset ', dataFileName)
            print('Loading dataset ', labelsFileName)

        # import data file
        if os.path.isfile(dataFileName):
            data = pd.read_csv(dataFileName, names=SETTINGS.SENSORS_AVAILABLE)
        else:
            print('prepareFoldData:fileDoesNotExist, ' + dataFileName + ' does not exist in the file system.')

        # import label file
        if os.path.isfile(labelsFileName):
            labels = pd.read_csv(labelsFileName, names=['Label']).astype(int)
        else:
            print('prepareFoldData:fileDoesNotExist, ' + labelsFileName + ' does not exist in the file system.')

        data, point_labels, labels, nRepetitions = splitIntoRepetitions(data, labels, label2segments(labels))

        Bulling.set_participant_raw_data(str(iSubject), data)
        Bulling.set_participant_raw_labels(str(iSubject), labels)

    return Bulling

if __name__ == '__main__':
    from old.mysettings import setting1
    Bulling = Bulling_dataloader(setting1)

