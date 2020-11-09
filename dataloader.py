import sys, os
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), 'Tools/labels'))
from Preprocessing.splitIntoRepetitions import splitIntoRepetitions
from label2segments import label2segments
from Dataset.Dataset import Dataset
from label_time_series import label_time_series


def Bulling_dataloader(SETTINGS):

    Bulling = Dataset()
    for iSubject in SETTINGS.SUBJECT_LIST:
        if SETTINGS.DATASET == "chew":
          dataFileName = SETTINGS.PATH_DATA + '/data/' + str(iSubject)
          labelsFileName = SETTINGS.PATH_DATA + '/data/ANNOTATION/' + str(
            iSubject) + '/CHEWING/chewing.csv'
          chewing_csv = SETTINGS.PATH_DATA + '/data/ANNOTATION/' + str(
            iSubject) + '/CHEWING/chewing.csv'
          inclusion_csv = SETTINGS.PATH_DATA + '/data/ANNOTATION/' + str(
            iSubject) + '/CHEWING/inclusion.csv'
        else:
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
        elif os.path.isdir(dataFileName): # load directory of csvs
            data_frames = []
            ext = ".csv"
            csv_files = [os.path.join(dataFileName, f) for f in os.listdir(dataFileName) if os.path.isfile(os.path.join(dataFileName,f)) and f.endswith(ext)]
            for filename in csv_files:
              data_frames.append(pd.read_csv(filename))
            data = pd.concat(data_frames, ignore_index=True)
        else:
            print('prepareFoldData:fileDoesNotExist, ' + dataFileName + ' does not exist in the file system.')
        # import label file
        if SETTINGS.DATASET == "chew":
            if os.path.isfile(labelsFileName):
                labels = pd.read_csv(labelsFileName)
            else:
                print('prepareFoldData:fileDoesNotExist, ' + labelsFileName + ' does not exist in the file system.')
        else:
            if os.path.isfile(labelsFileName):
                labels = pd.read_csv(labelsFileName, names=['Label']).astype(int)
            else:
                print('prepareFoldData:fileDoesNotExist, ' + labelsFileName + ' does not exist in the file system.')

        if SETTINGS.DATASET == "chew":
            chewing_df = pd.read_csv(chewing_csv)
            inclusion_df = pd.read_csv(inclusion_csv)

            data, labels = label_time_series(iSubject, chewing_df, data, inclusion_df)

            data, point_labels, labels, nRepetitions = splitIntoRepetitions(data, labels, label2segments(labels))

            for i in range(len(data)):
                data[i] = data[i].reset_index(drop=True)
                labels[i] = labels[i].reset_index(drop=True)
                labels[i] = labels[i].drop(columns=["Length", "Count"])

            Bulling.set_participant_raw_data(str(iSubject), data)
            Bulling.set_participant_raw_labels(str(iSubject), labels)
        else:
            data, point_labels, labels, nRepetitions = splitIntoRepetitions(data, labels, label2segments(labels))

            for i in range(len(data)):
                data[i] = data[i].reset_index(drop=True)
                labels[i] = labels[i].reset_index(drop=True)
                labels[i] = labels[i].drop(columns=["Length", "Count"])

            Bulling.set_participant_raw_data(str(iSubject), data)
            Bulling.set_participant_raw_labels(str(iSubject), labels)

    return Bulling

if __name__ == '__main__':
    from old.mysettings import setting1
    Bulling = Bulling_dataloader(setting1)

