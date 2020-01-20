import os

import pandas as pd

from Preprocessing.splitIntoRepetitions import splitIntoRepetitions
from Tools.labels.label2segments import label2segments


class Dataset:
    """
    A class for timeseries datasets.


    raw = {"PX": {
            "data": [[]],
            "labels": [[]]} }

    segmented = {"PX": {
                    "segments": [[]],
                    "labels": [[]],
                    "features": [[]]} }

    folds = {"train": [[]],  "test": [[]], "type": ""}

    """
    raw = {}
    segmented = {}
    folds = {"train": [], "test": [], "type": ""}

    def get_participants(self):
        return list(self.raw.keys())

    def set_participant_raw_data(self, p, raw_data):
        """ Adding a participant raw data. """
        if ~(p in self.raw.keys()):
            self.raw[p] = {}
        self.raw[p]["data"] = raw_data

    def get_participant_raw_data(self, p):
        """ """
        return self.raw[p]["data"]

    def set_participant_raw_labels(self, p, raw_labels):
        """ Adding a participant raw labels. """
        self.raw[p]["labels"] = raw_labels

    def get_participant_raw_labels(self, p):
        """ . """
        return self.raw[p]["labels"]

    def set_participant_segmentation_data(self, p, segments):
        """ """
        if ~(p in self.segmented.keys()):
            self.segmented[p] = {}

        self.segmented[p]["segments"] = segments

    def get_participant_segmentation_data(self, p):
        """ """
        return self.segmented[p]["segments"]

    def set_participant_segmentation_labels(self, p, labels):
        """ """
        self.segmented[p]["labels"] = labels

    def get_participant_segmentation_labels(self, p):
        """ """
        return self.segmented[p]["labels"]

    def set_participant_segments_features(self, p, features):
        """ """
        self.segmented[p]["features"] = features

    def get_participant_segments_features(self, p):
        """ """
        return self.segmented[p]["features"]

    #todo change to eval type
    def set_folds(self, fold_type="pi"):
        """

        """
        self.folds["train"] = []
        self.folds["test"] = []
        if fold_type == "pi":
            self.folds["type"] = "pi"
            for p in self.segmented.keys():
                others = list(self.segmented.keys())
                others.remove(p)
                self.folds["train"] = self.folds["train"] + [others]
                self.folds["test"] = self.folds["test"] + [[p]]
        if fold_type == "pd":
            print("not implemented yet")

        return self.folds

    def get_fold_data(self, foldi):
        """

        """
        train_data = []
        train_labels = []
        test_data = []
        test_labels = []
        test_segments = []

        if self.folds["type"] == "pi":
            for p in self.folds["train"][foldi]:
                train_data += self.segmented[p]["features"]
                train_labels += self.segmented[p]["labels"]

            for p in self.folds["test"][foldi]:
                test_data += self.segmented[p]["features"]
                test_labels += self.segmented[p]["labels"]
                test_segments += self.segmented[p]["segments"]

        return pd.concat(train_data), pd.concat(train_labels), pd.concat(test_data), pd.concat(test_labels), \
               pd.concat(test_segments)

    def __repr__(self):
        msg = "raw\nsegmented\nfolds"
        return msg


def Bulling_dataloader(SETTINGS):
    Bulling = Dataset()

    for iSubject in SETTINGS.SUBJECT_LIST:

        dataFileName = "../" + SETTINGS.PATH_DATA + '/subject' + str(iSubject) + '_' + SETTINGS.DATASET + '_data.csv'
        labelsFileName = "../" + SETTINGS.PATH_DATA + '/subject' + str(
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

        data, labels, segments, nRepetitions = splitIntoRepetitions(data, labels, label2segments(labels))

        Bulling.set_participant_raw_data(str(iSubject), data)
        Bulling.set_participant_raw_labels(str(iSubject), labels)

    return Bulling


if __name__ == '__main__':
    from old.mysettings import setting1

    Bulling = Bulling_dataloader(setting1)
    # folds = Bulling.set_folds(fold_type="pi")
    # train_data, train_labels, test_data, test_labels = Bulling.get_fold_data(0)
    # merged_segments = Bulling.merge_participants_segments(["1"])
