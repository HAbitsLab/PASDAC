import numpy as np
import pandas as pd
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Tools/labels'))
# from segments2label import segments2label
from .majorityVote import majorityVote


def segments2label(segments):
    """ Transform label representation to segment representation

    Parameters
    ----------
        segments:   dataFrame
                    with 'Start', 'End', 'Label' as headers
    return
    ------
        labeling:   dataFrame
                    with 'Label' as headers
    """

    labeling = [];

    if len(segments) == 0:
        return

    totalsize = segments["End"].iloc[-1]

    # segments contains labelList in the 4th column
    if (len(segments.columns) >= 4):
        labels = segments["Label"].values
    else:
        # label 1 is doing nothing by default
        labels = np.ones(len(segments))

    labeling = pd.DataFrame(data=np.zeros(totalsize), columns=["Label"], dtype=int)
    for index, seg in segments.iterrows():
        labeling["Label"].iloc[seg['Start'] - 1: seg['End']] = seg['Label']

    labeling = labeling.iloc[:totalsize]  # restrict to totalsize

    return labeling


def assignLabels(labels, segmentation, raw=True, method='majorityVote'):
    """Assign labels to segmentations

    Parameters
    ----------
        labels : dataFrame
            start and end of raw labels or labels as timeseries 
        segmentation : dataFrame
            start and end of each segment
        raw: boolean
            True of Raw, False if is as timeseries


    Return
    ------        
        seg_labels : dataFrame
        
    """

    if raw:
        # Getting the raw label of each point from the raw labels
        points_labels = segments2label(labels)
    else:
        points_labels = labels
    # initializing the seg_labels dataframe with 1 (assuming that 1 is the null value)
    seg_labels = pd.DataFrame(data=np.ones(segmentation.shape[0]), columns=segmentation.columns[2:],
                              dtype=int)  # label 1 is doing nothing by default

    # Looping though each segment to get the label
    for i in range(len(segmentation)):
        
        if method == 'majorityVote':
            # label is assigned based on the majority vote of all the labels of each point in the segment.
            seg_labels["Label"].iloc[i] = majorityVote(
                points_labels["Label"].iloc[segmentation["Start"].iloc[i]:segmentation["End"].iloc[i]].values)
        elif method == 'any':
            # In a binary classification a window will be 1 if 1 truth point exist
            for c in seg_labels.columns:
                seg_labels[c].iloc[i] = points_labels[c].iloc[segmentation["Start"].iloc[i]:segmentation["End"].iloc[i]].values.max()

    return seg_labels


if __name__ == "__main__":
    inFile = 'data/segments.csv'
    outFile = 'data/labels_assigned3.csv'

    labels = pd.read_csv(inFile).astype(int)  # names = ['Start',   'End',  'Length',  'Label',  'Count']
    segmentation = pd.read_csv('data/segmentation.csv', names=['Start', 'End']).astype(int)
    segments = assignLabels(labels, segmentation)
    segments.to_csv(outFile, index=None, header=False)
