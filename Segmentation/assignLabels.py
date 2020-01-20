import numpy as np
import pandas as pd
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Tools/labels'))
from segments2label import segments2label
from .majorityVote import majorityVote


def assignLabels(segments, segmentation): 
    """Assign labels to segmentations

    Parameters
    ----------
        segments:               dataFrame
        segmentation:           dataFrame

    Return
    ------        
        labels:                 dataFrame
        
    """
    labelsOrig = segments2label(segments)
    labels = pd.DataFrame(data = np.ones(len(segmentation)), columns = ["Label"], dtype = int)# label 1 is doing nothing by default

    for i in range(len(segmentation)):
        labels["Label"].iloc[i] = majorityVote(labelsOrig["Label"].iloc[segmentation["Start"].iloc[i]:segmentation["End"].iloc[i]].values)

    return labels


if __name__ == "__main__":

    inFile = 'data/segments.csv'
    outFile = 'data/labels_assigned3.csv'

    labels = pd.read_csv(inFile).astype(int) # names = ['Start',   'End',  'Length',  'Label',  'Count']
    segmentation = pd.read_csv('data/segmentation.csv', names = ['Start','End']).astype(int)
    segments = assignLabels(labels, segmentation)
    segments.to_csv(outFile, index = None, header=False)