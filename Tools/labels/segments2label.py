import pandas as pd
import numpy as np

def segments2label(segments):
    
    """ Transform label representation to segment representation

    Parameters
    ----------
        segments:   dataFrame
                    with 'Start', 'End', 'Length', 'Label', 'Counter' as headers
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

    labeling = pd.DataFrame(data = np.zeros(totalsize), columns = ["Label"] ,dtype = int)
    for index, seg in segments.iterrows():
        labeling["Label"].iloc[seg['Start']-1 : seg['End']] = seg['Label']

    labeling = labeling.iloc[:totalsize] # restrict to totalsize

    return labeling


if __name__ == "__main__":

    inFile = 'data/segments.csv'
    outFile = 'data/subject1_gesture_labels_s2l.csv'

    labels = pd.read_csv(inFile).astype(int) #, names = ['Start',   'End',  'Length',  'Label',  'Count']
    segments = segments2label(labels)
    segments.to_csv(outFile, index = None, header=False)
