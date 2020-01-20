import pandas as pd
import numpy as np


def label2segments(labels):

    """ Transform label representation to segment representation

    Parameters
    ----------
        labels:     dataFrame
                    with 'Label' as headers
    return
    ------
        segments:   dataFrame
                    with 'Start', 'End', 'Length', 'Label', 'Counter' as headers
    """

    labelsShift = pd.DataFrame(np.array([[0]]), columns=['Label']).append(labels.iloc[:-1])
    labelsShift = labelsShift.reset_index(drop=True)

    # find the start point of each segment
    diff = labels - labelsShift
    diff = diff.loc[diff['Label'] != 0]
    segStarts = diff.index.tolist()

    # add the first point if not included
    if 0 not in segStarts:
        segStarts = [0] + segStarts

    # generate data frame segments
    segments = pd.DataFrame((np.array(segStarts).transpose() + 1), index=None, columns=['Start'])
    segments['End'] = np.append(segments['Start'].values[1:] - 1, len(labels))
    segments['Length'] = segments['End'] - segments['Start'] + 1
    segments['Label'] = labels.iloc[np.array(segStarts)].values
    segments['Count'] = 0

    # add column 'Count' to segments
    labelArray = segments['Label'].values

    for labelX in set(labelArray):
        labelXind = np.where(labelArray == labelX)[0]
        cnt = 0
        for i in range(len(labelXind)):
            cnt = cnt + 1
            segments['Count'].iloc[labelXind[i]] = cnt
    return segments


if __name__ == "__main__":

    inFile = 'data/subject1_gesture_labels.csv'
    outFile = 'data/segments.csv'

    labels = pd.read_csv(inFile, names = ['Label']).astype(int)
    segments = label2segments(labels)
    segments.to_csv(outFile, index = None)
