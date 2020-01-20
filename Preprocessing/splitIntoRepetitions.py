import pandas as pd
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Tools/labels'))
from label2segments import label2segments
from collections import Counter

def splitIntoRepetitions(dataAll, labelsAll, segmentsAll):
    """Split data into a list of single repetitions

    Parameters
    ----------
        dataAll:                dataFrame
        labelsAll:              dataFrame
        segmentsAll:            dataFrame
        
    Return
    ------
        features:               list of dataFrames
        split into equal repitions based on smallest ubset of labels
    """
    """endIndices = segmentsAll.loc[segmentsAll['Label'] == labelsAll.max()[0], 'End']
    endIndices = endIndices.reset_index(drop = True)
    startIndices = endIndices.shift().fillna(0).astype(int)"""

    data = []
    labels = []
    segments = []
    """datanew = []
    labelsnew = []
    segmentsnew = []"""
    
    
    """nRepetitions = startIndices.shape[0]"""
    
    sortbyLabel = segmentsAll.sort_values(by=['Label'])
    counts = Counter(list(sortbyLabel['Label']))#count of each label
    
    nRepetitions = min(counts.values())
    currentindex =0
    starts = list(sortbyLabel['Start'])
    ends = list(sortbyLabel['End'])
    for i in counts.keys(): #itterate through each label
        for j in range(nRepetitions):#itterate once for each repetition
            inrepetition = counts[i]//nRepetitions#number of segments with i labels in each repetition
            remainder = counts[i]%nRepetitions
            if j < remainder:
                inrepetition += 1
            for repetition in range(inrepetition):
                s = starts[currentindex]-1  #start time of segment
                e = ends[currentindex]
                if len(data) <= j:#if there is no repetition in the list of dataframes create one otherwise update the list
                    data.append(dataAll.iloc[s:e,:].reset_index(drop = True).astype(int))
                    labels.append(labelsAll.iloc[s:e].reset_index(drop = True).astype(int))
                    sortbyLabel.iloc[currentindex]['Start'] = 1
                    sortbyLabel.iloc[currentindex]['End'] = sortbyLabel.iloc[currentindex]['Length']
                    #firstdf = pd.DataFrame(data=sortbyLabel.iloc[currentindex].values,columns=sortbyLabel.columns)
                    segments.append(sortbyLabel.iloc[currentindex].to_frame().transpose())
                else:
                    data[j] = data[j].append(dataAll.iloc[s:e,:].reset_index(drop = True).astype(int))
                    labels[j] = labels[j].append(labelsAll.iloc[s:e].reset_index(drop = True).astype(int))
                    newstart = segments[j].iloc[-1]['End'] + 1
                    sortbyLabel.iloc[currentindex]['Start'] = newstart
                    sortbyLabel.iloc[currentindex]['End'] = sortbyLabel.iloc[currentindex]['Length'] + newstart
                    segments[j] = segments[j].append(sortbyLabel.iloc[currentindex])
                currentindex+=1
    """for s, e in zip(startIndices, endIndices):
        datanew.append(dataAll.iloc[s:e,:].reset_index(drop = True).astype(int))
        labelsnew.append(labelsAll.iloc[s:e].reset_index(drop = True).astype(int))
        segmentsnew.append(label2segments(labelsAll.iloc[s:e].reset_index(drop = True).astype(int)))"""

    return data, labels, segments, nRepetitions
