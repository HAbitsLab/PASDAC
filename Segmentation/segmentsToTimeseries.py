import pandas as pd
import numpy as np


def segmentsToTimeseries(segments,value, fillup):

    """ convert from segments and value to time series points

    Parameters
    ----------
        segments:               dataFrame
        value:                  dataFrame

    Return
    ------
        timeseries:             dataFrame
    
    Example
    -------
         segments:
        Start   End
        0   32
        3   35
        6   38


         value:
        0   1   2   3   4   5   6   7   8   9   10  11
        1   0   0   0   0   0   0   0   0   0   0   0
        0   0   1   0   0   0   0   0   0   0   0   0

         Return timeseries:
        0   1   2   3   4   5   6   7   8   9   10  11
        1   0   0   0   0   0   0   0   0   0   0   0
        1   0   0   0   0   0   0   0   0   0   0   0
        1   0   0   0   0   0   0   0   0   0   0   0
        1   0   1   0   0   0   0   0   0   0   0   0
        1   0   1   0   0   0   0   0   0   0   0   0

    """

    timeseries = np.zeros([max(segments['End'].values), len(list(value.keys()))]) - float('Inf')

    for s in range(len(segments)):
        start = segments['Start'].iloc[s]
        stop = segments['End'].iloc[s]

        v = value.iloc[s].values
        assert(stop>=start)
        candidate1 = np.vstack([v]*(stop-start))
        candidate2 = timeseries[start:stop,:]

        # np.maximum will compare two matrix element by element.
        #   eg: np.maximum([[1,2,3,4],[1,2,3,4]],[[4,3,2,1],[4,3,2,1]]) -->  array([[4, 3, 3, 4],[4, 3, 3, 4]])
        
        timeseries[start:stop,:] = np.maximum(candidate1, candidate2)
    
    timeseries[timeseries==-float('Inf')] = fillup
    ts = pd.DataFrame(data = timeseries, columns = list(value.keys()))
    
    return ts


if __name__ == "__main__":

    value = pd.read_csv("../data/cvScores.csv")
    seg = pd.read_csv("../data/cvTestSegmentation.csv")

    print(value)
    print(seg)

    print(segmentsToTimeseries(seg, value, -float('Inf')))
    segmentsToTimeseries(seg, value, -float('Inf')).to_csv("../data/timeseries.csv", index =  None)