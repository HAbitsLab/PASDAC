import numpy as np
import pandas as pd

def segmentSlidingWindow(data, wSize, sSize):
    """Sliding window algorithm realization Output 'segments' contains start and end indexes for each step

    Parameters
    ----------
    data:           numpy array
    wSize:          int
    sSize:          int

    Return
    ------
    segments:       dataFrame
    
    """
    if not ((type(wSize) == type(0)) and (type(sSize) == type(0))):
        raise Exception("**ERROR** type(winSize) and type(sSize) must be int.")
    if sSize > wSize:
        raise Exception("**ERROR** step Size must not be larger than window Size.")
    if wSize > len(data):
        raise Exception("**ERROR** window Size must not be larger than data sequence length.")


    length = len(data)  # size of the data input
    wCurr = 0
    segments = []  # initialize segments
    while (wCurr < length - wSize):
        segments.append(np.array([wCurr, wCurr + wSize]))  # appending start and end indexes to segments numpy array
        wCurr = wCurr + sSize

    residual = np.array([wCurr, length])
    segments.append(residual)

    segments = pd.DataFrame(data = segments, columns = ['Start', 'End'])

    return segments
        

if __name__ == '__main__':

    print(segmentSlidingWindow(np.arange(20),5,2))