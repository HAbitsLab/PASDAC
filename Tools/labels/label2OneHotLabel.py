import numpy as np
import pandas as pd

def label2OneHotLabel(labelDf, classLabels):

    """ label2OneHotLabel() transforms categorical representation to a format that has one boolean column for each category. 
    Only one of these columns could take on the value 1 for each sample.
    Note: in labelDf, Label starts from 1. Label 0 is not used for any activity. No activity is 1
    
    Parameters
    ----------
        labelDf:        dataFrame
        classLabels:    set
    Return
    ------
        oneHotDf:       dataFrame, eg: header = ['0', '1', ...., str(len(classLabels)-1)]

    """

    oneHot = np.zeros([len(labelDf), len(classLabels)])
    for i in range(len(labelDf)):
        oneHot[i,labelDf.iloc[i].values-1] = 1

    head = []
    for i in range(len(classLabels)):
        head.append(str(i))

    oneHotDf = pd.DataFrame(data = oneHot, columns = head, dtype =int)

    return oneHotDf
