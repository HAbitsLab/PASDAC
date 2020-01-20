import numpy as np

def fillinMatrix(smallArr, colIndicesArr, nCols):
    nRows = smallArr.shape[0]
    fullArr = np.zeros([nRows, nCols])

    for i in range(colIndicesArr.size):
        fullArr[:,colIndicesArr[i]] = smallArr[:,i]

    return fullArr
