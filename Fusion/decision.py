import pandas as pd
import numpy as np


def decision(scoresTimeseries):
    """ decide the final class for each time series data point

    Parameters
    ----------
        scoresTimeseries:                     dataFrame

    Return
    ------
        predictionDf:                         dataFrame

    """

    sArr = scoresTimeseries.values

    prediction = np.argmax(sArr, axis=1) # MAP: argmax[c] (scores) for each sample i

    predictionDf = pd.DataFrame(data = prediction, columns = ["Label"], dtype = int)

    return predictionDf
