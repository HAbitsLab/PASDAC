import pandas as pd
import numpy as np
from scipy.stats import *     

def standardizeData(trainingData, testData, verbose):
    """training set and test set z-normalization without taking use of scipy library

    Parameters
    ----------
        trainingData:           dataFrame
        testData:               dataFrame

    Return
    ------
        trainingZsc:            dataFrame
        testZsc:                dataFrame

    """

    # TODO: Comment in the following 2 lines and implement training set and test set z-normalization 
    # 		according to requirement
    

    if verbose >= 2:
        print('  -> Standardization')

    #  training set z-normalization

    # stats.zscore can take dataframe as input
    trainingZsc =  pd.DataFrame(data = stats.zscore(trainingData), columns = trainingData.keys())

    trainingMean = np.mean(trainingData).values
    trainingStd = np.std(trainingData).values
    testZsc = testData.copy(deep=True)


    for i in range(len(trainingData.keys())):
        testZsc.iloc[:,i] = testZsc.iloc[:,i] - trainingMean[i]
        testZsc.iloc[:,i] = testZsc.iloc[:,i]/trainingStd[i]

    # HACK: to avoid null-division: set stdvar==0 to stdvar=mean
    if 0 in trainingStd:
        ind = np.where((trainingStd==0))[0]
        for i in ind:
            trainingZsc.iloc[:,i] = trainingData.iloc[:,i] - trainingMean[i]
            testZsc.iloc[:,i] = testData.iloc[:,i] - trainingMean[i]  # todo: check with shibo why subtract with train mean



    return trainingZsc, testZsc



if __name__ == "__main__":

    trainingset1 = pd.DataFrame(data = np.zeros(10)+1, columns = ["Label"], dtype = int)
    trainingset1['Label1']= trainingset1['Label']

    testset1 = trainingset1.copy(deep=True)
    testset1['Label1'].iloc[0] = 10


    print("before z-score")
    print('trainingset1')
    print(trainingset1)
    print('testset1')
    print(testset1)

    trainingset1,testset1 = standardizeData(trainingset1, testset1, 2)

    print("after z-score")
    print('trainingset1')
    print(trainingset1)
    print('testset1')
    print(testset1)



    trainingset2 = pd.DataFrame(data = np.zeros(10)+1, columns = ["Label"], dtype = int)
    trainingset2['Label1']= trainingset2['Label']
    trainingset2['Label1'].iloc[0] = 10
    
    testset2 = trainingset2.copy(deep=True)
    testset2['Label1'].iloc[0] = 10


    print("before z-score")
    print('trainingset2')
    print(trainingset2)
    print('testset2')
    print(testset2)

    trainingset2,testset2 = standardizeData(trainingset2, testset2, 2)

    print("after z-score")
    print('trainingset2')
    print(trainingset2)
    print('testset2')
    print(testset2)
