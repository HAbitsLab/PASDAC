import numpy as np
import pandas as pd
from scipy import stats


def calculateFeaturesStats(data = np.array([]),col_name=""):
    
    """Calculates features Mean, Variance, ZCR, and MCR

    Parameters
    ----------
        data:                   numpy array

    Return
    ------
        dict:
        - dict.mean:            double
        - dict.variance:        double
        - dict.zcr:             int
        - dict.mcr:             int

    """
    if data.size == 0:
        return 
    
    f = []
    rv = {}
    rv[col_name+'_min'] = np.amin(data)
    rv[col_name+'_max'] = np.amax(data)
    rv[col_name+'_mean'] = np.mean(data)
    rv[col_name+'_median'] = np.median(data)
    rv[col_name+'_mode'] = stats.mode(data)[0][0]
    rv[col_name+'_std'] = np.std(data)
    rv[col_name+'_variance'] = np.var(data)
    rv[col_name+'_skew'] = stats.skew(data,axis = 0)
    rv[col_name+'_kur'] = stats.kurtosis(data,axis = 0)
    rv[col_name+'_eightperc'] = np.percentile(data, 80,axis = 0)
    rv[col_name+'_sixperc'] = np.percentile(data, 60,axis = 0)
    rv[col_name+'_fourperc'] = np.percentile(data, 40,axis = 0)
    rv[col_name+'_twoperc'] = np.percentile(data, 20,axis = 0)
    rv[col_name+'_rms'] = np.sqrt(np.mean(data**2))
    rv[col_name+'_iqr'] = stats.iqr(data,axis = 0)
    rv[col_name+'_countgeq'] = len(np.where( data > rv[col_name+'_mean'])[0])/float(len(data))
    rv[col_name+'_countleq'] = len(np.where( data < rv[col_name+'_mean'])[0])/float(len(data))
    rv[col_name+'_range'] = rv[col_name+'_max'] - rv[col_name+'_min']
    rv[col_name+'_zcr'] = (np.diff(np.sign(data)) != 0).sum()
    
    normalized = data - rv[col_name+'_mean']
    rv[col_name+'_mcr'] = (np.diff(np.sign(normalized)) != 0).sum()

    return rv
    
if __name__ == "__main__":

    mean = 0
    std = 1 
    num_samples = 500
    samples = np.random.normal(mean, std, size=num_samples)
    data = np.array(samples)
    df = pd.DataFrame(data=np.transpose(np.array([data,data])))
    features_df = pd.DataFrame()
    features ={}

    for c in df.columns:
        features.update(calculateFeaturesStats(df[c],str(c)))
        # features = calculateFeaturesStats(df[c],str(c))

    features_df = features_df.append(features,ignore_index=True)

    # features = calculateFeaturesStats(data)
    # print(features.keys())

    #do this for each sensor and then append it