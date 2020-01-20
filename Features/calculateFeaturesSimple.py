import numpy as np

def calculateFeaturesSimple(data = np.array([])):
    
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
    rv['mean'] = np.mean(data)
    rv['variance'] = np.var(data)
    rv['zcr'] = (np.diff(np.sign(data)) != 0).sum()
    
    normalized = data - rv['mean']
    rv['mcr'] = (np.diff(np.sign(normalized)) != 0).sum()

    f.append(rv['mean'])
    f.append(rv['variance'])
    f.append(rv['zcr'])
    f.append(rv['mcr'])

    return f
    
if __name__ == "__main__":

    mean = 0
    std = 1 
    num_samples = 500
    samples = np.random.normal(mean, std, size=num_samples)
    data = np.array(samples)
    features = calculateFeaturesSimple(data)
    print (features)