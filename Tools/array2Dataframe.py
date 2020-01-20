import pandas as pd
import numpy as np

def array2Dataframe(arrayOfDataframe):

    return pd.concat(arrayOfDataframe).reset_index(drop = True)


if __name__ == "__main__":

    labeling = pd.DataFrame(data = np.zeros(100), columns = ["Label"] ,dtype = int)
    arrdf = []
    arrdf.append(labeling)
    arrdf.append(labeling)
    print(array2Dataframe(arrdf))

