import pandas as pd
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt



def smoothing(dataDf, kwargs):

    """smoothing function including sliding smoothing(box smoothing) and gaussian filter

    Parameters
    ----------
        dataDf:                 dataFrame
        kwargs:                 Dict


    Return
    ------
        dataDf:                 dataFrame

    """


    allowedScipyMethods = {"gaussian":{"winsize":10,"sigma":8},"boxcar":{"winsize":51}}

    userSetArgs = kwargs


    #Check if there is the user set a method for smoothing. If no method is provided then ust return the DataFrame
    if 'method' in userSetArgs:
        method = userSetArgs["method"]
        print('Smoothing method: ',  method)
    else:
        print("No smoothing method applied")
        return dataDf


    if method in allowedScipyMethods.keys() :
        #getting the list of default arguments for the provided smoothign method
        defaultArgs = allowedScipyMethods[method]


        #Checking if all arguments are provided by the user.If an argument is missing the defaul argument will be used
        for arg in defaultArgs.keys():
            if not(arg in userSetArgs.keys()):
               userSetArgs[arg]= defaultArgs[arg]
               print("Arg '"+arg+"' is set as default ", userSetArgs[arg])


        names = list(dataDf.columns.values)
        arr = dataDf.values

        #Looping through columns to apply the snoothing method for each column
        for c in range(arr.shape[1]):
            col = arr[:,c]
            #TODO: add padding

            # when winsize is even, int(winsize/2) is bigger than int((winsize-1)/2) by 1
            # when winsize is odd, int(winsize/2) is the same as int((winsize-1)/2)
            pad_head = [col[0]] * int((userSetArgs['winsize']-1)/2)
            # print(pad_head)
            pad_tail = [col[-1]]* int(userSetArgs['winsize']/2)
            # print(pad_tail)

            s=np.r_[pad_head,col,pad_tail]

            w = eval('signal.'+method+'('+','.join(str(userSetArgs[arg]) for arg in defaultArgs.keys())+ ')')
            smoothed=np.convolve(w/w.sum(),s,mode='valid')
            arr[:,c] = smoothed

        dataDf = pd.DataFrame(data = arr, columns = names)

        #TODO: plot part of the signal after and before



    return dataDf




if __name__ == "__main__":

    trainingset = pd.DataFrame(data = np.zeros(20)+1, columns = ["Label"], dtype = int)

    # trainingset['Label1']= trainingset['Label']
    trainingset['Label'].iloc[0] = 10

    print("Orginal data")
    print(trainingset)
    plt.title('Before smoothing')
    plt.plot(trainingset)
    plt.show()




    trainingset = smoothing(trainingset, { "method": "gaussian","winsize":10})

    print("Smoothed data")
    print(trainingset)
    plt.title('After smoothing')
    plt.plot(trainingset)
    plt.show()
