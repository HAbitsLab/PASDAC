import matplotlib.pyplot as plt

def plotConfusionMatrix(cm):

    """Plot confusion matrix

    Parameters
    ----------
    cm:             numpy array
    
    """

    plt.matshow(cm)
    plt.title('Confusion matrix')
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    return
