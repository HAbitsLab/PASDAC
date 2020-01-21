import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from scipy import interp
import colorsys


def make_colors(num):
    for i in np.arange(0.0, 360.0, 360.0 / num):
        yield colorsys.hls_to_rgb(i / 360.0, 0.7, 0.9)


def plotROC(labels, probas, n_classes, savepath=None):
    """Plot all ROC curves

    Parameters
    ----------
    labels:             numpy array
    probas:             numpy array
    n_classes:          int

    Return
    ------

    """

    # labels starts from 0
    if n_classes == 2:
        label_col0 = (max(np.unique(labels)) - labels)[np.newaxis].T
        label_col1 = (labels - min(np.unique(labels)))[np.newaxis].T
        labels = np.hstack([label_col0, label_col1])
    else:
        labels = label_binarize(labels, classes=[i for i in range(n_classes)])

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(labels[:, i], probas[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot all ROC curves
    plt.figure()
    colors = list(make_colors(n_classes))
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")

    if savepath:
        plt.savefig(savepath, bbox_inches='tight')

    plt.show()


if __name__ == "__main__":

    from sklearn import svm, datasets
    from sklearn.model_selection import train_test_split
    from sklearn.multiclass import OneVsRestClassifier

    # Import some data to play with
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    print(y)

    # Binarize the output
    y = label_binarize(y, classes=[0, 1, 2])
    n_classes = y.shape[1]

    # shuffle and split training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5,
                                                        random_state=0)

    # Learn to predict each class against the other
    classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True))
    y_score = classifier.fit(X_train, y_train).decision_function(X_test)
    plotROC(y_test, y_score, n_classes)
