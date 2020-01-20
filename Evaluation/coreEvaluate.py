from __future__ import division
import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef, cohen_kappa_score


def coreEvaluate(labels, pred, classes):
    
    """ calculate metrics for class c and non-class c. 

    Parameters
    ----------
        labels:                 ndarray
        pred:                   ndarray
        c:                      int

    Return
    ------
        precisions:             ndarray
        recalls:                ndarray
        fallouts:               ndarray
        ccuracys:               ndarray

    """

    groundtruth = [a==classes for a in labels]
    detections = [a==classes for a in pred]
    
    precisions = np.zeros(2)
    recalls = np.zeros(2)
    fallouts = np.ones(2)
    specificities = np.zeros(2)
    NPVs = np.zeros(2)
    FDRs = np.zeros(2)
    FNRs = np.ones(2)
    accuracies = np.ones(2)
    f1_pos = np.zeros(1)
    MCC = 0
    CKappa = 0

    labelSteps = [1,0]

    for n in range(2):
        # detections is ndarray
        # detections = pred>=labelSteps[i]
        l = labelSteps[n]
        TP = 0
        for i in range(len(groundtruth)):
            if groundtruth[i] == l and detections[i] == l:
                TP = TP + 1
                
        FN = 0
        for i in range(len(groundtruth)):
            if groundtruth[i] == l and detections[i] != l:
                FN = FN + 1
                
        FP = 0
        for i in range(len(groundtruth)):
            if groundtruth[i] != l and detections[i] == l:
                FP = FP + 1
                
        TN = 0
        for i in range(len(groundtruth)):
            if groundtruth[i] != l and detections[i] != l:
                TN = TN + 1
        # TP = sum((a == l and b == l) for a, b in zip(groundtruth, detections))[0]  # groundtruth and detection      
        # print(TP)
        # FN = sum((a == l and b == l) for a, b in zip(groundtruth, [not e for e in detections]))[0] # groundtruth and no detection
        # print(FN)
        # FP = sum((a == l and b == l) for a, b in zip([not e for e in groundtruth], detections))[0] # groundtruth and no detection
        # print(FP)
        # TN = sum((a == l and b == l) for a, b in zip([not e for e in groundtruth], [not i for i in detections]))
        # print(TN)

        try:
            precisions[n] = (TP)/(TP+FP)     # or positive predictive value
        except ZeroDivisionError as err:
            precisions[n] = float('nan')

        try:
            recalls[n] = float(TP)/float(TP+FN)        # or true positive rate, hit rate, sensitivity
        except ZeroDivisionError as err:
            recalls[n] = float('nan')
        
        try:
            fallouts[n] = float(FP)/float(FP+TN)       # false positive rate
        except ZeroDivisionError as err:
            fallouts[n] = float('nan')

        try:
            specificities[n] = float(TN)/float(FP+TN)       # true negative rate
        except ZeroDivisionError as err:
            specificities[n] = float('nan')

        try:
            NPVs[n] = float(TN)/float(FN+TN)       # negative predictive value
        except ZeroDivisionError as err:
            NPVs[n] = float('nan')

        try:
            FDRs[n] = float(FP)/float(FP+TP)       # false discovery rate
        except ZeroDivisionError as err:
            FDRs[n] = float('nan')
        
        try:
            FNRs[n] = float(FN)/float(FN+TP)       # false negative rate
        except ZeroDivisionError as err:
            FNRs[n] = float('nan')

        accuracies[n] = float(TP+TN)/float(TP+TN+FP+FN)

        if n == 0:
            try:
                f1_pos = float(2*TP)/float(2*TP+FP+FN)       # f1-score, f measurement
            except ZeroDivisionError as err:
                f1_pos = float('nan')
            
            MCC = matthews_corrcoef(groundtruth, detections)

            CKappa = cohen_kappa_score(groundtruth, detections)

    return precisions, recalls, fallouts, specificities, NPVs, FDRs, FNRs, accuracies, f1_pos, MCC, CKappa

if __name__ == "__main__":
    
    import pandas as pd

    labels = pd.DataFrame(data = np.array([1, 1, 1, 2, 2, 2]))
    pred = pd.DataFrame(data = np.array([1, 1, 2, 2, 1, 1]))

    labelsArr = labels.as_matrix()
    predArr = pred.as_matrix()

    print(coreEvaluate(labelsArr, predArr, 1))
