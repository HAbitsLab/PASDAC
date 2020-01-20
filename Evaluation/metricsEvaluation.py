from __future__ import division
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef, cohen_kappa_score
from utils import *


def evaluateBiclass(groundtruth, detection):
    '''
    Note: inner function, called by core_evaluate, should not be called by user
    '''
    TP = 0
    FN = 0
    FP = 0
    TN = 0
    cm = np.zeros((2,2))

    for i in range(len(groundtruth)):
        if groundtruth[i] == 1 and detection[i] == 1:
            TP = TP + 1                
        elif groundtruth[i] == 1 and detection[i] != 1:
            FN = FN + 1
        elif groundtruth[i] != 1 and detection[i] == 1:
            FP = FP + 1
        else:
            TN = TN + 1
    
    cm[0,0] = TP
    cm[0,1] = FN
    cm[1,0] = FP
    cm[1,1] = TN

    try:
        precision = float(TP)/(TP+FP)     # or positive predictive value
    except ZeroDivisionError as err:
        precision = float('nan')

    try:
        recall = float(TP)/float(TP+FN)        # or true positive rate, hit rate, sensitivity
    except ZeroDivisionError as err:
        recall = float('nan')
        
    try:
        fallout = float(FP)/float(FP+TN)       # false positive rate
    except ZeroDivisionError as err:
        fallout = float('nan')

    try:
        specificity = float(TN)/float(FP+TN)       # true negative rate
    except ZeroDivisionError as err:
        specificity = float('nan')

    try:
        NPV = float(TN)/float(FN+TN)       # negative predictive value
    except ZeroDivisionError as err:
        NPV = float('nan')

    try:
        FDR = float(FP)/float(FP+TP)       # false discovery rate
    except ZeroDivisionError as err:
        FDR = float('nan')
        
    try:
        FNR = float(FN)/float(FN+TP)       # false negative rate
    except ZeroDivisionError as err:
        FNR = float('nan')

    accuracy = float(TP+TN)/float(TP+TN+FP+FN)

    try:
        f1_pos = float(2*TP)/float(2*TP+FP+FN)       # f1-score, f measurement
    except ZeroDivisionError as err:
        f1_pos = float('nan')
            
    MCC = matthews_corrcoef(groundtruth, detection)

    CKappa = cohen_kappa_score(groundtruth, detection)

    return cm, precision, recall, fallout, specificity, NPV, FDR, FNR, accuracy, f1_pos, MCC, CKappa


def metricsEvaluation(labels, pred, target_class=None,logfile=None):
    
    """ 
    Function description: 
    ----------
        Calculate metrics for target_class. 
        Output 'precision, recall, fallout, specificity, NPV, FDR, FNR, accuracy, f1_pos, MCC, CKappa' for target class;
        if target class is not passed, print the metrics for each class and output the metrics for the last class.

    Parameters
    ----------
        labels:                 list or tuple or ndarray or dataframe, the labels for instances.
        pred:                   list or tuple or ndarray or dataframe, the predictions for instances.
        target_class:           int, the target class that the returned metrics is based on. By default is None

    Return
    ------
        cm:                    confusion matrix, 2*2 ndarray
        precision:             also named positive predictive value, TP/(TP+FP)
        recall:                also named true positive rate or hit rate or sensitivity, TP/P
        fallout:               also named false positive rate (FPR), FP/(TN+FP)
        accuracy:              TP+TN/all
        specificity:           true negative rate, TN/(FP+TN) 
        NPV:                   negative predictive value, TN/(FN+TN)
        FDR:                   false discovery rate, FP/(FP+TP)
        FNR:                   false negative rate, FN/(FN+TP)
        f1_pos:                f1 score for positive class
        MCC:                   Matthews correlation coefficient
        CKappa:                a measure of how well the classifier performed as compared to how well it would have performed simply by chance,
                            in other words, a model will have a high Kappa score if there is a big difference between the accuracy and the null error rate.

    Author
    ------
    Shibo(shibozhang2015@u.northwestern.edu)
    """

    # input data type check
    assert isinstance(labels, (list, tuple, np.ndarray, pd.DataFrame))
    assert isinstance(pred, (list, tuple, np.ndarray, pd.DataFrame))

    if isinstance(labels, pd.DataFrame):
        labels = labels.values

    if isinstance(pred, pd.DataFrame):
        pred = pred.values


    # if argument target_class is not assigned, print all the metrics for all classes.
    if target_class == None:
        lprint(logfile, 'No target class is assigned.\n')        
        label_uniq = np.unique(labels)

        for tar_class in label_uniq:
            lprint(logfile, 'For class',tar_class)

            groundtruth = [a == tar_class for a in labels]
            detection = [a == tar_class for a in pred]

            cm, precision, recall, fallout, specificity, NPV, FDR, FNR, accuracy, f1_pos, MCC, CKappa = evaluateBiclass(groundtruth, detection)
            lprint(logfile, '  confusion matrix is ', cm)
            lprint(logfile, '  precision is ', precision)
            lprint(logfile, '  recall is ', recall)
            lprint(logfile, '  fallout is ', fallout)
            lprint(logfile, '  specificity is ', specificity)
            lprint(logfile, '  NPV is ', NPV)
            lprint(logfile, '  FDR is ', FDR)
            lprint(logfile, '  FNR is ', FNR)
            lprint(logfile, '  accuracy is ', accuracy)
            lprint(logfile, '  f1_pos is ', f1_pos)
            lprint(logfile, '  MCC is ', MCC)
            lprint(logfile, '  CKappa is ', CKappa, '\n')
    else:
        lprint(logfile, 'Target class is Class ',target_class)

        groundtruth = [a == target_class for a in labels]
        detection = [a == target_class for a in pred]

        cm, precision, recall, fallout, specificity, NPV, FDR, FNR, accuracy, f1_pos, MCC, CKappa = evaluateBiclass(groundtruth, detection)
        lprint(logfile, '  confusion matrix is ', cm)
        lprint(logfile, '  precision is ', precision)
        lprint(logfile, '  recall is ', recall)
        lprint(logfile, '  fallout is ', fallout)
        lprint(logfile, '  specificity is ', specificity)
        lprint(logfile, '  NPV is ', NPV)
        lprint(logfile, '  FDR is ', FDR)
        lprint(logfile, '  FNR is ', FNR)
        lprint(logfile, '  accuracy is ', accuracy)
        lprint(logfile, '  f1_pos is ', f1_pos)
        lprint(logfile, '  MCC is ', MCC)
        lprint(logfile, '  CKappa is ', CKappa, '\n')
        
    return cm, precision, recall, fallout, specificity, NPV, FDR, FNR, accuracy, f1_pos, MCC, CKappa




if __name__ == "__main__":
    
    print('test case 1: ')
    labels = pd.DataFrame(data = np.array([1, 1, 1, 2, 2, 2]))
    pred = pd.DataFrame(data = np.array([1, 1, 2, 2, 1, 1]))
    cm, precision, recall, fallout, specificity, NPV, FDR, FNR, accuracy, f1_pos, MCC, CKappa = metricsEvaluation(labels, pred, 1, logfile = 'logEvalTestCase1.txt')


    print('test case 2: ')
    labels = np.array([0, 0, 1, 1, 1, 2, 2, 2])
    pred = np.array([0, 1, 1, 1, 2, 2, 1, 1])
    print(metricsEvaluation(labels, pred)) # this test case also test when there is no argument logfile.


    print('test case 3: ')
    labels = [1, 1, 1, 2, 2, 2, 3, 3]
    pred = [1, 1, 2, 2, 1, 1, 3, 3]
    print(metricsEvaluation(labels, pred, logfile = 'logEvalTestCase2.txt'))
