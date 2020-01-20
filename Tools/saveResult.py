import sys, os
import numpy as np
import pandas as pd

def saveResult(confusion, scoreEval, experiment, SETTINGS):

    """ Save experiment result to csv files

    Parameters
    ----------
        confusion:              array
        scoreEval:              dataFrame
        experiment:             string
        SETTINGS:               class
        
    """

    SETTINGS.PATH_OUTPUT_RESULT = SETTINGS.PATH_OUTPUT + '/Experiments/' + experiment + '/'

    if not os.path.exists(SETTINGS.PATH_OUTPUT_RESULT):
        os.makedirs(SETTINGS.PATH_OUTPUT_RESULT)

    if SETTINGS.EVALUATION == 'pd':
        cmPath = SETTINGS.PATH_OUTPUT_RESULT + SETTINGS.EVALUATION + '_subject' + str(SETTINGS.SUBJECT)+ '_' + SETTINGS.DATASET + '_cm.csv'
        evalPath = SETTINGS.PATH_OUTPUT_RESULT + SETTINGS.EVALUATION + '_subject' + str(SETTINGS.SUBJECT)+ '_' + SETTINGS.DATASET + '_eval.csv'
    else:
        cmPath = SETTINGS.PATH_OUTPUT_RESULT + SETTINGS.EVALUATION + '_' + SETTINGS.DATASET + '_cm.csv'
        evalPath = SETTINGS.PATH_OUTPUT_RESULT + SETTINGS.EVALUATION + '_' + SETTINGS.DATASET + '_eval.csv'

    if SETTINGS.VERBOSE_LEVEL >= 2:
        print('Saving confusion matrix in ', cmPath)
        print('Saving scoreEval in ', evalPath)

    
    np.savetxt(cmPath, confusion, delimiter=",")
    scoreEval.to_csv( evalPath, index = None)
