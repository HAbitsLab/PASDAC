from Evaluation.coreEvaluate import coreEvaluate
from sklearn.metrics import confusion_matrix
import pandas as pd


def evaluation(predTimeseries, labelsTimeseries, scoresTimeseries, SETTINGS):
    """Evaluation:

    Parameters
    ----------

        finalPredTimeseries:                list of dataFrame
        labelsTimeseries:                   list of dataFrame
        scoresTimeseries:                   list of dataFrame

    Return
    ------
        confusion:              array
        scoreEval:              dataFrame

    """

    if SETTINGS.VERBOSE_LEVEL >= 2:
        print('Evaluation')

    # (5.1) Evaluating predTimeseries  confusion matrix
    labelsArr = labelsTimeseries.values
    predArr = predTimeseries.values
    confusion = confusion_matrix(labelsArr.astype(int), predArr.astype(int))

    # (5.2) Score-based performance evaluation
    columns = ['precisions', 'recalls', 'fallouts', 'specificities', 'NPVs', 'FDRs', 'FNRs', 'accuracies', 'f1_pos', 'MCC', 'CKappa']
    scoreEval = pd.DataFrame(columns = columns, index = range(SETTINGS.CLASSES))

    for c in range(SETTINGS.CLASSES):
        # Evaluation on Timeseries
        scoreEval['precisions'][c], scoreEval['recalls'][c], scoreEval['fallouts'][c], scoreEval['specificities'][c],\
        scoreEval['NPVs'][c],       scoreEval['FDRs'][c],    scoreEval['FNRs'][c],     scoreEval['accuracies'][c], \
        scoreEval['f1_pos'][c],     scoreEval['MCC'][c],     scoreEval['CKappa'][c] = coreEvaluate(labelsArr, predArr, c+1)

    return confusion, scoreEval