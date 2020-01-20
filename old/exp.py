import os

try:
    from old.mysettings import *
except ImportError:
    print('local mysettings.py file not found')
    pass

from prepareFoldData import prepareFoldData
from old.runEvaluation import runEvaluation
from Tools.saveResult import saveResult



features, fType, fDescr, segments, segmentations, labelsSegmentations, setting1 = prepareFoldData(setting1)

confusion, scoreEval = runEvaluation(features, fType, fDescr, segments, segmentations, labelsSegmentations, setting1)

print(confusion)
print(scoreEval)

saveResult(confusion, scoreEval, os.path.basename(__file__)[:-3], setting1)

