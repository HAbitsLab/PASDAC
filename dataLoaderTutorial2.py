import sys, os
import numpy as np
import pandas as pd

from Tools.class_settings import SETTING

sys.path.append(os.path.join(os.path.dirname(__file__), 'Tools/labels'))
from Preprocessing.splitIntoRepetitions import splitIntoRepetitions
from Dataset.Dataset import Dataset

def Tutorial2_dataloader(SETTINGS):

    ClassActivity = Dataset()
    activityLabelList = {}



    for rep in (SETTINGS.SUBJECT_LIST):
        dataAll = []
        labelsAll = []
        dataFileName = SETTINGS.PATH_DATA + '/Repetition' + str(rep) + '_' + SETTINGS.DATASET + '_data.csv'
        labelsFileName = SETTINGS.PATH_DATA + '/Repetition' + str(rep) + '_' + SETTINGS.DATASET + '_labels.csv'

        print(dataFileName)
        print(labelsFileName)

        if os.path.isfile(dataFileName):
            data = pd.read_csv(dataFileName)#, names=SETTINGS.SENSORS_AVAILABLE)

        else:
            print('prepareFoldData:fileDoesNotExist, ' + dataFileName + ' does not exist in the file system.')

        # import label file
        if os.path.isfile(labelsFileName):
            labels = pd.read_csv(labelsFileName, names=[ 'round','activity','start','end','label'])
            labelsSegmented,activityLabelList = segment(pd.read_csv(dataFileName),labels,activityLabelList)
            dataAll.append(data)
            labelsAll.append(labelsSegmented)


        else:
            print('prepareFoldData:fileDoesNotExist, ' + labelsFileName + ' does not exist in the file system.')

        ClassActivity.set_participant_raw_data(str(rep), dataAll)
        ClassActivity.set_participant_raw_labels(str(rep), labelsAll)
        #print(labelsAll)


    print(activityLabelList)
    return ClassActivity


def segment(data,labels,allClasses):
    activityTime = data['Date']
    startOfExp = activityTime.iloc[0] - labels.loc[labels['label'] == 'starExp']['start'][0]
    allUnique = labels.label.unique()

    for classes in allUnique:
        if classes not in allClasses:
            allClasses[classes]=len(allClasses)

    labelDF = pd.DataFrame(columns = ['Start','End','Label'])

    timeLabelDF = pd.DataFrame({'Date': activityTime, 'label': pd.Series([-1 for ii in range(0, len(activityTime))])})
    tempDF = timeLabelDF.copy()
    for index, row in timeLabelDF.iterrows():
        for index2, row2 in labels.iterrows():
            if row2['start'] < row['Date'] - startOfExp and row2['end'] > row['Date'] - startOfExp:
                new_df = pd.DataFrame({'label': allClasses[row2['label']]}, index=[index])
                tempDF.update(new_df)

                break
    timeLabelDF.update(tempDF)

    lastLabel = 0
    startOfLabel = 1
    for index, row in timeLabelDF.iterrows():
        if index==0:
            lastLabel=row['label']
            continue
        if row['label']!=lastLabel:
            newDF = pd.DataFrame([[startOfLabel,index-1,lastLabel]],columns = ['Start','End','Label'])
            startOfLabel=index
            lastLabel=row['label']
            labelDF = pd.concat([labelDF,newDF],ignore_index=True)
    newDF = pd.DataFrame([[startOfLabel, index - 1, lastLabel]], columns=['Start', 'End', 'Label'])
    labelDF = pd.concat([labelDF, newDF], ignore_index=True)

    #print(newDF,labelDF)

    #
    #
    # for index, row in labels.iterrows():
    #     newDF = pd.DataFrame([[startOfExp+row['start'],startOfExp+row['end'],np.where(allUnique==row['label'])[0][0]]],columns = ['Start','End','Label'])
    #     labelDF=pd.concat([labelDF,newDF],ignore_index=True)


    #print(labelDF)
    return labelDF,allClasses

if __name__=='__main__':
    setting = SETTING('Data2R', 'Output', '/feature')
    setting.set_SAMPLINGRATE(20)  # sampling rate
    setting.set_DATASET('classActivity')
    setting.CLASSLABELS = ['NULL', 'stand', 'walk']
    setting.SENSOR_PLACEMENT = ['Pant pocket Right']
    setting.CLASSES = len(setting.CLASSLABELS)
    setting.FOLDS = 2
    setting.set_SUBJECT_LIST([1, 2])
    setting.SENSORS_AVAILABLE = ['date', 'X (mg)', 'Y (mg)', 'Z (mg)', 'X (dps)', 'Y (dps)', 'Z (dps)', 'X (mGa)',
                                 'Z (mGa)', 'Z (mGa)']
    dataset = Tutorial2_dataloader(setting)
    raw_data = len(dataset.raw['1']['labels'][0])
    #print(raw_data)


