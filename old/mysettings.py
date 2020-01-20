
import copy

from Tools.class_settings import SETTING

setting1 = SETTING('Data2R','Output','/feature')
setting1.set_SAMPLINGRATE(32)# sampling rate
setting1.set_SUBJECT(1)
setting1.set_SUBJECT_LIST([1,2])
setting1.set_SUBJECT_TOTAL(2)
setting1.set_DATASET('gesture')
setting1.set_SMOOTHING_TECHNIQUE(method='boxcar',winsize=30)# Method can be 'boxcar' or 'gaussian'. For example: (method='boxcar',winsize=30) or (method='gaussian',winsize=10,sigma=8)

# note: comment in 'set_SEGMENTATION_TECHNIQUE' to test default setting 'method='slidingWindow',winSizeSecond=1,stepSizeSecond=0.1'
# setting1.set_SEGMENTATION_TECHNIQUE(method='slidingWindow',winSizeSecond=1,stepSizeSecond=0.1)# options: slidingWindow, energyPeakBased, energyPeakCenteredWindow
setting1.set_FEATURE_TYPE('VerySimple') # type of features to calculate, possible values: Raw, VerySimple, Simple, FFT, All
setting1.set_SAVE(0) # (de)activate saving of variable outputs
setting1.set_PLOT(1) # (de)activate plotting
setting1.set_VERBOSE_LEVEL(0) # verbose level for debug messages, possible values: 0 (quiet), 1 (results), 2 (processing steps)
setting1.set_FEATURE_SELECTION('none') # feature selection method to use, possible values: none, mRMR, SFS, SBS
setting1.set_FEATURE_SELECTION_OPTIONS(10) # number of features to select
setting1.set_FUSION_TYPE('early') # 'early' (i.e. feature-level) or 'late' (i.e. classifier-level) data fusion
setting1.set_CLASSIFIER('knnVoting') # classifier to use, possible values: knnVoting, NaiveBayes, SVM, liblinear, SVMlight, DA, cHMM, jointboosting
setting1.set_CLASSIFIER_OPTIONS('knnVoting')
setting1.set_EVALUATION('pi') # type of evaluation, possible values: pd (person-dependent), pi (person-independent, leave-one-person-out), loio (leave-one-instance-out)




setting2 = copy.deepcopy(setting1)
setting2.set_SEGMENTATION_TECHNIQUE(method='energyPeakBased',energyMethod='FFTEnergyParseval',\
									FFTWinSizeSecond=2,signalColumns=[0,1,2],valley=0,\
									minPeakHeight=100000,minPeakDistance=100)# options: slidingWindow, energyPeakBased, energyPeakCenteredWindow


setting3 = copy.deepcopy(setting1)
setting3.set_SEGMENTATION_TECHNIQUE(method='energyPeakBased',energyMethod='sumSquareEnergy',\
									FFTWinSizeSecond=2,signalColumns=[0,1,2],valley=0,\
									minPeakHeight=100000,minPeakDistance=100)# options: slidingWindow, energyPeakBased, energyPeakCenteredWindow


setting4 = copy.deepcopy(setting1)
setting4.set_SEGMENTATION_TECHNIQUE(method='energyPeakBased',energyMethod='FFTDynamicEnergy',\
									FFTWinSizeSecond=2,signalColumns=[0,1,2],valley=0,\
									minPeakHeight=100000,minPeakDistance=100)# options: slidingWindow, energyPeakBased, energyPeakCenteredWindow


setting5 = copy.deepcopy(setting1)
setting5.set_SEGMENTATION_TECHNIQUE(method='energyPeakBased',energyMethod='FFTEnergy',\
									FFTWinSizeSecond=2,signalColumns=[0,1,2],valley=0,\
									minPeakHeight=100000,minPeakDistance=100)# options: slidingWindow, energyPeakBased, energyPeakCenteredWindow



setting6 = copy.deepcopy(setting1)
setting6.set_SEGMENTATION_TECHNIQUE(method='energyPeakBased',energyMethod='FFTEnergyParseval',\
									FFTWinSizeSecond=2,signalColumns=[0,1,2],valley=1,\
									minPeakHeight=10000000,minPeakDistance=100)# options: slidingWindow, energyPeakBased, energyPeakCenteredWindow



setting7 = copy.deepcopy(setting1)
setting7.set_SEGMENTATION_TECHNIQUE(method='energyPeakBased',energyMethod='sumSquareEnergy',\
									FFTWinSizeSecond=2,signalColumns=[0,1,2],valley=1,\
									minPeakHeight=10000000,minPeakDistance=100)# options: slidingWindow, energyPeakBased, energyPeakCenteredWindow


setting8 = copy.deepcopy(setting1)
setting8.set_SEGMENTATION_TECHNIQUE(method='energyPeakBased',energyMethod='FFTDynamicEnergy',\
									FFTWinSizeSecond=2,signalColumns=[0,1,2],valley=1,\
									minPeakHeight=10000000,minPeakDistance=100)# options: slidingWindow, energyPeakBased, energyPeakCenteredWindow



setting9 = copy.deepcopy(setting1)
setting9.set_SEGMENTATION_TECHNIQUE(method='energyPeakBased',energyMethod='FFTEnergy',\
									FFTWinSizeSecond=2,signalColumns=[0,1,2],valley=1,\
									minPeakHeight=10000000,minPeakDistance=100)# options: slidingWindow, energyPeakBased, energyPeakCenteredWindow


setting10 = copy.deepcopy(setting1)
setting10.set_SEGMENTATION_TECHNIQUE(method='energyPeakCenteredWindow',winSizeSecond=10,energyMethod='FFTDynamicEnergy',\
									FFTWinSizeSecond=2,signalColumns=[0,1,2],valley=1,\
									minPeakHeight=10000000,minPeakDistance=100)# options: slidingWindow, energyPeakBased, energyPeakCenteredWindow




if __name__ == "__main__":
	print(setting1.SAMPLINGRATE)
	print(setting1.CLASSIFIER)
	print(setting1.CLASSLABELS)
	print(setting1.CLASSIFIER_OPTIONS_TRAINING)
	print(setting1.CLASSIFIER_OPTIONS_TESTING)
	print(setting1.EVALUATION)
	print(setting1.SAVE)
