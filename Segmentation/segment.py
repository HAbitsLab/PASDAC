# todo: implement majority with overlap threshold

import sys, os
import pandas as pd
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Tools'))
from secondsToSamples import secondsToSamples

sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
from detectPeaks import detectPeaks
import genEnergy

def slidingWindow(data, SETTINGS):
    """Sliding window algorithm realization Output 'segments'
    contains start and end indexes for each step

    Parameters
    ----------
    dataArr:        numpy array
    SETTINGS:       object

    Return
    ------
    segmentDf:       dataFrame

    """

    userSetArgs = SETTINGS.SEGMENTATION_TECHNIQUE
    winSizeSecond = userSetArgs["winSizeSecond"]
    stepSizeSecond = userSetArgs["stepSizeSecond"]

    # logger.info("Sliding window with win size %.2f second and step size %.2f second",
    #             winSizeSecond, stepSizeSecond)

    if stepSizeSecond > winSizeSecond:
        raise ValueError("Step size %.2f must not be larger than window size %.2f",
                         stepSizeSecond, winSizeSecond)

    start_time = 0
    end_time = data.shape[0]

    samplingRate = SETTINGS['SAMPLINGRATE']
    winSize = int(round(userSetArgs['winSizeSecond'] * samplingRate))
    stepSize = int(round(userSetArgs['stepSizeSecond'] * samplingRate))

    segments_start = np.arange(start_time, end_time - winSize, stepSize)
    segments_end = segments_start + winSize

    segment = pd.DataFrame({'Start': segments_start,
                            'End': segments_end},
                           columns=['Start', 'End'])

    return segment




def getEnergyPeak(dataArr, userSetArgs, SETTINGS):

    # generate the energy signal
    samplingRate = SETTINGS.SAMPLINGRATE
    FFTWinSize = int(secondsToSamples(userSetArgs['FFTWinSizeSecond'], samplingRate))

    if FFTWinSize > len(dataArr):
        raise Exception('**ERROR** FFT window size must not be larger than data sequence length.')

    energyMethod = userSetArgs['energyMethod']
    mpd = userSetArgs['minPeakDistance']

    # if userSetArgs['valley'] is 1, then set the peak height threshold as
    if userSetArgs['valley']:
        mph = -userSetArgs['minPeakHeight']
    else:
        mph = userSetArgs['minPeakHeight']

    threshold = userSetArgs['immediateNeighborThreshold']
    edge = userSetArgs['edge']
    kpsh = userSetArgs['keepPeaksSameHeight']
    valley = userSetArgs['valley']

    dataArr = dataArr[:, userSetArgs['signalColumns']]
    if energyMethod == 'FFTEnergy':
        energyArr = genEnergy.FFTEnergy(dataArr, FFTWinSize)
    elif energyMethod == 'FFTEnergyParseval':
        energyArr = genEnergy.ParsevalEnergy(dataArr, FFTWinSize)
    elif energyMethod == 'sumSquareEnergy':
        energyArr = genEnergy.sumSquareEnergy(dataArr)
    elif energyMethod == 'FFTDynamicEnergy':
        energyArr = genEnergy.FFTDynamicEnergy(dataArr, FFTWinSize)

    # peak finding
    # a comparison of python peak finding library: https://blog.ytotech.com/2015/11/01/findpeaks-in-python/
    if SETTINGS.VERBOSE_LEVEL:
        if userSetArgs['valley']:
            print('...valley finding with method:', energyMethod)
        else:
            print('...peak finding with method:', energyMethod)

    peakIndArr = detectPeaks(energyArr, mph=mph, mpd=mpd, threshold=threshold, edge=edge,
                             kpsh=kpsh, valley=valley, show=SETTINGS.VERBOSE_LEVEL)

    return peakIndArr

def energyPeakBased(dataArr, SETTINGS):

    """Sliding window algorithm realization Output 'segments' contains start and end indexes for each step

    Parameters
    ----------
    dataArr:        numpy array
    SETTINGS:       object

    Return
    ------
    segmentDf:       dataFrame

    """

    # import settings
    userSetArgs = SETTINGS.SEGMENTATION_TECHNIQUE

    defaultArgs = {'energyMethod': 'FFTEnergyParseval', 'FFTWinSizeSecond': 2, 'signalColumns': [0, 1, 2], \
                   'valley': 0, 'minPeakHeight': 100000, 'minPeakDistance': 100, \
                   'immediateNeighborThreshold': 0, 'edge': 'rising', 'keepPeaksSameHeight': False}

    for arg in defaultArgs.keys():
        if not (arg in userSetArgs.keys()):
            userSetArgs[arg] = defaultArgs[arg]
            print('Arg {} is set as default {}'.format(arg, userSetArgs[arg]))

    # generate energy from dataArr specified columns and find the peak of the energy signal
    peakIndArr = self.getEnergyPeak(dataArr, userSetArgs, SETTINGS)

    length = dataArr.shape[0]

    if peakIndArr.size:
        # eg:   len(energy)=200, peakind = [3, 10, 43, 89, 110],
        #       then after 'peakind = np.c_[peakindC1, peakindC2]', peakind = [[0, 3], [3, 10], [10, 43], [43, 89], [89, 110], [110, 200]]
        if peakIndArr[0]:
            peakIndArr = np.r_[np.array([0]), peakIndArr]
        if peakIndArr[-1] != length:
            peakIndArr = np.r_[peakIndArr, np.array([length])]

        peakindC1Arr = np.reshape(peakIndArr[:-1], (-1, 1))
        peakindC2Arr = np.reshape(peakIndArr[1:], (-1, 1))
        segmentArr = np.c_[peakindC1Arr, peakindC2Arr]

        segmentDf = pd.DataFrame(data=segmentArr, columns=['Start', 'End'])

    else:
        if userSetArgs['valley']:
            print('No valley found. PEAK_HEIGHT_THRESHOLD to be tuned.')
        else:
            print('No peak found. PEAK_HEIGHT_THRESHOLD to be tuned.')

        segmentArr = np.c_[np.array([0]), np.array([length])]
        segmentDf = pd.DataFrame(data=segmentArr, columns=['Start', 'End'])

    if SETTINGS.VERBOSE_LEVEL >= 2:
        print('segment dataframe:')
        print(segmentDf)
        print('\n')

    return segmentDf

def energyPeakCenteredWindow(dataArr, SETTINGS):

    """Sliding window algorithm realization Output 'segments' contains start and end indexes for each step

    Parameters
    ----------
    dataArr:        numpy array
    SETTINGS:       object

    Return
    ------
    segmentDf:       dataFrame

    """

    # import settings
    userSetArgs = SETTINGS.SEGMENTATION_TECHNIQUE

    defaultArgs = {'winSizeSecond': 10, 'energyMethod': 'FFTEnergyParseval', 'FFTWinSizeSecond': 2, \
                   'signalColumns': [0, 1, 2], 'valley': 0, 'minPeakHeight': 100000, \
                   'minPeakDistance': 100, 'immediateNeighborThreshold': 0, 'edge': 'rising', \
                   'keepPeaksSameHeight': False}

    for arg in defaultArgs.keys():
        if not (arg in userSetArgs.keys()):
            userSetArgs[arg] = defaultArgs[arg]
            print('Arg {} is set as default {}'.format(arg, userSetArgs[arg]))

    # generate energy from dataArr specified columns and find the peak of the energy signal
    peakIndArr = self.getEnergyPeak(dataArr, userSetArgs, SETTINGS)

    length = dataArr.shape[0]

    # generate the energy signal
    samplingRate = SETTINGS.SAMPLINGRATE
    winSize = int(secondsToSamples(userSetArgs['winSizeSecond'], samplingRate))

    if winSize > len(dataArr):
        raise Exception('**ERROR** window size must not be larger than data sequence length.')

    if peakIndArr.size:
        # eg:   peakind = [3, 10, 43], winSize = 10
        #       then after 'peakind = np.c_[peakindC1, peakindC2]', peakind = [[0, 9], [6, 16], [39, 49]]
        peakIndStartArr = peakIndArr - int((winSize - 1) / 2)
        peakIndStartArr[peakIndStartArr < 0] = 0
        peakIndEndArr = peakIndArr + int(winSize / 2)
        peakIndEndArr[peakIndEndArr > length] = length
        segmentArr = np.c_[peakIndStartArr, peakIndEndArr]

        segmentDf = pd.DataFrame(data=segmentArr, columns=['Start', 'End'])

    else:
        if userSetArgs['valley']:
            print('No valley found. PEAK_HEIGHT_THRESHOLD to be tuned.')
        else:
            print('No peak found. PEAK_HEIGHT_THRESHOLD to be tuned.')
        segmentDf = None

    if SETTINGS.VERBOSE_LEVEL >= 2:
        print('segment dataframe:')
        print(segmentDf)
        print('\n')

    return segmentDf


