import numpy as np
from scipy.fftpack import fft

# background knowledge
# about acceleration energy: https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/19970034695.pdf
# about power spectral density: http://www.vibrationresearch.com/university/lesson/what-is-the-psd/


def oneDimFFTSquare(yArr):
    '''
    return sum of squares of all the fft components
    :param yArr: 1d numpy array
    :return numpy.float64
    '''
    N = yArr.shape[0]  # Number of samplepoints
    yfArr = fft(yArr)
    ampArr = 2.0 / N * np.abs(yfArr[:int(N / 2)])

    return sum(i * i for i in ampArr)


def oneDimFFTSquareWoFundamental(yArr):
    '''
    return sum of squares of fft components except the fundamental frequency component
    :param yArr: 1d numpy array
    :return numpy.float64

    ref: https://en.wikipedia.org/wiki/Parseval%27s_theorem
    '''

    N = yArr.shape[0] # Number of samplepoints
    yfArr = fft(yArr)
    ampArr = 2.0/N * np.abs(yfArr[:int(N/2)])

    return sum(i*i for i in ampArr[1:])


def oneDimAveParsevalEnergy(yArr):
    '''
    return the average Parseval energy for 1-d signal
    :param yArr: 1d numpy array
    :return numpy.float64
    '''

    return np.sum(yArr ** 2) * 2 / yArr.size


def ParsevalEnergy(signalArr, winSize):
    '''
    calculate energy of 1d sensor signal or multi-axis sensor signals based on Parseval Theorem

    :param signalArr: 1d numpy array or 2d numpy array with shape N*M (M as number of signal's axes, N as sampling points)
    :param winSize: int, the size of Parseval energy window size
    :return:energyArr: 1d numpy array

    ref: https://en.wikipedia.org/wiki/Parseval%27s_theorem
    '''
    energyList = []

    if signalArr.ndim == 1:
        signalArr = signalArr.reshape([-1, 1])

    for i in range(0, signalArr.shape[0] - winSize):
        signalSliceArr = signalArr[i:i + winSize,:]
        energyNew = 0
        for j in range(signalSliceArr.shape[1]):
            energyNew += oneDimAveParsevalEnergy(signalSliceArr[:,j])
        energyList.append(energyNew)

    energyArr = np.array(energyList)
    padHeadArr = np.full(int(winSize/2), energyArr[0])
    padTailArr = np.full(winSize - int(winSize/2), energyArr[-1])
    energyArr = np.r_[padHeadArr, energyArr, padTailArr]

    return energyArr


def FFTEnergy(signalArr, winSize):
    '''
    calculate FFT energy of 1d sensor signal or multi-axis sensor signals

    :param signalArr: 1d numpy array or 2d numpy array with shape N*M (M as number of signal's axes, N as sampling points)
    :param winSize: int, the size of FFT window size
    :return:energyArr: 1d numpy array
    '''
    energyList = []

    if signalArr.ndim == 1:
        signalArr = signalArr.reshape([-1, 1])

    for i in range(0, signalArr.shape[0] - winSize):
        signalSliceArr = signalArr[i:i + winSize, :]
        energyNew = 0
        for j in range(signalSliceArr.shape[1]):
            energyNew += oneDimFFTSquare(signalSliceArr[:, j])
        energyList.append(energyNew)

    energyArr = np.array(energyList)
    padHeadArr = np.full(int(winSize/2), energyArr[0])
    padTailArr = np.full(winSize - int(winSize/2), energyArr[-1])
    energyArr = np.r_[padHeadArr, energyArr, padTailArr]

    return energyArr


def FFTDynamicEnergy(signalArr, winSize):
    '''
    calculate dynamic FFT energy of 1d sensor signal or multi-axis sensor signals without fundamental frequency component

    :param signalArr: 1d numpy array or 2d numpy array with shape N*M (M as number of signal's axes, N as sampling points)
    :param winSize: int, the size of FFT window size
    :return:energyArr: 1d numpy array
    '''
    energyList = []

    if signalArr.ndim == 1:
        signalArr = signalArr.reshape([-1, 1])

    for i in range(0, signalArr.shape[0] - winSize):
        signalSliceArr = signalArr[i:i + winSize, :]
        energyNew = 0
        for j in range(signalSliceArr.shape[1]):
            energyNew += oneDimFFTSquareWoFundamental(signalSliceArr[:, j])
        energyList.append(energyNew)

    energyArr = np.array(energyList)
    padHeadArr = np.full(int(winSize/2), energyArr[0])
    padTailArr = np.full(winSize - int(winSize/2), energyArr[-1])
    energyArr = np.r_[padHeadArr, energyArr, padTailArr]

    return energyArr


def sumSquareEnergy(signalArr):
    '''
    calculate sum of square energy of 1d sensor signal or multi-axis sensor signals

    with equation:
            energy[i] = signal[i]^2     or
            energy[i] = accX[i]^2 + accY[i]^2 + accZ[i]^2

    :param signalArr: 1d numpy array or 2d numpy array with shape N*M (M as number of signal's axes, N as sampling points)
    :return:energyArr: 1d numpy array

    '''
    energyList = []
    for i in range(0, signalArr.shape[0]):
        energyList.append(np.sum(signalArr[i] ** 2))
    energyArr = np.array(energyList)

    return energyArr


if __name__ == '__main__':
    try:
        import matplotlib.pyplot as plt
    except:
        print('No library matplotlib installed.')

    x = np.linspace(0.0, 50.0, 1000)
    y = np.cos(2 * np.pi * x)

    y1 = FFTEnergy(y, 50)
    y2 = FFTDynamicEnergy(y, 50)
    y3 = ParsevalEnergy(y, 50)
    y4 = sumSquareEnergy(y)

    plt.subplot(5, 1, 1)
    plt.plot(y)
    plt.ylabel('Raw')

    plt.subplot(5, 1, 2)
    plt.plot(y1)
    plt.ylabel('FFTEnergy')

    plt.subplot(5, 1, 3)
    plt.plot(y2)
    plt.ylabel('FFTDynamicEnergy')

    plt.subplot(5, 1, 4)
    plt.plot(y3)
    plt.ylabel('ParsevalEnergy')

    plt.subplot(5, 1, 5)
    plt.plot(y4)
    plt.ylabel('sumSquareEnergy')
    plt.show()