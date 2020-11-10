from __future__ import division
import numpy as np
from scipy import signal

np.seterr(all='ignore')


def assert_monotonic(x):
    """
    Assertion for a monotonic
    Parameters
    ----------
    x: numpy array
        Input data
    """
    if not np.all(np.diff(x) >= 0):
        raise Exception("Not monotonic")


def assert_vector(x):
    """
    Assertion for a vector
    Parameters
    ----------
    x: numpy array
        Input data
    """
    if not type(x) is np.ndarray and x.ndim != 1:
        raise Exception("Not a vector")


def peak_detection(arr, threshold=2, min_prominence=0.05):
    """
    Prominence based
    Parameters
    ----------
    arr: numpy array
        Input data
    threshold: number or ndarray or sequence
        Required threshold of peaks, the vertical distance to its neighbouring samples.
        Either a number, None, an array matching x or a 2-element sequence of the former.
        The first element is always interpreted as the minimal and the second, if supplied, as the maximal required threshold.
    min_prominence: number or ndarray or sequence,
        Required prominence of peaks. Either a number, None, an array matching x or a 2-element sequence of the former.
        The first element is always interpreted as the minimal and the second, if supplied, as the maximal required prominence.
    Returns:
        ndarray
        Indices of peaks in x that satisfy all given conditions.
    """
    peaks_index = signal.find_peaks(arr, threshold=threshold, prominence=min_prominence)[0].astype(int)

    return peaks_index


def get_periodic_stat(sequence):
    """
    Given a sequence, find its periodicity statistics
    Parameters
    ----------
    sequence: numpy array
        input sequence to find its periodicity statistics
    Returns
    -------
    periodicy_stats: dictionary
        pmin
        pmax
        eps
        length
    """
    assert_vector(sequence)
    assert_monotonic(sequence)

    periodicy_stats = {}

    diff = []
    for i in range(len(sequence) - 1):
        diff.append(sequence[i + 1] - sequence[i])

    periodicy_stats['pmin'] = min(diff)
    periodicy_stats['pmax'] = max(diff)
    periodicy_stats['eps'] = periodicy_stats['pmax'] / periodicy_stats['pmin'] - 1
    periodicy_stats['start'] = sequence[0]
    periodicy_stats['end'] = sequence[-1]
    periodicy_stats['length'] = len(sequence)

    return periodicy_stats


def periodic_subsequence(peaks_index, peaks_time, min_length=5, max_length=100, eps=0.15, alpha=0.1, low=500,
                         high=1000):
    """
    Find the periodic subsequence in sequence peaks_index and return the start and end times from peaks_time
    Parameters
    ----------
    peaks_index: list
        list of peak index
    peaks_time: list
        list of timestamps
    min_length: int
        minimum length of the subsequence
    max_length: int
        maximum length of the subsequence
    eps: float or int
        periodicity attribute
    alpha: float or int
        error bound percentage of upper margin
    low: int
        lower bound for series of p_min
    high: int
        upper bound for series of p_max
    Returns
    -------
    subsequences: a list of numpy vector
        each vector is one subsequence that contains the index of periodic peaks
    """
    assert_vector(peaks_time)
    assert_monotonic(peaks_time)

    subs_index = relative_error_periodic_subsequence(peaks_time, eps, alpha, low, high, min_length, max_length)

    subsequences = []
    for s in subs_index:
        tmp = [peaks_index[i] for i in s]
        subsequences.append(np.array(tmp))

    return subsequences


def relative_error_periodic_subsequence(sequence, eps, alpha, low, high, min_length, max_length):
    """
    Approximation algorithm that find eps-periodic subsequences
    Parameters
    ----------
    sequence: list
        sequence data which herein is a list of peak index
    eps: float or int
        periodicity attribute
    alpha: float or int
        error bound percentage of upper margin
    low: int
        lower bound for series of p_min
    high: int
        upper bound for series of p_max
    min_length: int
        minimum length of the subsequence
    max_length: int
        maximum length of the subsequence
    Returns
    -------
    a list of numpy vector
        each vector is one subsequence that contains the index of periodic peaks
    """
    assert_vector(sequence)
    assert_monotonic(sequence)

    subsequences = []

    n_steps = np.ceil(np.log(high / low) / np.log(1 + eps)).astype(int)
    for i in range(n_steps):
        pmin = low * np.power((1 + eps), i)
        pmax = pmin * (1 + eps) * (1 + alpha)

        if pmax > high:
            break

        seqs = absolute_error_periodic_subsequence(sequence, pmin, pmax)
        seqs = [np.array(s) for s in seqs if len(s) > min_length and len(s) < max_length]

        subsequences += seqs

    # sort subsequences by its start time
    start = [seq[0] for seq in subsequences]

    subsequences = [seq for _, seq in sorted(zip(start, subsequences), key=lambda pair: pair[0])]

    return subsequences


def absolute_error_periodic_subsequence(sequence, pmin, pmax):
    """
    Return longest subsequences which is periodic, satisfying the interval constraint by [pmin, pmax].
    Using dynamic programming approach, as illustrated in paper Figure 9.
    Parameters
    ----------
    sequence: list
        list of increasing numbers
    pmin: float, int
        minimum interval constraint
    pmax: float, int
        maximum interval constraint
    Returns
    -------
    a list of numpy vector
        each vector is one periodic subsequence satisfying the interval constraint by [pmin, pmax].
    """

    assert_vector(sequence)
    assert_monotonic(sequence)

    N = len(sequence)

    traceback = {}

    # create a list for each point to store longest periodic sequence
    for i in range(N):
        traceback[i] = []

    for i in range(1, N):
        valid = []  # store valid starting point of periodic sequence
        # go through all the points ahead of i and store valid starting point of periodic sequence
        for j in range(i - 1, -1, -1):
            # if satisfy the [pmin, pmax] constraint then store in 'valid' list
            if sequence[i] - sequence[j] > pmax:
                break
            if sequence[i] - sequence[j] >= pmin:
                valid.append(j)

        # reverse valid list to starting from minimum
        valid = list(reversed(valid))

        # now find valid predecessor for i
        for j in valid:
            # if the first time in the loop, i.e., traceback[j] empty
            if not traceback[j]:
                opt_len = 2  # optimal length is 2 points for the first time
            else:  # if not the first time in the loop
                # 'opt_len' stores optimal length
                opt_len = traceback[j][0]['opt_len'] + 1  # optimal length +1 based on record in traceback[j]

            predecessor = {'prev': j, 'opt_len': opt_len}

            index_revisit = []
            # save the predecessors with optimal length greater than 'predecessor['opt_len']' for revisit
            for k in range(len(traceback[i])):
                if traceback[i][k]['opt_len'] >= predecessor['opt_len']:
                    index_revisit.append(k)

            # store the predecessors with optimal length greater than or equal to 'predecessor['opt_len']'
            traceback[i] = [traceback[i][k] for k in index_revisit]
            traceback[i].append(predecessor)

    subsequences = []
    sequence = []
    i = N - 1

    # form final periodic subsequences
    while i >= 0:
        # if current i has predecessor
        if traceback[i]:
            # append current i to sequence
            sequence.append(i)
            # trace back to the predecessor of current i
            i = traceback[i][0]['prev']
        # if current i has no predecessor
        else:
            # if sequence is not empty, i.e. current i is in a sequence
            if len(sequence) > 0:
                # append current i
                sequence.append(i)
                reverse = list(reversed(sequence))
                subsequences.append(reverse)
                sequence = []
            # go to the next point (in the left)
            i -= 1

    return list(reversed(subsequences))


def peak_unittest():
    peaks_index = np.array([1, 11, 16, 19, 22, \
                            26, 33, 37, 39, 41, \
                            45, 50, 58, 70, 77, \
                            79, 87, 106, 124, 128])

    peaks_time = np.array([64660, 65160, 65410, 65560, 65710, \
                           65910, 66260, 66460, 66560, 66660, \
                           66860, 67110, 67510, 68110, 68460, \
                           68560, 68960, 69910, 70810, 71010])

    subsequences = periodic_subsequence(peaks_index, peaks_time, min_length=4, max_length=100,
                                        eps=0.1, alpha=0.45, low=400, high=1200)
    # print(subsequences)
    # expected value for unit test
    expected = [np.array([1, 11, 22, 33, 41, 50, 58, 70, 79, 87]),
                np.array([1, 11, 22, 33, 45, 58, 70]),
                np.array([1, 11, 22, 33, 45, 58, 70]),
                np.array([1, 16, 33, 45, 58, 70, 87]),
                np.array([1, 16, 33, 50, 70, 87, 106, 124]),
                np.array([1, 16, 33, 50, 70, 87, 106, 128]),
                np.array([11, 22, 33, 45, 58, 70])]

    for i in range(len(subsequences)):
        np.testing.assert_array_equal(subsequences[i], expected[i], err_msg='Not equal peaks', verbose=True)


def main():
    peak_unittest()


if __name__ == '__main__':
    main()