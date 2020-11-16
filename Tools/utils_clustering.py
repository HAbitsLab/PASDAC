import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
from collections import namedtuple
from sklearn.cluster import DBSCAN


def perdelta(start, end, delta):
    """
    Given start and end, yield increasing time step

    Parameters
    ----------
    start: start time
    end: end time
    delta: stride size

    Returns:
    --------
    increasing time step

    """
    curr = start
    while curr < end:
        yield curr
        curr += delta


def group_meal_detected(group1, group2, threshold=0.5):
    """
    Given ground truth episode and prediction episode, return detected results

    Parameters
    ----------
        group1:     the episodes to check against group2, prediction groups
                    a list of [start (datetime), end (datetime)]
        group2:     the episodes used for checking group1, ground truth groups
                    a list of [start (datetime), end (datetime)]
        threshold:  the threshold to judge if one episode is detected,
                    if one episode in group1 has more than $threshold$ covered by any episode in group2,
                    then it is defined as detected.
    Returns
    -------
        count: int
            number of detected group1 episodes
        epi_detected: list
            the episodes in group1 that are detected by group2
        epi_missed: list
            the episodes in group1 that are NOT detected by group2

    """
    epi_detected = []
    epi_missed = []
    count = 0
    overlap_sec = 0
    for interval1 in group1:
        detected_flag = 0
        for interval2 in group2:
            if overlap_seconds_ratio(interval1, interval2) > threshold:
                detected_flag = 1
                continue
        if detected_flag == 1:
            epi_detected.append(interval1)
            count += 1
        else:
            epi_missed.append(interval1)

    return count, epi_detected, epi_missed


def overlap_seconds(interval1, interval2):
    """
    Given two intervals, return how many seconds are overlapped

    Parameters
    ----------
    interval1: the first interval
    interval2: the second interval

    Returns
    -------
    int, number of seconds that are overlapped

    """
    dt_s1 = interval1[0]
    dt_e1 = interval1[1]
    dt_s2 = interval2[0]
    dt_e2 = interval2[1]

    Range = namedtuple("Range", ["start", "end"])
    r1 = Range(start=dt_s1, end=dt_e1)
    r2 = Range(start=dt_s2, end=dt_e2)
    latest_start = max(r1.start, r2.start)
    earliest_end = min(r1.end, r2.end)
    overlap_sec = (earliest_end - latest_start).total_seconds()
    overlap_sec = max(0, overlap_sec)

    return overlap_sec


def overlap_seconds_ratio(interval1, interval2):
    """
    Given two intervals, return the ratio of seconds that are overlapped to interval1

    Parameters
    ----------
    interval1: the first interval
    interval2: the second interval

    Returns
    -------
    float, ratio of seconds that are overlapped to interval1

    """
    overlap_sec = overlap_seconds(interval1, interval2)
    interval1_sec = (interval1[1] - interval1[0]).total_seconds()

    # after truncating millisecond, interval1[1] may equal to interval1[0],
    # then need to round up to 1 second
    if interval1_sec == 0:
        interval1_sec = 1
    return overlap_sec / interval1_sec


def score_to_spread_list(score):
    """
    Converting from an array of scores to an array of index of non-zero positive points,
        for example, from [0,0,0,1,1,0,0,1] to [3,4,7]

    Parameters
    ----------
    score: numpy array
        input data

    Returns
    -------
    numpy array
        converted array of index of non-zero positive points

    """
    pointwise_positive_list = np.where(score > 0)[0].tolist()
    pointlist = [
        [point] * time
        for point, time in zip(
            pointwise_positive_list, score[np.where(score > 0)].tolist()
        )
    ]
    pointlist = [item for sublist in pointlist for item in sublist]  # ravel

    return np.array(pointlist).reshape(-1, 1)


def remove_small_auc_subseq(score, minpts_parameter, eps_parameter):
    """
    Removing small sparse subsequence from score and converting from integer array to boolean array, for example
        convert from 00000223010 to 00000111000, if the last second digit '1' is considered as noise
        under the selection of minpts and eps.

    (DBSCAN takes the role of a filter)

    Parameters
    ----------
    score: numpy array
        input array with each point representing
    minpts_parameter: int
        DBSCAN minpts parameter
    eps_parameter: int
        DBSCAN eps parameter

    Returns
    -------
    numpy array
        score array composing of only 0 and 1 after removing sparse subsequence considered as noise under DBSCAN

    """
    # noise removal with sklearn DBSCAN
    # score: [0,0,0,1,2,0,0,1]
    pointwise_positive_predicted_array = score_to_spread_list(score)
    # pointwise_positive_predicted_array: [3,4,4,7]

    _dbscan = DBSCAN(min_samples=minpts_parameter, eps=eps_parameter)
    _dbscan.fit(pointwise_positive_predicted_array)
    clustering_labels_list = _dbscan.labels_.tolist()
    # clustering_labels_list: [0,0,0,-1]
    new_pointwise_positive_predicted_array = [
        pointwise_positive_predicted_array.ravel()[i]
        for i, x in enumerate(clustering_labels_list)
        if x != -1
    ]

    score_bool = np.zeros_like(score)
    score_bool[new_pointwise_positive_predicted_array] = 1

    return score_bool


def pw_fusion(data_df, output_column, starttimes, endtimes, preds):
    """
    Merging predicted chewing subsequence by naive pointwise merging and generate a score for each second

    Parameters
    ----------
    data_df: dataframe
        prediction result with segment start and end and prediction

    output_column: str
        the column name where the pointwise score will be stored

    starttimes: datetime
        start time of each segment

    endtimes: datetime
        end time of each segment

    preds: numpy array
        the prediction for all the segments

    Returns
    -------
    pandas DataFrame
        Description of dataframe:
        ==========          ==============================================================
        type                description
        time                datetime in seconds
        score               number of occurances of chewing prediction for each second
        ==========          ==============================================================
    """
    for st, end, pred in zip(starttimes, endtimes, preds):
        if pred == 1:
            indices = data_df.index[(data_df.index >= st) & (data_df.index <= end)].tolist()
            for ind in indices:
                data_df[output_column][ind] = data_df[output_column][ind] + 1
    return data_df


def get_pointwise_scored_df(raw_df):
    """
    Converting from start-end-pair dataframe to pointwise dataframe.

    Parameters
    ----------
    raw_df: pandas DataFrame
        start-end-pair dataframe

    Returns
    -------
    pandas DataFrame
        Description of dataframe:
        ==========          ==============================================================
        type                description
        time                datetime in seconds
        score               number of occurances of chewing prediction for each second
        ==========          ==============================================================

    """
    # create a dataframe with all the seconds in this day, from the first predicted chewing sequence to the last one.
    earliest = raw_df["start"].min().replace(microsecond=0)
    if raw_df["end"].max().microsecond:
        latest = raw_df["end"].max().replace(microsecond=0) + timedelta(seconds=1)
    else:
        latest = raw_df["end"].max()

    pointwise_df = pd.DataFrame()
    pointwise_df["time"] = pd.date_range(start=earliest, end=latest, freq="1s")
    pointwise_df["score"] = 0
    pointwise_df = pointwise_df.set_index("time")

    starttimes = raw_df["start"].tolist()
    endtimes = raw_df["end"].tolist()
    starttimes = [i.replace(microsecond=0) for i in starttimes]
    endtimes = [
        i.replace(microsecond=0) + timedelta(seconds=1) if i.microsecond else i
        for i in endtimes
    ]
    preds = raw_df["prediction"].values

    pointwise_df = pw_fusion(pointwise_df, "score", starttimes, endtimes, preds)

    return pointwise_df


def expend_dur_to_sec_list(gt_df):
    """
    Expanding duration in a dataframe to a second list

    Parameters
    ----------
    gt_df: pandas DataFrame
        containing the ground truth chewing sequence start and end time

    Returns
    -------
    a list, containing all the datetime seconds of ground truth chewing sequences

    """
    gt_sec_list = []
    for (s, e) in zip(gt_df["start"].tolist(), gt_df["end"].tolist()):
        s = s.replace(microsecond=0)
        e = e.replace(microsecond=0)
        for result in perdelta(s, e, timedelta(seconds=1)):
            gt_sec_list.append(result)

    return gt_sec_list


def merge_seq_to_epi(pointwise_df, score_boolean, threshold, remove_short_dur_threshold=60):
    """
    Merging chewing sequences (chewing bouts) to eating episodes.

    Parameters
    ----------
    pointwise_df: pandas DataFrame
        pointwise DataFrame with index as datetime seconds

    score_boolean: numpy array
        an array of score (0 or 1) for each second representing if this second is chewing or non-chewing

    threshold: float
        distance threshold for merging two consecutive chewing sequences into one eating episode

    remove_short_dur_threshold: int or float
        the threshold for a noise 'episode' that is too short as a meal or real episode episode

    Returns
    -------
    pred_epi_sec_list: list
        list of all the seconds that are predicted as eating

    pred_epi_dur_df_concat_new: list
        list of pandas DataFrames each with an eating episode start and end time

    """
    pred_epi_sec_list = []
    pred_epi_dur_df_concat = []
    pred_epi_dur_df_concat_new = []
    # run DBSCAN again with params (eps,minpts) and fill in the gaps within one group.
    # score_boolean example: [0,0,0,1,1,0,0,1]
    # pointwise_positive_predicted_array: [3,4,7]
    pointwise_positive_predicted_array = score_to_spread_list(score_boolean)

    if pointwise_positive_predicted_array.size:  # avoid input to DBSCAN is empty
        _dbscan = DBSCAN(min_samples=1, eps=threshold)
        _dbscan.fit(pointwise_positive_predicted_array)
        clustering_labels = _dbscan.labels_

        # get set of labels, ie, identical groups (clusters)
        clustering_labels_list = clustering_labels.tolist()
        cluster_label_set = set(clustering_labels)

        # convert to list of predicted positive seconds (points)
        pointwise_positive_predicted_list = (pointwise_positive_predicted_array.ravel().tolist())

        # loop through all the clusters
        for label in set(cluster_label_set):
            # if not noise
            if label != -1:
                pointwise_positive_predicted_list_new = [e for i, e in enumerate(pointwise_positive_predicted_list)
                    if clustering_labels_list[i] == label]
                start = min(pointwise_positive_predicted_list_new)
                end = max(pointwise_positive_predicted_list_new)
                time_point_second = np.array(pointwise_df.index.tolist())

                # note: pandas cannot create DataFrame from numpy array of time_stamps
                # #13287. from: https://github.com/pandas-dev/pandas/issues/13287
                dur_df = pd.DataFrame.from_records(
                    data=np.array(
                        [time_point_second[start], time_point_second[end]]
                    ).reshape(1, 2),
                    columns=["start", "end"],
                )
                pred_epi_dur_df_concat.append(dur_df)

                # prepare for jaccard calculation, replace millisecond with 0
                for i in range(len(dur_df)):
                    start_time = dur_df["start"].iloc[i]
                    end_time = dur_df["end"].iloc[i]
                    if end_time - start_time > timedelta(seconds=remove_short_dur_threshold):
                        for result in perdelta(start_time, end_time, timedelta(seconds=1)):
                            pred_epi_sec_list.append(result)
            else:
                print(
                    "Error happens: cannot have -1 in cluster_label_set when script param minpts_parameter < threshold"
                )
                raise
                exit()

    for df in pred_epi_dur_df_concat:
        df["delta"] = df["end"] - df["start"]
        df = df[df["delta"] > timedelta(seconds=remove_short_dur_threshold)]
        if len(df) > 0:
            pred_epi_dur_df_concat_new.append(df)

    return pred_epi_sec_list, pred_epi_dur_df_concat_new


def period_overlap_eval(gt_df, pred_df, threshold=0.5):
    """

    Parameters
    ----------
    gt_df: a list of [start (datetime), end (datetime)]
        ground truth episodes

    pred_df: a list of [start (datetime), end (datetime)]
        prediction episodes

    threshold: float
        the threshold to judge if one episode is detected,
        if one episode in group1 has more than $threshold$ covered by any episode in group2,
        then it is defined as detected.

    Returns
    -------
        cnt_epi_detected: int
            number of detected episodes from ground truth by prediction

        epi_detected: list
            the detected episodes from ground truth by prediction

        epi_missed: list
            the missed episodes from ground truth by prediction

        cnt_truepostive: int
            number of true postives

        epi_truepositive: list
            true postives

        epi_falsepositive: list
            false positives

    """

    ground_truth = []
    gt_second_list = []
    for (start, end) in zip(gt_df["start"].tolist(), gt_df["end"].tolist()):
        start = start.replace(microsecond=0)
        end = end.replace(microsecond=0)
        ground_truth.append([start, end])
        for result in perdelta(start, end, timedelta(seconds=1)):
            gt_second_list.append(result)

    pred = []
    if len(pred_df) > 0:
        for (start, end) in zip(pred_df["start"].tolist(), pred_df["end"].tolist()):
            start = start.replace(microsecond=0)
            end = end.replace(microsecond=0)
            pred.append([start, end])
    # recall
    cnt_epi_detected, epi_detected, epi_missed = group_meal_detected(
        ground_truth, pred, threshold=threshold
    )
    # precision
    cnt_truepostive, epi_truepositive, epi_falsepositive = group_meal_detected(pred, ground_truth, threshold=threshold)

    return cnt_epi_detected, epi_detected, epi_missed, cnt_truepostive, epi_truepositive, epi_falsepositive


def unittest_group_meal_detected():
    # TODO: docstring test - TIM
    group1 = [datetime.now(), datetime.now() + timedelta(hours=1)]
    print((group1[1] - group1[0]).total_seconds())
    group2 = []
    cnt_epi_detected, epi_detected, epi_missed = group_meal_detected(group1, group2)
    print(cnt_epi_detected, epi_detected, epi_missed)


if __name__ == "__main__":
    unittest_group_meal_detected()
