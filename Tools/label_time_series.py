import os

import pandas as pd


def label_time_series(subject, labels, raw_df, inclusion=None):
    """convert the time series label data label to a label series of raw data.

        Parameters
        ----------
        subject:        string
        labels:         dataFrame
        raw_df:         dataFrame
        inclusion:      dataFrame[Optional]

        Return
        ------
        raw_dataframe:  dataFrame
        labels:         dataFrame

        """

    raw_df["Time"] = pd.to_datetime(raw_df["Time"], unit='ms')
    raw_df["Time"] = raw_df["Time"].dt.tz_localize('utc').dt.tz_convert('US/Central') # convert time utc to local time
    raw_df = raw_df.sort_values(by=['Time'])

    # if inclusion file inclused first mask the data to only inclusion
    included_df = []
    if inclusion is not None:
        print("inclusion included")
        for index, row in inclusion.iterrows():
            print(row['start'], row['end'])
            mask = (raw_df['Time'] >= row['start']) & (
                    raw_df['Time'] <= row['end'])  # mask orginal dataframe to included times
            included_df.append(raw_df.loc[mask])
        raw_df = pd.concat(included_df, ignore_index=True)

    raw_df['Label'] = 0
    # maybe -1 as unknown label.
    # TODO how to handle unknown

    labels = labels.sort_values(by=['start'])

    # apply mask to df based on chewing labels
    for index, row in labels.iterrows():
        mask = (raw_df['Time'] >= row['start']) & (raw_df['Time'] <= row['end'])  # mask orginal dataframe to chewing labels
        raw_df.loc[mask, 'Label'] = 1  # set chewing label on df

    # save label column as csv
    header = ["Label"]
    outdir = 'Output/Necksense'
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    # output to csv.
    #raw_df.to_csv('Necklace/{}_gesture_labels.csv'.format(subject), columns=header, header=None, index=False)
    #raw_df.to_csv('Necklace/{}_raw.csv'.format(subject), index=False)

    # save only inclusion dataframe is csv
    return raw_df, raw_df["Label"]
