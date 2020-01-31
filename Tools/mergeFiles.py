import pandas as pd
import numpy as np
import time

def mergeIMUFiles(files,output):
    pd.set_option('display.max_columns', None)
    AccelFile = files[0]#'C:\\Users\\Sougata\\Downloads\\20200130_103850_Accelerometer.csv'
    GyroFile = files[1]#'C:\\Users\\Sougata\\Downloads\\20200130_103850_Gyroscope.csv'
    MagnetFile = files[2]#'C:\\Users\\Sougata\\Downloads\\20200130_103850_Magnetometer.csv'

    #read the three files
    accelData = pd.read_csv(AccelFile,names=['Date','HostTimestamp (ms)','NodeName','NodeTimestamp','RawData','X (mg)','Y (mg)','Z (mg)'])
    gyrodata = pd.read_csv(GyroFile,names=['Date','HostTimestamp (ms)','NodeName','NodeTimestamp','RawData','X (dps)','Y (dps)','Z (dps)'])
    magnetometerData = pd.read_csv(MagnetFile,names=['Date','HostTimestamp (ms)','NodeName','NodeTimestamp','RawData','X (mGa)','Y (mGa)','Z (mGa)'])

    # merge accel and gyro data based on Date
    mergeAccelAndGyro = pd.merge(accelData,gyrodata,on='Date')
    #merge accel+ gyro data with magnetometer data, based on date
    mergeAccelGyroMagn = pd.merge(mergeAccelAndGyro,magnetometerData,on='Date')

    # Take only the relevant rows
    mergedDataFrame = mergeAccelGyroMagn[['Date','X (mg)','Y (mg)','Z (mg)','X (dps)','Y (dps)','Z (dps)','X (mGa)','Z (mGa)','Z (mGa)']][4:]


    # convert human readable time to epoch time
    utc_time =  pd.to_datetime(mergedDataFrame['Date'],infer_datetime_format=True)# format='%d/%m/%Y %H:%M:%S.%fZ')#,                                 "%d-%m-%YT%H:%M:%S.%fZ")
    pd.DatetimeIndex(utc_time)
    utc_time = utc_time.astype(np.int64) / int(1e6)
    print(utc_time[4])
    mergedDataFrame['Date'] = utc_time

    #write it to the file in data2R folder
    mergedDataFrame.to_csv (output, index = None, header=True)


if __name__=='__main__':
    output = '../Data2R/Repetition1_classActivity_data.csv'
    files = ['C:\\Users\\Sougata\\Downloads\\20200130_103850_Accelerometer.csv','C:\\Users\\Sougata\\Downloads\\20200130_103850_Gyroscope.csv','C:\\Users\\Sougata\\Downloads\\20200130_103850_Magnetometer.csv']
    mergeIMUFiles(files,output)