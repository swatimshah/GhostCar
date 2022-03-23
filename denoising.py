from numpy import loadtxt
from scipy import signal
from numpy import savetxt 
import numpy
import matplotlib.pyplot as plt


csv_file = "ghostcar_signal_50Hz_outliers_zero_mean.csv"

timestamps = loadtxt('D:\\recording\\EEG_recording_2022-03-23-09.18.20.csv', delimiter=',', skiprows=1, usecols=(0))
print(timestamps.shape)
validTimestamps = timestamps[1500:,]

#=============================================================


tp9 = loadtxt('D:\\recording\\EEG_recording_2022-03-23-09.18.20.csv', delimiter=',', skiprows=1, usecols=(1))
print(tp9.shape)
tp9 = tp9[1500:,]
b_notch, a_notch = signal.iirnotch(50, 30, 256)
outputSignal = signal.filtfilt(b_notch, a_notch, tp9)
#fig = plt.figure(figsize =(10, 7))
#plt.boxplot(outputSignal)
too_high = outputSignal > 200.
outputSignal[too_high] = 200.
too_low = outputSignal < -350.
outputSignal[too_low] = -350.
outputSignalMedian = numpy.median(outputSignal)
tp9_zero_mean = outputSignal - outputSignalMedian

#============================================================

af7 = loadtxt('D:\\recording\\EEG_recording_2022-03-23-09.18.20.csv', delimiter=',', skiprows=1, usecols=(2))
print(af7.shape)
af7 = af7[1500:,]
b_notch, a_notch = signal.iirnotch(50, 30, 256)
outputSignal = signal.filtfilt(b_notch, a_notch, af7)
#fig = plt.figure(figsize =(10, 7))
#plt.boxplot(outputSignal)
too_high = outputSignal > 200.
outputSignal[too_high] = 200.
too_low = outputSignal < -300.
outputSignal[too_low] = -300.
outputSignalMedian = numpy.median(outputSignal)
af7_zero_mean = outputSignal - outputSignalMedian

#==========================================================

af8 = loadtxt('D:\\recording\\EEG_recording_2022-03-23-09.18.20.csv', delimiter=',', skiprows=1, usecols=(3))
print(af8.shape)
af8 = af8[1500:,]
b_notch, a_notch = signal.iirnotch(50, 30, 256)
outputSignal = signal.filtfilt(b_notch, a_notch, af8)
#fig = plt.figure(figsize =(10, 7))
#plt.boxplot(outputSignal)
too_high = outputSignal > 200.
outputSignal[too_high] = 200.
too_low = outputSignal < -250.
outputSignal[too_low] = -250.
outputSignalMedian = numpy.median(outputSignal)
af8_zero_mean = outputSignal - outputSignalMedian

#=========================================================

tp10 = loadtxt('D:\\recording\\EEG_recording_2022-03-23-09.18.20.csv', delimiter=',', skiprows=1, usecols=(4))
print(tp10.shape)
tp10 = tp10[1500:,]
b_notch, a_notch = signal.iirnotch(50, 30, 256)
outputSignal = signal.filtfilt(b_notch, a_notch, tp10)
#fig = plt.figure(figsize =(10, 7))
#plt.boxplot(outputSignal)
too_high = outputSignal > 150.
outputSignal[too_high] = 150.
too_low = outputSignal < -300.
outputSignal[too_low] = -300.
outputSignalMedian = numpy.median(outputSignal)
tp10_zero_mean = outputSignal - outputSignalMedian

#========================================================

combinedData_1 = numpy.append(tp9_zero_mean.reshape(len(tp9_zero_mean), 1), af7_zero_mean.reshape(len(af7_zero_mean), 1), axis=1)
combinedData_2 = numpy.append(combinedData_1, af8_zero_mean.reshape(len(af8_zero_mean), 1), axis=1)
combinedData_3 = numpy.append(combinedData_2, tp10_zero_mean.reshape(len(tp10_zero_mean), 1), axis=1)
combinedData = numpy.append(combinedData_3, validTimestamps.reshape(len(validTimestamps), 1), axis=1)


savetxt(csv_file, combinedData, delimiter=",", header="TP9, AF7, AF8, TP10")
plt.show()
