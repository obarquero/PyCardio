# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 07:08:22 2014

@author: obarquero
"""

#first import all the modules we neeed

from __future__ import division
import numpy as np
import scipy as sc
from scipy.interpolate import interp1d
from scipy import signal
import matplotlib.pylab as plt

##########################################
#   Base line removal
#########################################

def detrendSpline(signal,fs,l_w = 1.2):
    """ input signal: ECG signal to be detrended
              fs: sampliing frequency
              l_w: window length in secs (1 sec by default)
        output ecg_detrend: ecg without the base line
               s_pp: trend
    """
    L_s = len(signal)
    t = np.arange(0,L_s)*1./fs
    
    numSeg = np.floor(t[-1]/l_w)

    s_m = np.zeros(numSeg)
    t_m = np.zeros(numSeg)
    
    for k in np.arange(numSeg):
        #for over each window and compute the median
        ind_seg = (t >= (k)*l_w) & (t <= (k+1)*l_w)
        t_aux = t[ind_seg]
        t_m[k] = t_aux[np.round(t_aux.size/2)]
        s_m[k] = np.median(signal[ind_seg]);
   
    #fit the spline to the median points
    #Add first and last value in
    t_m = np.concatenate(([0],t_m,[t[-1]]))
    s_m = np.concatenate(([signal[0]],s_m,[signal[-1]]))
    cp = interp1d(t_m,s_m,kind = 'cubic')
    trend = cp(t)
    ecg_detrend = signal - trend;
    return ecg_detrend,trend

###############################################################################

###############################################################################
# Band pass filtering
###############################################################################

def bandpass_qrs_filter(ecg, fs, fc1 = 5,fc2 = 15):
    """input fc1: cut frequency 1 in Hz
             fc2: cut frequency 2 in Hz
             fs: sampling frequency in Hz
    
        output ecg_filt: filtered ecg
    """
    fn = fs/2. #Nyquist frequency
    N = 128 #filter order
    
    b = signal.firwin(N, [fc1/fn, fc2/fn], pass_zero = False)
    
    #zero phase filter
    
    ecg_filtered = signal.filtfilt(b,np.arange(1,2),ecg)
    return ecg_filtered

###############################################################################
# Exponential threshold beat detection
###############################################################################
def exp_beat_detection(ecg,fs,Tr = .180,a = .7,b = 0.999):
    """ input ecg: ecg to be detected
              Tr: refractory period in seconds. Default 180 ms
              a: correction fraction of the threshold with respecto to the
              maximun of the R peak. Default 0.7
              b: exponential decay. Default 0.999
              
        output beat: beat detected
               th: threshold function
               qrs_index: qrs index  
    """
    Tr = np.floor(Tr*fs)
    end = len(ecg) #end point
    maximum_value = np.max(ecg/3.)
    
    beat = np.zeros((end,1))
    qrs_index = []
    th = []
    th.append(a*max(ecg)) # threshold initialization 
    
    detect = Tr*(-1) + 1 #variable that controls the refractory period
    
    for k in range(1,end):
        #for over all the points in the ECG
        if (ecg[k] > th[k-1]) & (k > (detect + Tr)):
            detect = k #new refractory period
            qrs_index.append(k)
            beat[k] = maximum_value
            
        #if no detection, update the threshold
        update_th = np.max((a*ecg[k],b*th[k-1]))
        th.append(update_th)
        
    return beat, th, qrs_index

###############################################################################
# R-peak detection
###############################################################################    

def r_peak_detection(ecg_filtered,ecg_original,fs,beat,th,qrs_index,Tr = .100):
    """ r-peak detection in the original ecg. Once, the QRS is detected in the
    filtered ecg, this functions allows to estimate de position of the r-peak
    
    inputs ecg_filtered: ecg filtered on which the qrs detection is performed.
           ecg_original: original ecg
           fs: sampling frequency.
           beat: qrs detections
           th: threshold function that allowed the detection of qrs complexes
           qrs_index: indices of the qrs in the filtered ecg.
           
    outputs
            r_peaks: position of the r_peaks on the original ecg
            rr: RR-interval time series
    """    
    Tr_s = np.round(Tr * fs)
    r_peaks = np.zeros(len(ecg_filtered))

    for idx in qrs_index:
        
        #build an r-peak search window around the qrs_complex detected
        window = range(idx-3,idx + int(Tr_s/2))
        
        if (window[0] > 0) & (window[-1] < len(ecg_filtered)):
            beat_original = ecg_original[window]
            idx_rpeak = np.argmax(beat_original)
            
            r_peaks[idx - 3 + idx_rpeak] = 1
    #end of for        
    r_peak = np.nonzero(r_peaks)[0]
    rr = (np.diff(r_peak)/fs)*1000.
    return r_peak, rr
#load ecg

#ecg = np.loadtxt("opensignals_201510266565_2016-11-16_12-51-32.txt",delimiter = ',')
#ecg = ecg[:,7]
#ecg = np.loadtxt("4554_5054994_Data.txt",delimiter = ',')
#fs = 500.
#fs = 1000.
#gain = 200.
#t = np.arange(0,len(ecg))/fs
#ecg_1 = ecg[:]/max(ecg[:])
#
#
#
##detrend 
#ecg_d,trend = detrendSpline(ecg_1,fs)
#plt.plot(t,ecg_1)
#plt.plot(t,trend,color='r',linewidth = 3)
#plt.xlabel('Time (sec)')
#plt.ylabel('ECG (mV)')
##bandpass filter
#ecg_filtered = bandpass_qrs_filter(ecg_d,fs = fs,fc1 = 8,fc2 = 20)
#
#plt.figure()
#plt.subplot(211)
#plt.plot(t,ecg_d)
#plt.xlabel('Time (sec)')
#plt.ylabel('ECG (mV)')
#plt.subplot(212)
#plt.plot(t,ecg_filtered)
#plt.xlabel('Time (sec)')
#plt.ylabel('ECG (mV)')
##plt.show()
#
###non linear transformation
#ecg_filtered = np.abs(ecg_filtered)
##exponential detection
#beat,th, qrs_index = exp_beat_detection(ecg_filtered,fs)
#r_peak, rr = r_peak_detection(ecg_filtered, np.abs(ecg_1) ,fs,beat,th,qrs_index)
##
#plt.figure()
#plt.plot(t,ecg_filtered)
#plt.plot(t,th)
#plt.plot(t[qrs_index],ecg_filtered[qrs_index],'r*')
#plt.xlabel('Time (sec)')
#plt.ylabel('ECG (mV)')
#
#plt.figure()
#plt.subplot(211)
#plt.plot(t,ecg_1)
#plt.plot(t[r_peak],ecg_1[r_peak],'r*')
#plt.xlabel('Time (sec)')
#plt.ylabel('ECG (mV)')
#plt.subplot(212)
#plt.plot(rr,'.-')
#plt.xlabel('# of beat')
#plt.ylabel('RR-interval (ms)')
#plt.show()
#
#######
##file_path = '/Users/obarquero/Dropbox/AASM_MasterIB/Tema_2/signals/ecg/pat1.dat'
#
##ecg_line = np.loadtxt(file_path)
##fs_new = 300.
##ecg_d,trend = detrendSpline(ecg_line,fs)
#
##plt.figure()
##plt.plot(ecg_line)
##plt.plot(trend,'r',linewidth = 3)
#
##plt.figure()
##plt.plot(ecg_d)
##plt.show()
#
#
########################################
## PCA
########################################
##from sklearn.decomposition import PCA
##
##
##pca_1 = PCA()
##X_pca = pca_1.fit_transform(ecg[:,1:])
##
##ecg_free = np.loadtxt("2989662.txt",delimiter = ',')
##pca_2 = PCA()
##X_pca_2 = pca_2.fit_transform(ecg_free[:,1:])
##
##plt.figure()
##plt.subplot(121)
##for n in range(12):
##    plt.plot(ecg[:,n+1]+n*200,color = 'k')
##    
##plt.subplot(122)
##for m in range(11):
##    plt.plot(X_pca[:,11-m]+m*200,color = 'k')
##
##plt.figure()
##plt.subplot(121)
##for n in range(12):
##    plt.plot(ecg_free[:,n+1]+n*200,color = 'k')
##    
##plt.subplot(122)
##for m in range(11):
##    plt.plot(X_pca_2[:,11-m]+m*200,color = 'k')
##plt.show()