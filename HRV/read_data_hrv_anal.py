#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 11:57:24 2017

@author: Rebeca
"""

import numpy as np
import matplotlib.pyplot as plt
from qrs_detector import *
from HRV import HRV
from HRV_Entropy import HRV_entropy


f = np.loadtxt("sigs/opensignals_201607181603_2018-01-09_13-14-55.txt")
#f = np.loadtxt("opensignals_Oscar_2017-10-17_12-27-32.txt")

ecg=f[:,6];
#eda=f[:,6];
#marcas=f[:,1];

fs=1000;
ecg_d,t = detrendSpline(ecg,fs,l_w = 1.2)
ecg_filtered = bandpass_qrs_filter(ecg_d, fs, fc1 = 5,fc2 = 15)
beat, th, qrs_index= exp_beat_detection(ecg_filtered,fs,Tr = .180,a = .7,b = 0.999)
r_peak, rr = r_peak_detection(ecg_filtered,ecg,fs,beat,th,qrs_index,Tr = .100)

t=np.arange(0,len(ecg_d))*fs
plt.figure(), plt.plot(t,ecg_d)
plt.figure(), plt.plot(rr)

#t_eda=np.arange(0,len(eda))*fs
#plt.figure(), plt.plot(t_eda,eda)

#HRV Analysis

labels=['N']*len(rr)
hrv_anal = HRV()
prct = 0.2
ind_not_N_beats=hrv_anal.artifact_ectopic_detection(rr, labels, prct, numBeatsAfterV = 4)
valid = hrv_anal.is_valid(ind_not_N_beats,perct_valid = 0.2)
rr_corrected = hrv_anal.artifact_ectopic_correction(rr, ind_not_N_beats, method='cubic')
plt.figure()
plt.plot(rr_corrected)
hrv_pat = hrv_anal.load_HRV_variables(rr_corrected)
avnn = hrv_anal.avnn(rr_corrected)

r = np.std(rr_corrected)*0.2
hrv_en = HRV_entropy()
sampen = hrv_en.SampEn(rr_corrected,m = 2,r=r,kernel = 'Heaviside')

