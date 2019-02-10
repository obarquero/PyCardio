# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este archivo temporal se encuentra aquí:
C:\Users\Vierczy\.spyder2\.temp.py
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import special, optimize, signal


       
def RR(signal_in,fs_input = 500,fs_output = 500):
## 
#ind.rr_samples (matriz de muestas de rr por lead)
#ind.rr_time (matriz de tiempos de rr por lead)
#ind.HR (ritmo medio cardiaco)
#ind.signal_out (complta tras filtrado )
#ind.fs_out (500?)
##  Initializing variables and signal
#    addpath(genpath(pwd))
#    if nargin<3:
#        fs_output=500;


    maximum_frequency=200;  #bpm 
    min_rr= 60./maximum_frequency*1000.; # minimum rr in ms

    #Check size of signal 
    m=len(signal_in[:,1]);
    n=len(signal_in[1,:]);
    if (m<n):
        signal_in=signal_in.T;
    else:
        signal_in=signal_in;
     
    #Check frequency and convert to 500Hz

    if fs_output!=fs_input:
        signal_in= len(resample(signal_in,fs_output,fs_input)[1,:]);
        h=len(resample(signal_in,fs_output,fs_input)[:,1]);
        
    fs=fs_output; 
        # fs=working frequency
        
        ## Apply low-pass filter and detrend : CREATE ECGflat
    # Detrend: signal should be arrond 60bpm or 1 bps and then window trend equal to 1 # second should be applied
    # Low pas filter will be 150Hz and order 5;
    
    WindowdTrend=fs; # window for detrend spline
    LowPassOrder=5; # Low pass filter order
    LowPassCutOff=150; # Fequency for low pass filter
    if LowPassCutOff>=fs/2:
        no_filter=True;
    else:
        #b_LowPassFilter = signal.firwin(LowPassOrder+1, 2*LowPassCutOff/fs); 
        ##Óscar Change
        b_LowPassFilter = signal.firwin(LowPassOrder+1, 2*float(LowPassCutOff)/float(fs)); 
        no_filter=False;  # Coefficient for low pass filert 
    
    # createECGmean: Allocation
    n_samples= len(signal_in[:,1]);
    Nb_leads=len(signal_in[1,:]);
    
    #change for Óscar
#    ECGfilt = zeros((n_samples,Nb_leads));
 #   ECGflat = zeros((n_samples,Nb_leads));
    ECGfilt = np.zeros((n_samples,Nb_leads));
    ECGflat = np.zeros((n_samples,Nb_leads));
    
    # Compute ECGflat
    
    for lead in np.arange(0,Nb_leads,1):   
          # Filtering
            if no_filter:
                ECGfilt[:,lead]=signal_in[:,lead];          
            else:
                ECGfilt[:,lead]=signal.filtfilt(b_LowPassFilter,np.arange(1,2),signal_in[:,lead]);
          
            # Flattening
            #ECGflat[:,lead] , s_pp = detrendSpline(ECGfilt[:,lead],WindowdTrend,fs,n_samples);       
    
        ## Calculating ECG Deriv
     # Compute derivative by Green2006
    
    # we derive only non zero leads
    ECGfilt_exist=signal_in.sum(axis=0); # Array containing zeros on nonvalid leads
    if sum(ECGfilt_exist)==0:
        #TO_DO reformulate this in python 
        print 'Hola'
        #ind=[],texto='Invalid signal on case nº: '+ str(caso) + ', an empty output is provided', disp(texto)

    ECGfilt2deriv=ECGfilt[:,ECGfilt_exist!=0]; # ECGfilt2deriv contains only existing ECGfilt leads
    leads=np.where(ECGfilt_exist!=0)[0]
    valid_leads=np.array(leads);
    idxDeriv = np.arange(5,n_samples,1)
    ECGDeriv= np.zeros((len(idxDeriv)+5,len(valid_leads.T)));
    ECGDeriv[idxDeriv,:] = (2*ECGfilt2deriv[idxDeriv,:]+ECGfilt2deriv[idxDeriv-1,:]-ECGfilt2deriv[idxDeriv-3,:]-2*ECGfilt2deriv[idxDeriv-4,:])/8;
    ECGDeriv = abs(np.prod(ECGDeriv,1)).T;
    
    # Artefacts removal
    # set a high pass filter to identify artifacts.
    HighPassOrder=250;
    HighPassCutOff=250;
    if HighPassCutOff>=fs/2:
        no_filter=True;
    else:
        b_HighPassFilter = signal.firwin(HighPassOrder, 2*HighPassCutOff/fs,'high');
        ArtifactsThreshold  = 0.04;
        ECGDeriv_Artifacts=ECGDeriv;  # Artifacts Threshold
        artifacts = any(signal.filtfilt(b_HighPassFilter,1,signal)**2>ArtifactsThreshold,2);
        if sum(artifacts)!=0:
            s=inputdlg('existen artefactos');
            ECGDeriv[artifacts] = 0; # Setting artefacts to 0 value
    
        no_filter=False;
  
    ## Peak Detection
    detect = ECGDeriv > np.mean(ECGDeriv);  # Compute QRS detection
    
                       # Compute onset and offset
    
    #OJKI
    # I think that this can be done more simpler
    detect_int = detect.astype(int) #conversion to integers
    
    #take the derivative
    detect_int_diff = np.diff(detect_int)
    
    #Now if detect_int_diff == 1 => going form FALSE to TRUE
    onset = np.where(detect_int_diff == 1)
    onset = onset[0]
    
    #If detect_int_diff == -1 => going from TRUE to FALSE
    offset = np.where(detect_int_diff == -1)
    offset = offset[0]
    offset = offset + 1 #to be the same index
#    k=0;
#    onset=np.zeros((1,100));
#    for i in np.arange(len(detect)-1):
#        if(detect[i]==False and detect[i+1]==True):
#            onset[0,k]=i;
#            k=k+1;
#            
#    aux=onset.nonzero();
#    onset=onset[0,np.arange(1,len(aux[0]),1)];
#    k=0;
#    offset=np.zeros((1,100));
#    for i in np.arange(len(detect)-1):
#        if(detect[i]==True and detect[i+1]==False):
#            offset[0,k]=i+1;
#            k=k+1;
#    aux=offset.nonzero();
#    offset=offset[0,np.arange(1,len(aux[0]),1)];
#    
#    if detect[len(detect)-1]: 
#        onset=np.delete(onset,[len(onset)-1]);
#        # Eliminate last onset if detect is 1 at the end
#    
#    if detect[0]:   
#        offset=np.delete(offset,[0]); 
#     # Eliminate first offset if detect is 1 at the beginning
    
    flag=1;
    while flag:      # Eliminate onset and offset whose ~detect window is smaller than min_rr ms (merge windows detection)
    #OJKI
        bad = np.where(onset[1:]-offset[:-1] < np.floor(min_rr*fs/1000))[0]       
    #OJKI    
        #bad =np.where(onset[np.arange(2,len(onset.T),1)]-offset[np.arange(1,len(offset.T)-1,1)] < np.floor(min_rr*fs/1000) );
        #bad=np.array(bad);
        if bad.size:    # elimination required
            onset= np.delete(onset,[bad+1]) ;
            offset= np.delete(offset,[bad]) ;
            flag=1;
        else:    # no more elimination
            flag=0;
    
    
    ## Enlarging Searching window
    security_Samples = np.floor(75*fs/1000); # we enlarge searching window a additional 75 ms before and after
    idx_onset = onset>security_Samples;
    onset[idx_onset] = onset[idx_onset] - security_Samples;
    onset[np.logical_not(idx_onset)] = 1;
    idx_offset = offset < n_samples - security_Samples;
    offset[idx_offset] = offset[idx_offset] + security_Samples;
    offset[np.logical_not(idx_offset)] = n_samples;
    
    ## Detection of main peak
    #Ojki
    #Compact format
    (n_samples4Detection,Nb_leads4Detection) = ECGfilt2deriv.shape
    #Ojki
   # n_samples4Detection=len(ECGfilt2deriv[:,1]);
   # Nb_leads4Detection=len(ECGfilt2deriv[1,:]);
    ind_peak = np.zeros((len(onset),Nb_leads4Detection));  # Allocation ind_peak
    direction = np.zeros((len(onset),Nb_leads4Detection));  # Allocation direction
   # direction=direction.astype(int64); Why????
    detect = np.zeros((n_samples4Detection,1));  # Reset detect to false to make reconstruction
    for j in np.arange(len((onset))):     # Search direction of each main peak of each lead of each QRS complex
        #Ojki
        #easier
        direction[j] = 1-2*(np.max(ECGflat[onset[j]:offset[j]+1,leads],axis = 0) > abs(np.min(ECGflat[onset[j]:offset[j]+1,leads],axis = 0))); # Direction of the Main Peaks
        #
        #direction[j] = 1-2*int(np.amax(ECGflat[np.arange(int(onset[j]),int(offset[j]),1),np.array(leads).T]) > abs(np.amin(ECGflat[np.arange(int(onset[j]),int(offset[j]),1),np.array(leads).T]))); # Direction of the Main Peaks

    #Óscar Change
#   direction = np.sign(sum(direction);    
    direction = np.sign(np.sum(direction,axis = 0));
    direction[direction==0] = 1;  # Take the direction of the majority for each lead
    #direction will be ngative if max is positive.
    
    for j  in np.arange(len((onset))):
        #[M,I] = min(bsxfun(@times,ECGflat(onset(j):offset(j),leads),direction(1,:))); # Main Peaks are the minimum (Multiply ECGflat by this direction to make MainPeaks minima)
        #Ojki
        #numpy notation for matrix product
        MM=np.min(np.dot(ECGflat[onset[j]:offset[j]+1,leads],np.diag(direction)),axis = 0)# function is equivalent to last one        
        II = np.argmin(np.dot(ECGflat[onset[j]:offset[j]+1,leads],np.diag(direction)),axis = 0)
        #Ojki
#        MM=aux2[1,:]
#        II=aux2[:,1];    
        ind_peak[j,:] = onset[j]+II;  # Adjust ind_peaks
        detect[onset[j]:offset[j]] = True;   # Reconstruct detect
    
    ## Heart Rate
    dind = np.diff(ind_peak,axis = 0);
    dind= np.concatenate((dind[0:1,], dind),axis=0);
    HeartRate = np.median(dind); # Compute normal rate (in samples), samples between peaks of signal 
    HeartRate_matrix=dind;
    all_peaks_indexes=ind_peak;
    RR_time = dind/fs; 
    RR_samples=dind;
    ## Saving variables to ind
    #ind.signal_in=signal_in;
    #ind.signal=signal;
    #ind.ECGfilt=ECGfilt;
    #ind.ECGflat=ECGflat;
    #ind.ECGDeriv_Artifacts=ECGDeriv_Artifacts;
    #ind.ECGDeriv=ECGDeriv;
    #ind.onset=onset;
    #ind.offset=offset;
    #ind.ind_peak=ind_peak;
    #ind.HeartRate =HeartRate;
    #ind.fs=fs;
    #ind.RR_time=RR_time;
    #ind.RR_samples=RR_samples;
    #ind.ECGflat=ECGflat;
    #ind.n_samples=n_samples;
    #ind.valid_leads=valid_leads;
    #ind.Nb_leads=Nb_leads;
    
    ind = {'HeartRate':HeartRate,'HeartRate_matrix':HeartRate_matrix,
           'all_peaks_indexes':all_peaks_indexes,'RR_time':RR_time,
           'RR_samples':RR_samples}
    return ind;


##
##
#   LOCAL FUNCTIONS 
#
##
##
def detrendSpline(ECGfilt,L_w,Fe,siz):
#    function [ECGflat s_pp] = detrendSpline(ECGfilt,L_w,Fe,siz)
    # Compute the trend of the signal and remove it to the signal in order to flatten this last
    t =arange(0,siz).T;
    n_t =arange (1,siz-L_w,L_w/2);
    s_m=zeros((len(n_t),1));
    t_m = t(n_t+L_w/2-1);
    for k in arange(len(n_t)):
        s_m[k]=mean(ECGfilt[arange(n_t[k],n_t[k]+L_w-1)]);
   
    pp=csaps(t_m,s_m);
    s_pp=ppval(pp,t);
    ECGflat = ECGfilt - s_pp;
    return ECGflat,s_pp

######################

#Calling the function

#load the signal
import scipy.io

mat = scipy.io.loadmat('ECG_Signal.mat')

#extract the signal

sig = mat['signal']

#Actual calling to the function

ind=RR(sig)

print(ind.keys())

plt.plot(ind['RR_time'])
plt.xlabel('# beat')
plt.ylabel('RR interval (ms)')
plt.show()
