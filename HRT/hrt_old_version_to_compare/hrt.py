# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 09:58:28 2013

HRT classical Analysis

@author: Óscar Barquero Pérez
         Rebeca Goya Esteban
         Teresa Quintanilla
"""

import scipy as sc
import numpy as np
import matplotlib.pyplot as plt

class HRT(object):
    """HRT turbulence class. Makes all the analysis on a given signal wich is 
    stored as variable in the class
    """
    
    
    def __init__(self,rr = [],labels = []):
        """Initialize all the variables of the class
        Input arguments:
            rr = Vector (np.array or list) with rr interval time series in ms
            labels = Vector (np.chararray or list) with labels for each rr interval
        np.size(rr) == np.size(labels)
        """
        self.rr = rr
        self.labels = labels
        self.VPC_tachs_ok = None
        self.VPC_pos_tachs_ok = None
        self.mean_VPC_tach = None
        self.TS = None
        self.TO = None
        

    def HRT_preprocessing(self,Filter_type = 'Watanabe'):
        """Function that performs a complete HRT preprocessing step that includes
            1) Find all positions of Ventricular beats
            2) Find all VPC-tachograms (5beats pre+VB+PC+20beats post)
            3) Filter all VPC-tachograms; i.e. find all the availables VPC-tachograms
            to be used on the HRT analysis
        Input arguments
            Filter_type = 'Watanabe', it allows to select filter from Bauer papers
        Output arguments
            VPC_tachogram = Dict with HRT struct to be used in following analysis
            TO_DO explain this struct
        """
        
        post_cp_beats = 20 #number of beats post_pc
        idx_vpc_tach = range(-6,post_cp_beats+1)
        #Realtive indexes for VPC_tacogram extraction
        # idx(VPC) = -1
        # idx(CP) = 0
        
        VPC_pos = self.labels == 'V'
        VPC_pos = np.argwhere(VPC_pos == True)
        # Guarantee last VPC position allows 20 beats after it.        
        if VPC_pos[-1] + post_cp_beats > len(self.rr):
            VPC_pos = np.delete(VPC_pos,-1)         
            
        VPC_tachs_aux = list() #all possible VPC_tachograms
        Labels_tachs = list()
        
        for m in range(len(VPC_pos)):
            idx_tachs = VPC_pos[m] + idx_vpc_tach+1
            #absolute idx on the rr-interval time seris
            VPC_tachs_aux.append(self.rr[idx_tachs])            
            Labels_tachs.append(self.labels[idx_tachs])
          
        
        #Filtering process of VPC_tachograms
        
        #Every VPC_tachogram is going to be "filtered" to determine if it is
        # a valid VPC_tachogram to be used in HRT analysis
        
        #Indices relatives
        idx_all = np.asarray(range(-6,post_cp_beats+1)) + 6
        idx_no_vpc = np.asarray(range(-6,-2+1) +range(0,post_cp_beats+1)) + 6
        idx_no_vpc_no_cp = np.asarray(range(-6,-2+1) +range(1,post_cp_beats+1)) + 6
        
        VPC_tachs_ok = list()
        VPC_pos_tachs_ok = list()
        
        i = 0
        for vpc,lab in zip(VPC_tachs_aux,Labels_tachs):
            #compute all the conditions that a VPC tachogram must fulfill
            
            #All beats, except for VPC, must be 'N'
            cond_1 = np.all(lab[idx_no_vpc] == 'N')
            
            #Sinus beats of the tachogram of RR intervals >= 􏰁300 ms
            cond_2 = np.all(vpc[idx_no_vpc_no_cp] >= 300)
            
            #Sinus beats of the tachogram of RR intervals <= 􏰁2000 ms
            cond_3 = np.all(vpc[idx_no_vpc_no_cp] <= 2000)
            
            #Jumps 􏰂 <= 200 ms from one interval to the next
            cond_4 = np.all(np.abs(np.diff(vpc[0:5]))<=200) and np.all(np.abs(np.diff(vpc[7:]))<=200)
        
            #Reference RR interval, mean of the five N RR-intervals precedding the VPC
            refRRinterval = np.mean(vpc[0:5])
            
            #All the RR-intervals should be lower than 1.2*refRRinterval
            cond_5 = np.all(vpc[idx_no_vpc_no_cp]<= 1.2*refRRinterval)
            
            #CVP should be at least 20% shorter than refRRinterval, i.e must be 
            #lower than 80% than refRRinterval
            cond_6 = vpc[5] <= 0.8*refRRinterval
            
            #CP should be greater than 1.2*refRRinterval
            cond_7 = vpc[6] >= 1.2*refRRinterval
            
            #Every one condition must be fulfilled
            if np.all([cond_1,cond_2,cond_3,cond_4,cond_5,cond_6,cond_7]):
                VPC_tachs_ok.append(vpc)
                VPC_pos_tachs_ok.append(VPC_pos[i])
            #increase counter
            i += 1
            
        self.VPC_tachs_ok = VPC_tachs_ok
        self.VPC_pos_tachs_ok = VPC_pos_tachs_ok
        self.mean_VPC_tach = np.mean(VPC_tachs_ok,axis=0)
            
    def TurbulenceSlope(self,VPC_tach,posPC = 6,posFin = 16,plotFlag = 0):
        """Computes Turbulence Slope on tachogram given by parameter
        """
        seg_len = 5

        slopes = list()        
        ordenada = list()
        
        for m in range(posPC+1,posFin):
            seg = VPC_tach[m:m+seg_len]
            p = np.polyfit(range(m,m+seg_len),seg,1)
            slopes.append(p[0])
            ordenada.append(p[1])
            
        TS = max(slopes)
        
        if plotFlag:
            self.plot_TS()
            
        self.TS = TS
        return TS
        
    def TurbulenceOnset(self,VPC_tach,posPC = 6,posFin = 16,plotFlag = 0):
    
        TO = ((VPC_tach[7]+VPC_tach[8])-(VPC_tach[3]+VPC_tach[4]))/(VPC_tach[3]+VPC_tach[4])*100;
        
        self.TO
        return TO
        
        
#main
#plt.ion()
#hrt_ex = HRT(rr = rr, labels = labels)
#hrt_ex.HRT_preprocessing()
#hrt_ex.TurbulenceSlope(hrt_ex.mean_VPC_tach)
#hrt_ex.TurbulenceOnset(hrt_ex.mean_VPC_tach)
#print 'TS' + str(hrt_ex.TS)
#print 'TO' + str(hrt_ex.TO)
#ts = hrt_ex.TurbulenceSlope(vpc)