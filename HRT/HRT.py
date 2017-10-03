# -*- coding: utf-8 -*-
"""
Extracts from all rr intervals:
    1.  The valid tachograms (associated to the correspondient labels and their ventricular beat position for each)
    2.  The mean tachogram
    3.  The value of the turbulence slop (TS) and the turbulence onset (TO) for each tachogram,
        including the mean tachogram (TS_average, TO_average)
    4.  The RR intervals previous to the tachograms       
"""

from __future__ import unicode_literals
import matplotlib.pylab as plt
import scipy as sp
import numpy as np


class HRT(object):
    
    def __init__(self, RRInts = [], Labels = []):
        self.RRInts = RRInts
        self.Labels = Labels
        self.tachograms = []            #only condition 1 tachograms [tach = 5N + V + 21N],
        self.v_pos_tachs = []           #and their ventricular beat position for each
        self.tachograms_ok = []         #all conditions tachograms,
        self.v_pos_tachs_ok = []        #and their ventricular beat position for each
        self.mean_tachogram_ok = []     #mean tachogram of all conditions tachograms
        self.TO = []                    #turbulence onset for each all conditions tachograms
        self.TS = []                    #turbulence slope for each all conditions tachograms
        self.TO_average = None          #turbulence onset for the mean tachogram
        self.TS_average = None          #turbulence slope for the mean tachogram
        self.RR_before_V = []           #values ​​of the RR intervals in a period of three minutes (by default) previous to the all conditions tachograms,
        self.pos_RR_bef_V = []          #and their associated positions

        
    def fill_HRT(self):
        """
        Returns a dictionary with all the values of the heart rate turbulence calculated
        """
        self.HRT_preprocessing()
        self.RRBeforeV() 
        self.compute()
        hrt_pat = {}
        hrt_pat['RRInts'] = self.RRInts
        hrt_pat['Labels'] = self.Labels
        hrt_pat['tachograms_ok'] = self.tachograms_ok
        hrt_pat['v_pos_tachs_ok'] = self.v_pos_tachs_ok
        hrt_pat['mean_tachogram_ok'] = self.mean_tachogram_ok
        hrt_pat['RR_before_V'] = self.RR_before_V
        hrt_pat['pos_RR_bef_V'] = self.pos_RR_bef_V
        hrt_pat['TO'] = self.TO
        hrt_pat['TS'] = self.TS
        hrt_pat['TS_average'] = self.TS_average
        hrt_pat['TO_average'] = self.TO_average
        return hrt_pat
        
        
    def is_tach_valid(self, tach):
        """Checks if tach is a valid tachogram following these two conditions:
            i)  Has a ventricular beat (V) at position 6 ([5])
            ii) The rest of beats are normal (N)
        """         
        aux = tach.copy()
        aux[5] = 'N' 
        V = np.where(aux == 'V')
        A = np.where(aux == 'A')
        AV = np.append(A, V, axis=None)
        if AV.size > 0:
            return False
        else:
            return True
            
            
    def paint_tach(self, tachogram):        
        """
        Paints a tachogram (tach = 5N + V + 21N)
        """         
        plt.close('all')    
        x = sp.linspace(0, 1, 27)
        plt.figure()
        plt.plot(x, tachogram) 
        
        
    def HRT_preprocessing(self, Filter_type = 'Watanabe'):        
        """
        Function that performs a complete HRT preprocessing steps that includes:
            1.  Find all positions of Ventricular beats
            2.  Find all VPC-tachograms (5beats pre+VB+PC+20beats post)
            3.  Filter all VPC-tachograms; i.e. find all the availables VPC-tachograms
                to be used on the HRT analysis
        This function fills the following variables described at the begining:
            i.   tachograms
            ii.  v_pos_tachs
            iii. tachograms_ok
            iv.  v_pos_tachs_ok
            v.   mean_tachogram_ok
        Input arguments:
            Filter_type = 'Watanabe', it allows to select filter from Bauer papers
        """
        
        possible_tachs = []             #all possible tachograms
        labels_tachs = []               #and their labels
        
        post_cp_beats = 20              #number of beats post_pc
        idx_vpc_tach = range(-6,post_cp_beats+1)
        
        v_pos = self.Labels == 'V'
        v_pos = np.argwhere(v_pos == True)
        # Guarantee last VPC position allows 20 beats after it.        
        if v_pos[-1] + post_cp_beats > len(self.RRInts):
            v_pos = np.delete(v_pos,-1)              

        
        for m in range(len(v_pos)):
            idx_tachs = v_pos[m] + idx_vpc_tach+1
            
            #absolute idx on the rr-interval time series
            if idx_tachs[-1] >= len(self.RRInts):
                continue
            possible_tachs.append(self.RRInts[idx_tachs])            
            labels_tachs.append(self.Labels[idx_tachs])          
        
        #Filtering process of tachograms:        
            #Every possible tachogram is going to be "filtered" 
            #to determine if it is a valid tachogram to be used 
            #in HRT analysis
        
        #Indices relatives for conditions
        idx_no_vpc = np.asarray(range(-6,-2+1) + range(0,post_cp_beats+1)) + 6
        idx_no_vpc_no_cp = np.asarray(range(-6,-2+1) + range(1,post_cp_beats+1)) + 6
        
        
        i = 0
        for vpc,lab in zip(possible_tachs,labels_tachs):
            #compute all the conditions that a VPC tachogram must fulfill
            
            #All beats, except for VPC, must be 'N'
            cond_1 = np.all(lab[idx_no_vpc] == 'N')
            
            #Sinus beats of the tachogram of RR intervals >= 300 ms
            cond_2 = np.all(vpc[idx_no_vpc_no_cp] >= 300)
            
            #Sinus beats of the tachogram of RR intervals <= 2000 ms
            cond_3 = np.all(vpc[idx_no_vpc_no_cp] <= 2000)
            
            #Jumps <= 200 ms from one interval to the next
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
            
            #Only condition 1 must be fulfilled
            if np.all([cond_1]):
                self.tachograms.append(vpc)
                self.v_pos_tachs.append(v_pos[i])
            
            #Every one condition must be fulfilled
            if np.all([cond_1,cond_2,cond_3,cond_4,cond_5,cond_6,cond_7]):
                self.tachograms_ok.append(vpc)
                self.v_pos_tachs_ok.append(v_pos[i])  
                
            i += 1
         
        if len(self.tachograms_ok) == 0:   
            #there is no tachograms that fulfill all conditions
            self.mean_tachogram_ok = np.nan
        else:
            self.mean_tachogram_ok = np.mean(self.tachograms_ok,axis=0)
            
            
            
    def RRBeforeV(self, mins_before_V = 3):
        """
        Function that saves the values of the rr intervals during mins_before_V
        minutes before the ventricular beat.        
        Input arguments:
            mins_before_V = period of minutes (3 minutes by default) before the tachogram, 
                            in which we want to save the values ​​of the RR intervals
        """        
        ms_before_V = mins_before_V * 60000 #in miliseconds

        rr = self.RRInts
        v_pos = self.v_pos_tachs_ok
        
        sumRR = np.cumsum(rr)
        
        for i in v_pos:
            #for each tachogram, we save all the values of the RR intervals in the given period,
            #along with their initial position and their final position for localized them
            limInf = np.where(sumRR > sumRR[i] - ms_before_V)
            limSup = np.where(sumRR < sumRR[i])
            pos_aux = np.append(limInf[0][0],limSup[0][-1],axis=None)
            self.pos_RR_bef_V.append(pos_aux)
            #NOTE: pos_RR_bef_V = (initial_position, final_position) is a tuple
            self.RR_before_V.append(rr[pos_aux[0]:pos_aux[1]])
            pos_aux = np.array([])
            
            
    def TurbulenceSlope(self, VPC_tach):        
        """
        Computes the Turbulence Slope on the tachogram given by parameter
        """        
        seg_len = 5
        posPC = 6
        posFin = 16

        slopes = []        
        ordenada = []
        
        for m in range(posPC+1, posFin):
            seg = VPC_tach[m:m+seg_len]
            p = np.polyfit(range(m, m+seg_len), seg, 1)
            slopes.append(p[0])
            ordenada.append(p[1])
            
        TS = max(slopes)
            
        return TS
        
        
    def TurbulenceOnset(self, VPC_tach, posPC = 6, posFin = 16):
        """
        Computes the Turbulence Onset on the tachogram given by parameter
        """    
        TO = ((VPC_tach[7]+VPC_tach[8])-(VPC_tach[3]+VPC_tach[4]))/(VPC_tach[3]+VPC_tach[4])*100;
        
        return TO
        
        
    def compute(self):
        """
        Computes the TS and TO on all conditions tachograms (tachograms_ok) and on the mean tachogram
        """        
        if np.sum(np.isnan(self.mean_tachogram_ok)) > 0:
            #if there is no tachograms that fulfill all conditions, the value of the mean tachogram is nan,
            #so the TS and TO values will be nan too
            self.TS = np.nan
            self.TO = np.nan
            self.TS_average = np.nan
            self.TO_average = np.nan
        else:            
            #compute TS and TO for each VPC tachogram
            for tac in self.tachograms_ok:
                self.TS.append(self.TurbulenceSlope(tac))
                self.TO.append(self.TurbulenceOnset(tac))             
            #compute the TS and TO from the average VPC tachogram           
            self.TS_average = self.TurbulenceSlope(self.mean_tachogram_ok)
            self.TO_average = self.TurbulenceOnset(self.mean_tachogram_ok)
            
    