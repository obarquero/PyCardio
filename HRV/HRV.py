# -*- coding: utf-8 -*-
"""
Calculates from the RR intervals the statistical time domain variables and the 
geometrical variables to characterize the heart rate variability (HRV).
The RR intervals used are all of them previous to valid tachograms according to the 
conditions evaluated in the characterization of the heart rate turbulence (HRT).
"""

from __future__ import unicode_literals
import numpy as np
import matplotlib.pylab as plt
from scipy import interpolate
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy import signal

class HRV(object):
    
    def __init__(self):
        self.HRV_statistical = {} 
        self.HRV_geometrical = {}
        
        
    def load_HRV_variables(self, rr):
        """
        Returns a dictionary with all the considered values of the heart rate variability 
        calculated from the list of the rr intervals passed as a parameter
        """
        hrv_pat = {}
        avnn = self.avnn(rr)
        nn50 = self.nn50(rr)
        pnn50 = self.pnn50(rr)            
        rmssd = self.rmssd(rr)            
        sdnn = self.sdnn(rr)           
        sdsd = self.sdsd(rr)
        hrvTriangIndex = self.hrvTriangIndex(rr)
        logIndex = self.logIndex(rr)          
        tinn = self.tinn(rr)
#            
        hrv_pat['AVNN'] = avnn
        hrv_pat['NN50'] = nn50
        hrv_pat['PNN50'] = pnn50
        hrv_pat['RMSSD'] = rmssd
        hrv_pat['SDNN'] = sdnn
        hrv_pat['SDSD'] = sdsd
        hrv_pat['HRVTriangIndex'] = hrvTriangIndex
        hrv_pat['logIndex'] = logIndex
        hrv_pat['TINN'] = tinn
        
        return  hrv_pat
        
       
############################# HRV Indices #############################       
        
    ## STATISTICAL TIME DOMAIN HRV VARIABLES ##    

    
    def avnn(self, nn):
        """
        Function that computes the AVNN, that is, the average value of all NN 
        intervals computed over the complete time series that is passed as 
        input parameter.
        """
        #Mean of the NN interval series
        mu = np.mean(nn)
        return mu
        
       
    
    def nn50(self, nn):
        """
        This function computes the NN50 index, that is, the number of adjacent 
        pairs of NN intervals that are more than 50 msg of the entire time 
        series that is passed as the input parameter.
        """
        #Differences between adjacent NN intervals.
        d = np.diff(nn)
        #Number of adjacent intervals whose distance is greater than 50ms
        res= sum(abs(d) > 50)
        return res
        
       
        
    
    def pnn50(self, nn):
        """
        Function that computes the pNN50 index, that is, the percentage 
        of adjacent pairs of NN intervals that are more than 50 msg of 
        the entire time series that is passed as the input parameter.
        """
        #Differences between adjacent NN intervals.
        d = np.diff(nn)
        #Number of adjacent intervals whose distance is greater than 50ms
        num = float(sum(abs(d) > 50))
        #Percentage
        res = num/len(d)*100
        return res
        
        
    
    def pnnX(self, nn, x):
        """
        This function computes the pNNX index, that is, the percentage of 
        adjacent pairs of NN intervals that are more than X msg of the entire 
        time series that is passed as the input parameter.
        """
        #Differences between adjacent NN intervals.
        d = np.diff(nn)
        #Number of adjacent intervals whose distance is greater than x ms
        num = float(sum(abs(d) > x))
        #Percentage
        res = num/len(d)*100
        return res


    
    def rmssd(self, nn):
        """
        Function that computes the RMSSD, that is, the square root of the squared 
        differences between successive NN intervals of the entire time series 
        that is passed as input parameter.
        """
        #Differences between adjacent NN intervals.
        d = np.diff(nn)
        #Square of the differences between adjacent NN intervals
        d2 = d**2
        #Square root mean squared differences between adjacent NN intervals.
        res = np.sqrt(np.mean(d2))
        return res
        
        
    def sdann(self, nn, t = None, window_min = 5):
        """
        Function that computes the SDANN, that is, the standard deviation 
        of the means of segments of 5 min of the whole time series that is 
        passed as input parameter
        """                 
        if t == None:
            t = np.cumsum(nn)/1000.     #in seconds
        
        tau = window_min * 60; 
        #Rounding down the last element of t divided by tau
        numSeg = float(t[-1]/tau)
        numSeg = np.floor(numSeg);
        numSeg = int(numSeg)

        mus = []
        V_inicio = np.zeros((1, numSeg))
        V_fin = np.zeros((1, numSeg))

        #Calculation of the mean of each segment
        for m in range(numSeg):
            #Initial and final indices for each segment of 5 min   
            inicio = np.where(t >= (m)*tau)[0]
            fin = np.where(t <= (m+1)*tau)[0]
            V_inicio[0][m]= inicio[0]
            V_fin[0][m] = fin[-1]
            seg = nn[inicio[0]:(fin[-1])+1]
            mus.append(np.mean(seg))              
        
        #Sdann computing
        stadev = np.std(mus, ddof=1)
        return stadev
        
        
    
    def sdnn(self, nn):
        """
        Function that computes the SDNN, that is, the standard deviation of 
        all NN intervals computed over the complete time series that is passed 
        as input parameter.
        """
        #Standard deviation of the series of NN intervals.
        stdev = np.std(nn, ddof=1)
        return stdev

    
    
    def sdnnidx(self, nn, t = None, window_min = 5):
        """
        Funcion que computa la sdnnidx, es decir, el valor medio de las std del
        segmento de 5 minutos computado sobre la serie temporal completa que se 
        pasa como parametro de entrada
        """
        if t == None:
            t = np.cumsum(nn)/1000.             
        
        #Obtaining the temporary instants of heartbeats
        tau = window_min * 60
        numSeg = float(t[-1]/tau)
        numSeg = np.floor(numSeg)
        numSeg = int(numSeg)

        stdSeg5min = []
        V_inicio = np.zeros((1,numSeg))
        V_fin = np.zeros((1,numSeg))

        #Calculation of the mean of each segment
        for m in range(numSeg):
            #Initial and final indices for each segment of 5 min    
            inicio = np.where(t >= (m)*tau)[0]
            fin = np.where(t <= (m+1)*tau)[0]
            V_inicio[0][m]= inicio[0]
            V_fin[0][m] = fin[-1]
            seg = nn[inicio[0]:(fin[-1])+1]
            #Standard deviation of the segment   
            stdSeg5min.append(np.std(seg, ddof=1))

        #sdnnidx computing
        res = np.mean(stdSeg5min)
        return res
        
    
    def sdsd(self, nn):
        """
        Function that computes sdsd, that is, standard deviation of the 
        differences between adjacent NN intervals.
        """
        #First we obtain the differences of the intervals NN
        d = np.diff(nn)        
        #The standard deviation is then calculated
        res = np.std(d, ddof=1)
        return res
        
        
    
    ### GEOMETRICAL TIME DOMAIN HRV VARIABLES ##
        
    def elipse(self, xc, yc, theta, sd1, sd2, pintar = None):
        """
        Function that constructs an ellipse and paints it, with the parameters 
        that are indicated in its entrance
        
        Input arguments:        
            Xc:     coordinate x of the center of the ellipse
            Yc:     coordinate y from the center of the ellipse
            Theta:  angle of the coordinate axes of the ellipse, with center c (xc, yc), 
                    with respect to the horizontal
            Sd1:    length of the x axis of the ellipse
            Sd2:    length of the y axis of the ellipse        
        Output arguments:
            X:      points of the ellipse along the x-axis
            Y:      points of the ellipse along the y-axis     
        """
       #By default the ellipse is not painted
        if pintar == None:
            pintar = 0

        #Number of points
        n = 100

        #Angle variation
        l = np.arange(0, n+1, dtype=float)
        ang = l*2*np.pi/n

        #Construction of the ellipse
        paso1 = np.matrix([[xc],[yc]]) * np.ones((1,ang.size))
        paso2 = np.matrix([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
        paso3 = np.matrix([np.cos(ang)*sd2,np.sin(ang)*sd1])
        xy = paso1 + paso2 * paso3

        xyList = xy.tolist()

        if pintar:
            plt.plot(xyList[0], xyList[1], color='r', linewidth=2.0)
            
        return xy
        
        
    def hrvTriangIndex(self, rr, flag=None):
        """
        Function that computes the triangular index, that is, the total number 
        of intervals rr between the height of the histogram.
        """        
        if flag == None:
           flag = 0

        #Number of bins with fs = 128, recommendation of the ref.
        fs = 128.
        ts = 1/fs*1000. #ms

        #Bins computing
        x = np.arange(min(rr), max(rr), ts)

        #Number of bins for the histogram
        nhist = x.size

        #Histogram
        [N, X] = plt.histogram(rr,nhist)

        #Only the non-empty bins are taken into account
        ind = np.where(N != 0)
        N = N[ind]
        X = X[ind]

        #Histogram maximum
        yo = max(N)

        res = sum(N)*1./yo

        if flag:
        #Graphic representation
            plt.hist(rr,nhist)
            plt.title('HRVTriangIndex')
            plt.xlabel('Duracion intervalos RR [ms]')
            plt.ylabel('Numero de intervalos RR')
        
        return res
        
        
        
    def logIndex(self, rr, pintar=None):
        """        
        Function that computes the triangular interpolation of the intervals rr
        Output arguments:
            res: exponent of the exponential that best adjusts -> k*exp(-res*t)
        """
       
        if pintar == None:
            pintar = 0

         #We create the difference series
        diffSer = np.diff(rr)

        #We create the histogram:

        #number of bins with fs = 128, recommendation of ref.
        fs = 128. 
        ts = 1/fs*1000. #ms

        #bins computing
        x = np.arange(rr.min(0), rr.max(0), ts)

        #number of bins for the histogram.
        nhist = x.size

        [Nabs, X] = np.histogram(abs(diffSer), nhist)

        #non-empty bins
        ind = np.where(Nabs != 0)[0]
        Nabs_full = Nabs[ind]
        X_full = X[ind]

        #Adjusting the exponential k*exp(-phi*t):

        #Constants
        k = max(Nabs_full)

        #Number of iterations n=4000
        Niter = 10000
        phi = np.linspace(-1, 1, Niter)

        #Error
        error = np.zeros((Niter,1))

        for m in range(Niter):          
            error[m] = sum((Nabs_full - k*np.exp(phi[m]*X_full))**2)
    
        #Minimum error
        indmin = np.argmin(error)

        #Phi for best setting of the exponential
        res = phi[indmin]

        #Graphic representation
        if pintar: 
            plt.close('all')
            plt.bar(X_full,Nabs_full)
            plt.plot(X_full, k*np.exp(res*X_full), 'r')
       
        return res
        
        
        
    def mediasPoincare(self, rr, flag = None):
        """
        Function that computes the geometric HRV indices based on the Poincare Plot.
        
        Output Parameters:        
            sd1:    dispersion of map points perpendicular to the axis of the identity line
        
            sd2:    dispersion of map points along the axis of the identity line
        
            cup:    contributions for the decelerations of the heart rhythm by the Poincare points,
                    based on the asymmetries of the map
        
            cdown:  contributions for the accelerations of the cardiac rhythm by the points of the Poincare,
                    based on the asymmetries of the map
        """
       
        if flag == None:
           flag = 0

        #In the input parameter rr are the rr intervals without ectopic,
        #the vectors x and y (Vid Ref) are defined as:
        x = rr[:]
        x = x[:-1] #we removed the last element
        y = rr
        y = y[1:] #we removed the first element
        L = x.size

        #The standard indices sd1 and sd2:
        sd1 = np.sqrt((1./L) * sum(((x - y) - np.mean(x - y))**2)/2.)
        sd2 = np.sqrt((1./L) * sum(((x + y) - np.mean(x + y))**2)/2.)

        #Index sd1I (moment of second order around the identity line).
        sd1I = np.sqrt((1./L) * (sum((x - y)**2)/2.))

        #Quantification of the contributions of the points above and below 
        #the identity line.
        xy = (x - y)/np.sqrt(2)
        indices_up = np.where(xy > 0)[0]
        indices_down = np.where(xy < 0)[0]
        sd1up = np.sqrt(sum(xy[indices_up]**2)/L)
        sd1down = np.sqrt(sum(xy[indices_down]**2)/L)

        #Finally, the relative contributions
        cup = sd1up**2/sd1I**2
        cdown = sd1down**2/sd1I**2

        #Graphic representations
        if flag:
        #poincarePlot
            plt.plot(x,y,'.')

        #identity line and the perpendicular
            xc = np.mean(x)
            yc = np.mean(y)

            l1 = (np.tan(np.pi/4)*(x-xc)) + yc
            l2 = (np.tan(3*np.pi/4)*(x-xc)) + yc

            xl = np.sort(x)
            l1 = np.sort(l1)
            
            xData = [xl[0],xl[-1]]
            yData1 = [l1[0],l1[-1]]
            yData2 = [max(l2),min(l2)]
            
            plt.hold(True)            
            plt.plot(xData, yData1, color='r', linestyle=':', linewidth=2.0)
            plt.hold(True) 
            plt.plot(xData, yData2, color='r', linestyle=':', linewidth=2.0)
            #Paint more thick the area of ​​the sd

            #We paint the ellipse
            plt.hold(True)
            self.elipse(xc,yc,np.pi/4,sd1,sd2,1)
            
        return sd1,sd2,cup,cdown
        
        
        
    def tinn(self, rr, flag = None):
        """                
        Function that computes the triangular interpolation of NN
        
        Output arguments:
            res:    width of the triangular interpolation
        """
        #Number of bins with fs = 128, recommendation of ref.
        if flag == None:
            flag = 0

        fs = 128
        ts = 1./fs*1000 #ms

        #Bins computing
        x = np.arange(min(rr), max(rr), ts)

        #Number of bins for the histogram
        nhist = x.size
        #Histogram
        [N,X] = np.histogram(rr, bins = nhist)
        #Only the non-empty bins are taken into account
        ind = np.where(N != 0)
        N = N[ind]
        X = X[ind]

        #Center position of the histogram
        yo = max(N)
        k = np.argmax(N)
        xo = X[k]

        #Approximation of each half of the histogram

        #Number of maximum iterations for interpolation
        Nstep = 4000

        #First half
        N1 = N[0:k]
        X1 = X[0:k]
        
        #Second half
        N2 = N[k+1:]
        X2 = X[k+1:]

        #Compute of errors
        errorsm = np.zeros((Nstep,1))
        errorsn = np.zeros((Nstep,1))

        if k == 0:
            res = np.nan
        else:
            mrange = np.linspace(min(X1)/2, max(X1), Nstep)
            nrange = np.linspace(min(X2), 2*max(X2), Nstep)
    
            for h in range(Nstep):
               
               #First half          
               aux1 = np.where(X1 < mrange[h])           
               aux2 = np.where(X1 >= mrange[h])
               errorsm[h]= sum(N1[aux1]**2) + sum((N1[aux2] - (yo*X1[aux2]-yo*xo)/(xo-mrange[h]) - yo)**2)
               
               #Second half
               aux1 = np.where(X2 <= nrange[h])
               aux2 = np.where(X2 > nrange[h])
               errorsn[h] = sum(N2[aux2]**2) + sum((N2[aux1] - (yo*X2[aux1]-yo*xo)/(xo-nrange[h]) - yo)**2)
               
            errorsm = errorsm/N1.size
            errorsn = errorsn/N2.size
    
            mm = min(errorsm)
            km = np.argmin(errorsm)
            m = mrange[km]
            nn = min(errorsn)
            kn = np.argmin(errorsn)
            n = nrange[kn]
    
            #Area percentage explained by TINN
            k = min(abs(X1-m))
            km = np.argmin(abs(X1-m))
            k = min(abs(X2-n))
            kn = np.argmin(abs(X2-n))
            total = sum(N)
            explained = sum(N1[km:]) + yo + sum(N2[0:kn+1])        
            tinnpercent = (total-explained)*1./total*100.
    
    
            res=(n-m)


        if flag:
        #Graphic representation
            #close all
            Y1 = (yo*X1 -yo*xo)/(xo-m) + yo
            aux1 = np.where(X1 < m)[0]
            Y1[aux1] = np.zeros((aux1.size))

            Y2 = (yo*X2-yo*xo)/(xo-n) + yo
            aux1 = np.where(X2 > n)[0]
            Y2[aux1] = np.zeros((aux1.size))

            XX = np.hstack((X1, xo, X2))
            YY = np.hstack((Y1, yo, Y2))

            plt.figure(1)
            plt.hist(rr, nhist)
            plt.plot(XX, YY, color='r', linewidth=2.5)
            plt.xlabel('NN (ms)')
            plt.title('tinn') 
            
        return res
        
############################## HRV spectral #################################
    def interp_to_psd(self,rr, t = None, fs = 4., method = 'cubic'):
    
        ts = 1/fs #sampling frequency
        
        t_new = np.arange(t[0],t[-1],ts) #nuevo vector para la interpolacion
        #Interpolacion
        f = interp1d(t, rr, kind = method) #crea el objeto para interpolar
        
        #Ahora realizamos la interpolación realmente
        rr_interp = f(t_new)
        
        return rr_interp,t_new
    
    
    #Función que calcula el periodograma de Welch para el intervalo de tiempo rr
    #dado, que se supone que se interpola en fs
    def Welch_Periodogram(self,rr, fs = 4., window = 'hamming', nperseg = 256, noverlap = 256/2, nfft  = 1024):
        
        rr = rr - np.mean(rr)
        
        #Eliminar la tendencia lineal a lo largo del eje de datos
        rr = signal.detrend(rr)
        
        f, p = signal.welch(rr, fs, window = window, nperseg = nperseg, noverlap = noverlap, nfft = nfft)
        #f, p = signal.welch(rr, fs, window = 'hanning', nperseg = 256, noverlap = 128, nfft  = 1024)
        
        return f, p #p: densidad espectral de potencia
                    #f: vector frecuencia
                 
    def spectral_indices(self,Pxx, f, duration = 5):
        
        if duration == 5:
            indVlf = f <= 0.04
            indUlf = []
        elif duration >= 5:
            indUlf = f <= 0.003
            indVlf = np.bitwise_and(f > 0.003, f <= 0.04);
            
        ind = f <= 0.4
        indLf = np.bitwise_and(f > 0.04, f <= 0.15)
        indHf = np.bitwise_and(f > 0.15, f <= 0.4)
        
        df = f[1]
        
        #Cálculo de la potencia total
        Ptot = df * sum(Pxx[ind])
        
        #Cálculo potencia en la banda ULF
        if len(indUlf) == 0:
            Pulf = df * sum(Pxx[indUlf]) #En ms^2
        else:
            Pulf = [];
            
        #Cálculo potencia en la banda VLF
        Pvlf = df * sum(Pxx[indVlf]); 
        
        #Cálculo potencia en la banda LF
        Plf = df * sum(Pxx[indLf]);
        
        #Cálculo potencia en la banda HF
        Phf = df * sum(Pxx[indHf]);
        
        #Cáculo del ratio LF/HF
        lfhf_ratio = Plf/Phf;
         
        return Ptot, Pulf, Pvlf, Plf, Phf, lfhf_ratio
    
    def main_interp(self,rr, t = None, duration = 5):
        if t == None:
            t = np.cumsum(rr)/1000.
    
        #Interpolación
        
        rr_interpolated_4_hz, t_new = self.interp_to_psd(rr,t)
        
        return rr_interpolated_4_hz, t_new
    
    def main_welch(self,rr_interpolated_4_hz, t = None, duration = 5):
        
        f, Pxx = self.Welch_Periodogram(rr_interpolated_4_hz,  4., 'hamming', 256, 128, 1024)
        
        return f, Pxx
    
    def main_spectral_indices(self,Pxx, f = None, duration = 5):
        
        Ptot, Pulf, Pvlf, Plf, Phf, lfhf_ratio = self.spectral_indices(Pxx, f, duration = 5.)
        
        return Ptot, Pulf, Pvlf, Plf, Phf, lfhf_ratio
        
############################# HRV_Preprocessing #############################       
               
         
         
    def threshold_filter(self, rr):
        """
        Function that identifies a rr-interval as non-sinusal following the rule:
            if RR(n) > thr_up or RR(n) < thr_low
                where:
                thr_up = 2000
                thr_low = 300
        Verify this thresholds and find a reference
        Output arguments:
            ind_not_N:  has 1 in the position where there is a non-sinusal beat as
                        classified by the threshold criterion
        """
        ind_not_N = [False]*len(rr)
        ind_not_N = np.array(ind_not_N) #convert to a numpy array
        
        pos_ind_not_N = np.concatenate((np.where(rr > 2000)[0],np.where(rr<300)[0]))
        
        if len(pos_ind_not_N) > 0:
            ind_not_N[pos_ind_not_N] = True
            
        return ind_not_N
        
        
        
        
    def beat_label_filter(self, beat_labels, numBeatsAfterV = 4):
        """
        Function that identifies non-normal beats, and filters the rr signal to
        produce a vector identifying the positions where are non-normal beats.

        Input arguments:
            numBeatsAfterV <= 4
        Output arguments:
            ind_not_N:  has 1 in the position where there is a non-sinusal beat as
                        classified by the label information.
        """
        beat_labels = np.array(beat_labels)
        ind_not_N =[False] * len(beat_labels) #vector with False in every position
        ind_not_N = np.array(ind_not_N) #convert to a numpy array
  
        pos_ind_not_N = np.where(beat_labels != 'N')[0]
         
        if len(pos_ind_not_N) > 0:
            
            ind_not_N[pos_ind_not_N] = True
        
        #Identify as non normal 3 beats after a ventricular one
        ind_V = np.where(beat_labels == 'V')[0]

        if len(ind_V) > 0 :
            
            for g in range(1, numBeatsAfterV+1):
                #For each group of posterior beats to one ventricular,
                #we eliminate the beat that is 'g' positions behind a ventricular one
                if ind_V[-1] + g < len(ind_not_N):
                    
                    ind_not_N[ind_V+g] = True
                    
        return ind_not_N
        
        
        
        
    def perct_filter(self, rr, prct):
        """
        Function that identifies a rr-interval as non-sinusal following the rule

        if RR(n) > prct * RR(n-1) then RR(n) is non-sinusal

        Output arguments:
            ind_non_N:  has 1 in the position where there is a non-sinusal beat as
                        classified by the percentage criterion.


        TO_DO implement a method that takes into account that the previous beat should be labeled
        as normal
        """

        ind_not_N = [False]*len(rr)
        ind_not_N = np.array(ind_not_N) #convert to a numpy array
        #Construct a matrix with the percentage * RR(n-1)

        percMatrix = np.abs(np.diff(rr) / rr[:-1])
        
        pos_ind_not_N = np.where(percMatrix > prct)[0]
        
        if len(pos_ind_not_N) > 0:
            ind_not_N[pos_ind_not_N+1] = True  #consider the first rr-interval as normal.      
        
        return ind_not_N
        
        
        
        
    def artifact_ectopic_detection(self, rr, labels, prct, numBeatsAfterV = 4):
        """
        Function that calls detection methods to evaluate rr.         
        
        For a rr interval to be valid, it must pass all three detection methods.
        
        NOTE: RECOMMENDATION use this function with a detrended signal, get
        better results
        
        Output arguments:
            ind_not_N_beats:  has 1 in the position where there is a non-sinusal beat as
                              classified by the detection methods
        """

        ind_not_N_1 = np.array(self.beat_label_filter(labels, numBeatsAfterV))               
                      
        ind_not_N_2 = np.array(self.perct_filter(rr, prct))
            
        ind_not_N_3 = np.array(self.threshold_filter(rr))
            
        ind_not_N_beats = np.logical_or(ind_not_N_1, np.logical_or(ind_not_N_2, ind_not_N_3))
                    
        return ind_not_N_beats
        
        
        
        
    def is_valid(self, ind_not_N,perct_valid = 0.2):
        """
        Function that checks if there are more than 20% of invalid values ​​in the vector, 
        where True is an invalid value
        
        Returns True if it contains less than 15% of invalid values
        """
              
        num_not_valid = sum(ind_not_N == True)
        
        # if  percentage of num_not_valid is lower than perc_valid then the RR interval segment
        #is valid (return True)
        if num_not_valid*100/len(ind_not_N) <= perct_valid*100:
            return True
        else:
            return False
            
            
            
            
    def artifact_ectopic_correction(self, rr, ind_not_N, method='cubic'):
        
        """
        Function that corrects ectopic beat by interpolation. 
        
        The interpolator method is given by the string method (cubic','linear','nearest').
        """
        
                
        
        t_rr = np.cumsum(rr)

        rr_aux = rr[np.logical_not(ind_not_N)]

        t_rr_aux = t_rr[np.logical_not(ind_not_N)]        
        
        #TO_DO verify extrapolation with splines in scipy
        #Meanwhile: extrapolate using the mean value of the 5 first, 
        f = interpolate.interp1d(t_rr_aux,rr_aux,method,fill_value = (np.mean(rr_aux[:5]),np.mean(rr_aux[-5:])),bounds_error = False)
        
        rr_corrected = f(t_rr)
        
        return rr_corrected
        
        
        
        
        
        
        
        
        
        
        
        