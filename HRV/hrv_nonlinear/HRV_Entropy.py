# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 10:42:52 2013

@author: Óscar Barquero Pérez
         Rebeca Goya Esteban
"""

import scipy as sc
import numpy as np

class HRV_entropy(object):
    """
    Classes to provide all the available non-linear methods for HRV
    """

    def __init__(self,rr = []):
        self.rr = rr
    
    def SampEn(self,rr,m = 2,r = 0.2,kernel = 'Heaviside'):
        """Compute SampEn
           Translation based on Rebeca's Master Thesis
        """
        
        
        N = len(rr)        
        B_m_i = np.zeros(N-m);
        A_m_i = np.zeros(N-m);
        res = None
#    
#        se crea una matriz que contendra todos los vectores a ser comparados
#        entre si.
        for n in range(2):
            M = np.zeros((N-m,m+n));
            f,c = np.shape(M);
            for i in range(f):
                M[i,:]=rr[i:i+m+n]
    
#        #calculo de la medida de correlacion.
#    
            for i in range(f):
#            #se construye una matriz cuyas filas contien el vector a ser comparado con el resto de los 
#           #vectores, replica la matriz dada con dimensiones fx1.    
                Mi = np.tile(M[i,:], (f, 1))
#        
#        #para cada fila de la matrix el maximo entre las columnas de la matriz de
#        #diferencias
                dist = np.max(abs(Mi-M),axis = 1)
#        #para eliminar las autocomparaciones
#        
                dist = np.delete(dist,[i])
                #dist[i,:]=[]
#        
                if n == 0:
                    B_m_i[i]=sum(dist<=r)/float((N-m-1))
                else:
                    A_m_i[i]=sum(dist<=r)/float((N-m-1))
#       
        B_m = np.mean(B_m_i)
        A_m = np.mean(A_m_i)
        res= np.log(B_m) - np.log(A_m)

        return res
    
    def TimeIrreversibility(self,rr,tau):
            '''
            ARGUMENTOS DE ENTRADA: 
              .-rr ---> señal de la que se pretende estimar la MTI.
              .-tau ---> número de escalas para el análisis.

           ...DE SALIDA:
           .-indexAsimetry ---> Indice de asimetria Time Irr
               BIBLIOGRAFIA:
                Costa, M., A. L. Goldberger, et al. (2008). 
                Multiscale Analysis of Heart Rate Dynamics: Entropy and Time
                Irreversibility Measures."
                
                Author: Rebeca Goya Esteban
            '''

            N = len(rr) #length RR-interval time series
            x_i = rr[:]
            A_tau = [] #new implementation with list
            
            for j in np.arange(1,tau+1):
                y_i = np.zeros((N-j))
                
                for i in np.arange(1,N-j+1):
                    y_i[i-1]=(x_i[i+j-1]-x_i[i-1])                    
                    
                
                A_tau.append = (np.sum(self.heaviside(-y_i))-np.sum(self.heaviside(y_i)))/ (N-j)
                
            return np.sum(A_tau)
             
    def heaviside(self,x):
            '''
            Implementation heaviside function
            '''
            #convert to numpy
            x_np = np.array(x[:])
            y = np.zeros(np.shape(x_np))
            y[x_np >= 0] = 1
            y[np.isnan(x_np)] = np.nan
            return y

###### main

#hrv_non = HRV_entropy()

#a = hrv_non.SampEn(z, r = 0.2*std(z))

#import pyegg as pyegg

#b = pyegg.samp_entropy(z,2,0.2*std(z))
#print ('Our implementation = ' +str(a))
#print ('')
#print ('Implementation = ' +str(b))

###############################################
############## SAMPLE ENTROPY #################
###############################################
"""
import matplotlib.pyplot as plt

from mix_processes import *
r = np.logspace(-2,0,4) #vector of r (np.logspace(-2,0,100))
s = np.logspace(-2,0,16)
hrv = HRV_entropy() #create class of hrv entropy

SampEn_01 = []
SampEn_09 = []

for r_aux in r:
    SampEn_aux_01 = []
    SampEn_aux_09 = []
    for i in range(4): # Cambiar a range(100)
        Mix1 = mix(1000,0.1)
        Mix9 = mix(1000,0.9)
        print(str(i))
        S1 = hrv.SampEn(Mix1[:,0],r = r_aux)
        S2 = hrv.SampEn(Mix9[:,0],r = r_aux)
        SampEn_aux_01.append(S1)
        SampEn_aux_09.append(S2)

    SampEn_01.append(SampEn_aux_01)
    SampEn_09.append(SampEn_aux_09)


plt.figure()
plt.errorbar(s, SampEn_01, color='r');
plt.errorbar(s, SampEn_09, color='k');
"""
###############################################
########### TIME IRREVERSIBILITY ##############
###############################################

#p = np.linspace(0.05,0.95,50)

#TimeIrr_01 = []
#TimeIrr_09 = []

#for p_aux in p:
#    TimeIrr_aux_01 = []
#    TimeIrr_aux_09 = []
#    for i in range(100):
#        Mix_TI_01 = mix(1000,0.1) # Cambiar Mix = mix(1000, p_aux)
#        Mix_TI_09 = mix(1000,0.9) 
#        TI_01 = hrv.TimeIrreversibility(Mix_TI_01[:,0])
#        TI_09 = hrv.TimeIrreversibility(Mix_TI_09[:,0])
#        TimeIrr_aux_01.append(TI_01)
#        TimeIrr_aux_09.append(TI_09)
    
#    TimeIrr_01.append(TimeIrr_aux_01)
#    TimeIrr_09.append(TimeIrr_aux_09)
    
#figure(2)

#############################################