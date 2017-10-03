# -*- coding: utf-8 -*-
"""
Extracts the data contained in the Holter's file of a patient.
"""


from __future__ import unicode_literals
import codecs
import numpy as np

class HolterData(object):
    
    def __init__(self): 
        self.name = ''          #name of the file
        self.HRegTime = {}      #begin and end time of the recording
        self.HolterInfo = {}    #various information of the recording
        self.RRInts = []        #values of the RR intervals
        self.Labels = []        #types of beat associated to the RR intervals:
                                #[N:normal, V:ventricular, A:atrial]
        
    def read_holter_file(self, fileName):        
        """
        Extracts the information contained in fileName
        Saves the file's information in a dictionary with the entrances described below:
            - Name: file's name
            - HolterInfo: First line of the file, Holter's various information
            - HRegTime: Begin and end time of the recording
            - RRInt: Values of the RR Intervals
            - Labels: Types of beat (N:normal, V:ventricular, A:atrial)
        """
        headerHolter = codecs.open(fileName, 'r', "iso-8859-1")    
        line1 = headerHolter.readline()
        line2 = headerHolter.readline()        
        headerHolter.close()
    
        #File name (first element of line 1)
        line1 = line1.split('\t')   #First line of the file
        self.name = line1[0]        #First element in first line of the file    
        line1 = line1[2:]           #Once saved, remove the name
    
        #Now we save the rest of the information of line 1 in HolterInfo   
        for elem in line1:
            name = elem.split(':')[0]
            value = elem.split(':')[1]
            self.HolterInfo[name] = value 
        
        #Begin and end time including in line 2       
        line2 = line2.split(' ')
        self.HRegTime[line2[3]] = line2[4]
        self.HRegTime[line2[5]] = line2[6]
     
        #From the third line to the end, we find the information of the RR intervals
        holter = np.loadtxt(fileName, dtype = str, skiprows = 2, delimiter = '\t')
        self.RRtimes = np.array(holter[:,0], dtype = str)       #First column
        self.RRInts = np.array(holter[:,1], dtype = float)      #Second column
        self.Labels = np.array(holter[:,2], dtype = str)        #Third column
        
        #Now we add all the values to the dictionary associated to the patient
        pat = {}
        pat['Name'] = self.name
        pat['HolterInfo'] = self.HolterInfo
        pat['HRegTime'] = self.HRegTime
        pat['RRInt'] = self.RRInts
        pat['Labels'] = self.Labels
        return pat
        
        
        
        
        
        
        
        