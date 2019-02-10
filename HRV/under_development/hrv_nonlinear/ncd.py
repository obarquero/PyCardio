#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 09:21:48 2017

@author: obarquero
"""
from __future__ import division
import numpy as np
import zlib
from backports import lzma
import bz2



class NCD(object):


   def __init__(self,x=[],y=[],compressor = []):
       
       self.x = x
       self.y = y
       self.compressor = compressor
       
   def ncd(self,x,y,compressor):
       """
       Compute ncd value.
       
       Example: ncd(x,y,zlib)
       
       where compressor can be zlib lzma bz2
       """
       xy = np.concatenate((x,y))
       c_x = compressor.compress(x)
       c_y = compressor.compress(y)
       c_xy = compressor.compress(xy)
       ncd = (len(c_xy) - np.min((len(c_x),len(c_y)))) / (np.max((len(c_x),len(c_y))))
       
       return ncd
       ##ncd = (c_xy-min(cx,cy))/(max(Cx,Cy))
       
   def __ncd(self):
       xy = np.concatenate((self.x,self.y))
       c_x = self.compressor.compress(self.x)
       c_y = self.compressor.compress(self.y)
       c_xy = self.compressor.compress(xy)
       ncd = (len(c_xy) - np.min((len(c_x),len(c_y)))) / (np.max((len(c_x),len(c_y))))
       
       return ncd
       ##ncd = (c_xy-min(cx,cy))/(max(Cx,Cy))   
   def getX(self):
       return self.x
   
   def getY(self):
       return self.y

   def getCompressor(self):
       return self.compressor
       

      