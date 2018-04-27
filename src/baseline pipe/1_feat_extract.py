import pandas as pd
import numpy as np
import wfdb
import pickle

dirName = 'data/training2017/'

x = np.load(dirName + 'train')
y = np.load(dirName + 'trainlabel')



def genFeat(signal, fs):
    '''
    chunk signal into chunks of size chunkSize
    Input: signal: nonempty 1D np array or list, y: label, chunkSize: int>0
    Output: 2D numpy array of chunked signal and its label
    '''

