import pandas as pd
import numpy as np
import wfdb
from wfdb import processing
from scipy.stats import kurtosis as kurt, skew
import pickle

dirName = 'data/training2017/'
toFeats(dirName, 'train')
toFeats(dirName, 'test')
toFeats(dirName, 'valid')

def toFeats(dirName, fName):
    x = np.load(dirName + fName + '.npy')
    feats=[]
    for row in x:
        feats.append(genFeat(row, 300))
    np.save(dirName + fName + 'Feat', np.array(feats))

def genFeat(signal, fs):
    '''
    given signal, return list of features
    Input: signal: nonempty 1D np array, fs: sampling rate, int>0
    Output: 1D np array
    '''
    idx = processing.xqrs_detect(signal, fs, verbose=False)
    if len(idx)<2:
        return np.zeros(11)
    delta = idx[1:]-idx[:-1]
    out = [delta.mean(), delta.std(), delta.var(),
        skew(delta), kurt(delta), delta.max()-delta.min()]
    rVals=signal[idx]
    out.extend([rVals.std(), rVals.var(), skew(rVals),
        kurt(rVals), rVals.max()-rVals.min()])
    return np.array(out)