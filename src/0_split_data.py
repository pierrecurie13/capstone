import pandas as pd
import numpy as np
import wfdb
import pickle

dirName = 'data/training2017/'
df = pd.read_csv(dirName + 'REFERENCE.csv', header=None, names = ['name','label'])
df = df[df.label != 'O']
test = df.sample(frac=.2)
trainIdx = list(set(df.index) - set(test.index))
train = df.loc[trainIdx]

valid = train.sample(frac=.2)
trainIdx = list(set(train.index) - set(valid.index))
train = df.loc[trainIdx]

sigs, ys = consolidateData(train, dirName)
np.save(dirName + 'train', sigs)
np.save(dirName + 'trainlabel', ys)

sigs, ys = consolidateData(test, dirName)
np.save(dirName + 'test', sigs)
np.save(dirName + 'testlabel', ys)

sigs, ys = consolidateData(valid, dirName)
np.save(dirName + 'valid', sigs)
np.save(dirName + 'validlabel', ys)

def chunk(signal, y, chunkSize=4096):
    '''
    chunk signal into chunks of size chunkSize
    Input: signal: nonempty 1D np array or list, y: label, chunkSize: int>0
    Output: 2D numpy array of chunked signal and its label
    '''
    reps=1
    if len(signal)<chunkSize: #loop signal until it's long enough
        tmp=signal
        for _ in range(chunkSize//len(signal)):
            tmp=np.hstack([tmp,signal]) #not fastest; fix later
        signal = [tmp[:chunkSize]]

    elif len(signal)>chunkSize:
        offset = np.random.randint(0,len(signal)%chunkSize + 1)
        out = []
        reps=len(signal)//chunkSize
        for i in range(0,reps):
            tmp=signal[offset+i*chunkSize:offset+(i+1)*chunkSize]
            out.append(tmp)
        signal=out

    else:
        signal =[signal]

    y=[y]*reps

    return signal, y

def consolidateData(df, dirName, chunkSize=4096):
    '''
    turn dataset into single file, with signals chunked into size chunkSize
    Input: df: dataframe with cols 'name' and 'label'; name is name of file
        dirName: dir where file can be found
        chunkSize: int>0
    Output: 2D numpy array of chunked signals and their labels
    '''
    sigOut=[]
    yOut=[]
    for _, row in df.iterrows():
        fname = dirName + row['name']
        sig, _ = wfdb.rdsamp(fname)
        sig=sig.reshape(-1)
        sigs, ys = chunk(sig, [row['name'], row.label], chunkSize)
        sigOut.extend(sigs)
        yOut.extend(ys)
    return np.array(sigOut), np.array(yOut)