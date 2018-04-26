import pandas as pd
import numpy as np

df = pd.read_csv('data/training2017/REFERENCE.csv', header=None, names = ['name','label'])
df = df[df.label != 'O']
test = df.sample(frac=.2)
trainIdx = list(set(df.index) - set(test.index))
train = df.loc[trainIdx]

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