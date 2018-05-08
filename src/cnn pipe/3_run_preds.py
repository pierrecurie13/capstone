import numpy as np
from keras.models import load_model

avg, std = np.load('models/normParamsSemifinal.npy')
clf = load_model('models/cnnSemifinal')

def makePred(x):
    x = (x-avg)/std
    preds = clf.predict(x.reshape(*x.shape,1))
    return preds