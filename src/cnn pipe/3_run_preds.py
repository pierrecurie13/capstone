#import pandas as pd
import numpy as np

# from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv1D
# from keras.layers import MaxPooling1D, Dropout
from keras.models import load_model
# import keras.backend as K
# from keras.utils.np_utils import to_categorical

# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

# K.set_image_data_format('channels_last')
# labEnc = LabelEncoder()

avg, std = np.load('models/normParamsSemifinal.npy')
clf = load_model('models/cnnSemifinal')

def makePred(x):
    x = (x-avg)/std
    preds = clf.predict(x.reshape(*x.shape,1))
    print(preds)
    return preds