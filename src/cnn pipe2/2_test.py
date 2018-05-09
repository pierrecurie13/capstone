import pandas as pd
import numpy as np

from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv1D
from keras.layers import MaxPooling1D, Dropout
from keras.models import Model, model_from_json
import keras.backend as K
from keras.utils.np_utils import to_categorical

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

K.set_image_data_format('channels_last')
labEnc = LabelEncoder()

def score_func(y, y_pred):
    mask = y[:,2]==0
    y=y[mask]
    y_pred=y_pred[mask]
    y_pred=y_pred[:,0]/(y_pred[:,0]+y_pred[:,1])
    mask = np.isnan(y_pred)
    y_pred[mask]=.5
    return roc_auc_score(y[:,0], y_pred)

def score_func2(y, y_pred):
    y=(y[:,0]==1)|(y[:,2]==1)
    y_pred=y_pred[:,0]+y_pred[:,2]
    return roc_auc_score(y, y_pred)

def score_func3(y, y_pred):
    mask = (y[:,2]==0)&(y[:,3]==0)
    y=y[mask]
    y_pred=y_pred[mask]
    y_pred=y_pred[:,0]/(y_pred[:,0]+y_pred[:,1])
    mask = np.isnan(y_pred)
    y_pred[mask]=.5
    return roc_auc_score(y[:,0], y_pred)

def confMat(y, y_pred):
    #each row corresponds to a true value
    #each col corresponds to a prediction
    y = np.argmax(y,axis=1)
    y_pred=np.argmax(y_pred,axis=1)
    return confusion_matrix(y, y_pred)

dirName = 'training2017/'
xTrain = np.load(dirName + 'train.npy')
xTrain = np.vstack([xTrain, np.load(dirName + 'train2.npy')])
yTrain = np.load(dirName + 'trainlabel.npy')
yTrain = np.vstack([yTrain, np.load(dirName + 'trainlabel2.npy')])
yTrain=yTrain[:,1]
yTrain=to_categorical(labEnc.fit_transform(yTrain))
xVal = np.load(dirName + 'valid.npy')
xVal = np.vstack([xVal, np.load(dirName + 'valid2.npy')])
yVal = np.load(dirName + 'validlabel.npy')
yVal = np.vstack([yVal, np.load(dirName + 'validlabel2.npy')])
yVal=yVal[:,1]
yVal=to_categorical(labEnc.fit_transform(yVal))

# # Normalize
avg = xTrain.mean()
std = xTrain.std()
xTrain = (xTrain-avg)/std
xVal = (xVal-avg)/std

def cnnOverlap(input_shape):
    """
    adding overlap in the conv layers, with batchnorm and dropouts
    
    Arguments:
    input_shape -- shape of the images of the dataset

    Returns:
    model -- a Model() instance in Keras
    """
    # Define the input placeholder as a tensor with shape input_shape. Think of this as your input image!
    X_input = Input(input_shape)

    # CONV -> BN -> RELU Block applied to X
    X = Conv1D(16, 8, strides=4, name = 'conv0')(X_input)
    X = BatchNormalization(name = 'bn0')(X)
    X = MaxPooling1D(2, name='max_pool')(X)
    X = Activation('relu')(X)

    X = Dropout(.1)(X)
    X = Conv1D(32, 4, strides=2, name = 'conv1')(X)
    X = BatchNormalization(name = 'bn1')(X)
    X = MaxPooling1D(2, name='max_pool1')(X)
    X = Activation('relu')(X)

    X = Dropout(.15)(X)
    X = Conv1D(64, 4, strides=2, name = 'conv2')(X)
    X = BatchNormalization(name = 'bn2')(X)
    X = MaxPooling1D(2, name='max_pool2')(X)
    X = Activation('relu')(X)

    X = Dropout(.2)(X)
    X = Conv1D(64, 4, strides=2, name = 'conv3')(X)
    X = BatchNormalization(name = 'bn3')(X)
    X = MaxPooling1D(2, name='max_pool3')(X)
    X = Activation('relu')(X)

    X = Dropout(.1)(X)
    # FLATTEN X (means convert it to a vector) + FULLYCONNECTED
    X = Flatten()(X)
    X = Dense(4, activation='softmax', name='fc')(X)

    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = Model(inputs = X_input, outputs = X, name='derp')
    return model


def run(model, epochs=100):
    clf = model((*(xTrain.shape[1:]),1))
    clf.compile('adam','categorical_crossentropy',['accuracy'])

    hist = clf.fit(xTrain.reshape(*xTrain.shape,1), yTrain, 16, epochs, validation_data=(xVal.reshape(*xVal.shape,1),yVal))
    return clf, hist.history

clf, hist = run(cnnOverlap, 100)

# is about best?

preds = clf.predict(xVal.reshape(*xVal.shape,1))
score_func(yVal, preds)
score_func2(yVal, preds)

#####
#final train
#####

dirName = 'training2017/'
xTrain = np.load(dirName + 'train.npy')
xTrain = np.vstack([xTrain, np.load(dirName + 'train2.npy')])
yTrain = np.load(dirName + 'trainlabel.npy')
yTrain = np.vstack([yTrain, np.load(dirName + 'trainlabel2.npy')])
xVal = np.load(dirName + 'valid.npy')
xVal = np.vstack([xVal, np.load(dirName + 'valid2.npy')])
yVal = np.load(dirName + 'validlabel.npy')
yVal = np.vstack([yVal, np.load(dirName + 'validlabel2.npy')])

xTrain=np.vstack([xTrain,xVal])
yTrain=np.vstack([yTrain,yVal])
yTrain=yTrain[:,1]
yTrain=to_categorical(labEnc.fit_transform(yTrain))

xVal = np.load(dirName + 'test.npy')
xVal = np.vstack([xVal, np.load(dirName + 'test2.npy')])
yVal = np.load(dirName + 'testlabel.npy')
yVal = np.vstack([yVal, np.load(dirName + 'testlabel2.npy')])
yVal=yVal[:,1]
yVal=to_categorical(labEnc.fit_transform(yVal))

# # # Normalize
avg = xTrain.mean()
std = xTrain.std()
xTrain = (xTrain-avg)/std
xVal = (xVal-avg)/std

clf, hist = run(cnnOverlap, 100)

