import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv1D
from keras.layers import MaxPooling1D, Dropout
from keras.models import Model
import keras.backend as K

K.set_image_data_format('channels_last')
labEnc = LabelEncoder()

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

    X = Dropout(.2)(X)
    X = Conv1D(64, 4, strides=2, name = 'conv2')(X)
    X = BatchNormalization(name = 'bn2')(X)
    X = MaxPooling1D(2, name='max_pool2')(X)
    X = Activation('relu')(X)

    X = Dropout(.3)(X)
    X = Conv1D(64, 4, strides=2, name = 'conv3')(X)
    X = BatchNormalization(name = 'bn3')(X)
    X = MaxPooling1D(2, name='max_pool3')(X)
    X = Activation('relu')(X)

    X = Dropout(.15)(X)
    # FLATTEN X (means convert it to a vector) + FULLYCONNECTED
    X = Flatten()(X)
    X = Dense(3, activation='softmax', name='fc')(X)

    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = Model(inputs = X_input, outputs = X, name='derp')
    return model

dirName = 'training2017/'
xVal = np.load(dirName + 'test.npy')
yVal = np.load(dirName + 'testlabel.npy')
yVal=yVal[:,1]
labels=labEnc.fit(yVal).classes_

avg, std = np.load('models/normParamsSemifinal.npy')
clf = cnnOverlap((4096,1))
clf.load_weights('models/cnnSemifinal')

def plotECG(x):
    plt.subplot(211)
    plt.plot(x[:2048])
    plt.subplot(212)
    plt.plot(x[2048:])
    plt.show()

def makePred(x):
    x = (x-avg)/std
    preds = clf.predict(x.reshape(*x.shape,1))
    return preds

def miniGame(idx=None):
    if idx==None:
        idx=np.random.randint(0,len(xVal))
    if idx >= len(xVal):
        return None
    x=xVal[idx]
    plotECG(x)
    guess=input('What do you think this is?\n')
    print('\nYour guess is ' + guess + '\n')
    preds=makePred(x.reshape(1,-1))
    print('The computer estimates:')
    print(labels)
    print(preds[0])
    print("\nThe computer's guess is {}".format(labels[np.argmax(preds)]))
    print("The correct answer is " + yVal[idx])