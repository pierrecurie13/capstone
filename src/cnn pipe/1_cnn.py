import pandas as pd
import numpy as np
import pickle

#from keras import layers
from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv1D
from keras.layers import MaxPooling1D, Dropout
from keras.models import Model
#from keras.utils import layer_utils
#from keras.utils.data_utils import get_file
#from keras.applications.imagenet_utils import preprocess_input
#import pydot
#from IPython.display import SVG
#from keras.utils.vis_utils import model_to_dot
#from keras.utils import plot_model
#from kt_utils import *
import keras.backend as K
from keras.utils.np_utils import to_categorical

from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
#from matplotlib.pyplot import imshow

K.set_image_data_format('channels_last')
labEnc = LabelEncoder()

dirName = 'data/training2017/'
xTrain = np.load(dirName + 'train.npy')
yTrain = np.load(dirName + 'trainlabel.npy')
yTrain=yTrain[:,1]
yTrain=to_categorical(labEnc.fit_transform(yTrain))
xVal = np.load(dirName + 'valid.npy')
yVal = np.load(dirName + 'validlabel.npy')
yVal=yVal[:,1]
yVal=to_categorical(labEnc.fit_transform(yVal))

# # Normalize image vectors
# xTrain = xTrain_orig/255.
# xVal = xVal_orig/255.

# # Reshape
# yTrain = yTrain_orig.T
# yVal = yVal_orig.T

print ("number of training examples = " + str(xTrain.shape[0]))
print ("number of test examples = " + str(xVal.shape[0]))
print ("xTrain shape: " + str(xTrain.shape))
print ("yTrain shape: " + str(yTrain.shape))
print ("xVal shape: " + str(xVal.shape))
print ("yVal shape: " + str(yVal.shape))

#model
def cnn(input_shape):
    """
    Implementation of the HappyModel.
    
    Arguments:
    input_shape -- shape of the images of the dataset

    Returns:
    model -- a Model() instance in Keras
    """
    
    # Define the input placeholder as a tensor with shape input_shape. Think of this as your input image!
    X_input = Input(input_shape)

    # CONV -> BN -> RELU Block applied to X
    X = Conv1D(10, 8, strides=4, name = 'conv0')(X_input)
    #X = BatchNormalization(axis = 3, name = 'bn0')(X)
    X = Activation('relu')(X)

    # MAXPOOL
    X = MaxPooling1D(2, name='max_pool')(X)

    # FLATTEN X (means convert it to a vector) + FULLYCONNECTED
    X = Flatten()(X)
    X = Dense(1, activation='softmax', name='fc')(X)

    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = Model(inputs = X_input, outputs = X, name='derp')

    return model

clf = cnn((*(xTrain.shape[1:]),1))
clf.compile('adam','categorical_crossentropy',['accuracy'])
hist = clf.fit(xTrain.reshape(*xTrain.shape,1), yTrain, 16, 5, validation_data=(xVal.reshape(*xVal.shape,1),yVal))
hist = hist.history

plt.plot(hist['acc'])
plt.plot(hist['val_acc'])

clf.summary()
plot_model(clf)#, to_file='HappyModel.png')
SVG(model_to_dot(clf).create(prog='dot', format='svg'))