import pandas as pd
import numpy as np
import pickle

from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv1D
from keras.layers import MaxPooling1D, Dropout
from keras.models import Model, model_from_json
#import pydot
#from IPython.display import SVG
import keras.backend as K
from keras.utils.np_utils import to_categorical

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
#from matplotlib.pyplot import imshow

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

def plot_score(y, y_pred):
    mask = y[:,2]==0
    y=y[mask]
    y_pred=y_pred[mask]
    y_pred=y_pred[:,0]/(y_pred[:,0]+y_pred[:,1])
    mask = np.isnan(y_pred)
    y_pred[mask]=.5
    fpr, tpr, thresh = roc_curve(y[:,0], y_pred)
    plt.scatter(fpr,tpr)
    plt.show()
    return pd.DataFrame([fpr, tpr, thresh]).T

def score_func2(y, y_pred):
    y=y[:0]
    y_pred=y_pred[:0]
    return roc_auc_score(y, y_pred)

def plot_score2(y, y_pred):
    y=y[:0]
    y_pred=y_pred[:0]
    fpr, tpr, thresh = roc_curve(y, y_pred)
    plt.scatter(fpr,tpr)
    plt.show()
    return pd.DataFrame([fpr, tpr, thresh]).T

def confMat(y, y_pred):
    y = np.argmax(y,axis=1)
    y_pred=np.argmax(y_pred,axis=1)
    return confusion_matrix(y, y_pred).T

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
    X = Conv1D(16, 8, strides=8, name = 'conv0')(X_input)
    X = BatchNormalization(name = 'bn0')(X)
    X = MaxPooling1D(2, name='max_pool')(X)
    X = Activation('relu')(X)

    X = Dropout(.15)(X)
    X = Conv1D(32, 4, strides=4, name = 'conv1')(X)
    X = BatchNormalization(name = 'bn1')(X)
    X = MaxPooling1D(2, name='max_pool1')(X)
    X = Activation('relu')(X)

    X = Dropout(.3)(X)
    X = Conv1D(64, 2, strides=2, name = 'conv2')(X)
    X = BatchNormalization(name = 'bn2')(X)
    X = MaxPooling1D(2, name='max_pool2')(X)
    X = Activation('relu')(X)

    X = Dropout(.3)(X)
    X = Conv1D(64, 2, strides=2, name = 'conv3')(X)
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

def cnn2(input_shape):
    """
    Implementation of the HappyModel.
    
    Arguments:
    input_shape -- shape of the images of the dataset

    Returns:
    model -- a Model() instance in Keras
    """
    # Define the input placeholder as a tensor with shape input_shape. Think of this as your input image!
    X_input = Input(input_shape)

    X = BatchNormalization(name = 'bnm1')(X_input)
    # CONV -> BN -> RELU Block applied to X
    X = Conv1D(16, 8, strides=8, name = 'conv0')(X)
    X = BatchNormalization(name = 'bn0')(X)
    X = MaxPooling1D(2, name='max_pool')(X)
    X = Activation('relu')(X)

    X = Dropout(.15)(X)
    X = Conv1D(32, 4, strides=4, name = 'conv1')(X)
    X = BatchNormalization(name = 'bn1')(X)
    X = MaxPooling1D(2, name='max_pool1')(X)
    X = Activation('relu')(X)

    X = Dropout(.3)(X)
    X = Conv1D(64, 2, strides=2, name = 'conv2')(X)
    X = BatchNormalization(name = 'bn2')(X)
    X = MaxPooling1D(2, name='max_pool2')(X)
    X = Activation('relu')(X)

    X = Dropout(.3)(X)
    X = Conv1D(64, 2, strides=2, name = 'conv3')(X)
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

def cnnOverlap(input_shape):
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

def cnnOverlapOneDrop(input_shape):
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
    X = Conv1D(16, 8, strides=4, name = 'conv0')(X_input)
    X = BatchNormalization(name = 'bn0')(X)
    X = MaxPooling1D(2, name='max_pool')(X)
    X = Activation('relu')(X)

    X = Conv1D(32, 4, strides=2, name = 'conv1')(X)
    X = BatchNormalization(name = 'bn1')(X)
    X = MaxPooling1D(2, name='max_pool1')(X)
    X = Activation('relu')(X)

    X = Conv1D(64, 4, strides=2, name = 'conv2')(X)
    X = BatchNormalization(name = 'bn2')(X)
    X = MaxPooling1D(2, name='max_pool2')(X)
    X = Activation('tanh')(X)

    X = Dropout(.3)(X)
    X = Conv1D(64, 4, strides=2, name = 'conv3')(X)
    X = MaxPooling1D(2, name='max_pool3')(X)
    X = Activation('tanh')(X)

    X = Dropout(.3)(X)
    # FLATTEN X (means convert it to a vector) + FULLYCONNECTED
    X = Flatten()(X)
    X = Dense(3, activation='softmax', name='fc')(X)

    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = Model(inputs = X_input, outputs = X, name='derp')
    return model

def cnnOverlapDeeper(input_shape):
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
    X = Conv1D(16, 8, strides=4, name = 'conv0')(X_input)
    X = BatchNormalization()(X)
    X = MaxPooling1D(2, name='max_pool')(X)
    X = Activation('relu')(X)

    # X = Conv1D(32, 4, strides=2, name = 'conv1')(X)
    # X = BatchNormalization()(X)
    # X = MaxPooling1D(2)(X)
    # X = Activation('relu')(X)

    X = Conv1D(32, 4, strides=2, name = 'conv2')(X)
    X = BatchNormalization()(X)
    X = MaxPooling1D(2, name='max_pool1')(X)
    X = Activation('relu')(X)

    X = Conv1D(64, 4, strides=2, name = 'conv3')(X)
    X = BatchNormalization()(X)
    #X = MaxPooling1D(2, name='max_pool2')(X)
    X = Activation(K.square)(X)

    X = Dropout(.2)(X)
    X = Conv1D(64, 4, strides=2, name = 'conv4')(X)
    X = MaxPooling1D(2, name='max_pool3')(X)
    X = Activation('relu')(X)

    #X = Dropout(.2)(X)
    # FLATTEN X (means convert it to a vector) + FULLYCONNECTED
    X = Flatten()(X)
    # X = Dense(8, activation='tanh', name='fc2')(X)

    X = Dropout(.2)(X)
    X = Dense(3, activation='softmax', name='fc')(X)

    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = Model(inputs = X_input, outputs = X, name='derp')
    return model

clf = cnnOverlapDeeper((*(xTrain.shape[1:]),1))
clf.compile('adam','categorical_crossentropy',['accuracy'])

hist = clf.fit(xTrain.reshape(*xTrain.shape,1), yTrain, 16, 100, validation_data=(xVal.reshape(*xVal.shape,1),yVal))
hist = hist.history

plt.plot(hist['acc'])
plt.plot(hist['val_acc'])
plt.show()

preds = clf.predict(xTrain.reshape(*xTrain.shape,1))
score_func(yTrain, preds)

preds = clf.predict(xVal.reshape(*xVal.shape,1))
score_func(yVal, preds)

# serialize model to JSON
model_json = clf.to_json()
with open("data/models/modelOverlap.json", "w") as f:
    f.write(model_json)
# serialize weights to HDF5
clf.save_weights("data/models/modelOverlap.h5")


# load json and create model
with open('data/models/modelOverlap.json', 'r') as f:
    loaded_model_json = f.read()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")


clf.summary()
plot_model(clf)#, to_file='HappyModel.png')
SVG(model_to_dot(clf).create(prog='dot', format='svg'))