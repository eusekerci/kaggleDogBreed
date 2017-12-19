# -*- coding: utf-8 -*-

""" 
Based on tflearn example:
https://github.com/tflearn/tflearn/blob/master/examples/images/convnet_cifar10.py
"""

from __future__ import division, print_function, absolute_import

import h5py
import tensorflow as tf
import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation

from zipfile import ZipFile
from io import BytesIO

import PIL.Image
from IPython.display import display

import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import itemfreq
from sklearn.model_selection import train_test_split

tf.__version__

nwidth = 32
nheight = 32

"""
archive_train = ZipFile("Data/train.zip", 'r')
archive_test = ZipFile("Data/test.zip", 'r')

s = (len(archive_train.namelist()[:])-1, nwidth, nheight,3)
allImage = np.zeros(s)
for i in range(1,len(archive_train.namelist()[:])):
    filename = BytesIO(archive_train.read(archive_train.namelist()[i]))
    image = PIL.Image.open(filename)
    image = image.resize((nwidth, nheight))
    image = np.array(image)
    image = np.clip(image/255.0, 0.0, 1.0)
    allImage[i-1]=image
    
pickle.dump(allImage, open( "Data/train" + '.p', "wb" ) )

s = (len(archive_test.namelist()[:])-1, nwidth, nheight,3)
allImage = np.zeros(s)
for i in range(1,len(archive_test.namelist()[:])):
    filename = BytesIO(archive_test.read(archive_test.namelist()[i]))
    image = PIL.Image.open(filename)
    image = image.resize((nwidth, nheight))
    image = np.array(image)
    image = np.clip(image/255.0, 0.0, 1.0)
    allImage[i-1]=image
    
pickle.dump(allImage, open( "Data/test" + '.p', "wb" ) )
"""

train = pickle.load( open( "Data/train.p", "rb" ) )
test = pickle.load( open( "Data/test.p", "rb" ) )

train_label_raw = pd.read_csv('Data/labels.csv')
train_label = train_label_raw["breed"].as_matrix()
train_label = train_label.reshape(train_label.shape[0],1)
print(train_label.shape)

##We convert our labels into binary arrays to give CNN
##Thanks to:
##https://www.kaggle.com/kaggleslayer/simple-convolutional-n-network-with-tensorflow
def matrix_Bin(train_label):
    labels_bin = np.array([])

    labels_name, labels0 = np.unique(train_label, return_inverse=True)
    print(labels0)

    for _, i in enumerate(itemfreq(labels0)[:,0].astype(int)):
        labels_bin0 = np.where(labels0 == itemfreq(labels0)[:,0][i], 1., 0.)
        labels_bin0 = labels_bin0.reshape(1, labels_bin0.shape[0])
        if(labels_bin.shape[0] == 0):
           labels_bin = labels_bin0
        else:
            labels_bin = np.concatenate((labels_bin, labels_bin0), axis=0)

    print("Nber SubVariables {0}".format(itemfreq(labels0)[:,0].shape[0]))
    labels_bin = labels_bin.transpose()
    print("Shape : {0}".format(labels_bin.shape))

    return labels_name, labels_bin

labels_name, labels_bin = matrix_Bin(train_label = train_label)


"""
exm_img = train[62,:,:,:]
plt.imshow(lum_img)
plt.show()
"""


#num_validation = 0.30
#X_train, X_test, Y_train, Y_test = train_test_split(train_filtered, labels_bin, test_size=num_validation, random_state=6)
num_validation = 0.30
X_train, X_test, Y_train, Y_test = train_test_split(train, labels_bin, test_size=num_validation, random_state=6)

# Real-time data preprocessing
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

# Real-time data augmentation
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)

# Convolutional network building
network = input_data(shape=[None, nwidth, nheight, 3],
                     data_preprocessing=img_prep,
                     data_augmentation=img_aug)
network = conv_2d(network, 32, 3, activation='relu')
network = max_pool_2d(network, 2)
network = conv_2d(network, 64, 3, activation='relu')
network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2)
network = fully_connected(network, 512, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 120, activation='softmax')
network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)

# Train using classifier
model = tflearn.DNN(network, tensorboard_verbose=0, checkpoint_path='dog_breed_identification.tfl.ckpt')
model.fit(X_train, Y_train, n_epoch=100, shuffle=True, validation_set=(X_test, Y_test),
          show_metric=True, batch_size=96, run_id='dog_breed')

model.save("dog_breed_identification.tfl")

