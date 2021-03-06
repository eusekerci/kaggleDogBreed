 # -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import scipy
import numpy as np
import argparse
import pandas as pd
from scipy.stats import itemfreq

parser = argparse.ArgumentParser(description='Decide if an image is a picture of a bird')
parser.add_argument('image', type=str, help='The image image file to check')
args = parser.parse_args()

nwidth = 32
nheight = 32

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
    print(labels_name)

    for _, i in enumerate(itemfreq(labels0)[:,0].astype(int)):
        labels_bin0 = np.where(labels0 == itemfreq(labels0)[:,0][i], 1.0, 0.0)
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
labels_cls = np.argmax(labels_bin, axis=1)

# Same network definition as before
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)
img_aug.add_random_blur(sigma_max=3.)

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
model.load("dog_breed_identification.tfl.ckpt-7500")

print("Test")

# Load the image file
img = scipy.ndimage.imread(args.image, mode="RGB")

# Scale it to 32x32
img = scipy.misc.imresize(img, (32, 32), interp="bicubic").astype(np.float32, casting='unsafe')

# Predict
prediction = model.predict([img])

pred = np.argmax(prediction, axis=1)
itemindex = np.where(labels_cls==pred)
print(labels_cls[itemindex][0])
print(prediction)
print(pred)
print(labels_name[labels_cls[itemindex][0]])
