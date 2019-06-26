import sys
import numpy as np
import pandas as pd
from scipy.misc import imread
import pickle
import os
import matplotlib.pyplot as plt
import cv2
import time
import tensorflow as tf
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras.regularizers import l2
from keras import backend as K
from sklearn.utils import shuffle
import numpy.random as rng
import utill as ut
import model as md

save_path = './save/'
model_path = './weights/'

with open(os.path.join(save_path, "train.pickle"), "rb") as f:
    (Xtrain, train_classes) = pickle.load(f)

with open(os.path.join(save_path, "val.pickle"), "rb") as f:
    (Xval, val_classes) = pickle.load(f)
    
model = md.get_siamese_model((105, 105, 1))
model.summary()

model.load_weights(os.path.join(model_path, "weights.20000.h5"))

ways = np.arange(1,21,1)
resume =  False
trials = 20

val_accs, train_accs,nn_accs = [], [], []
for N in ways:    
    val_accs.append(ut.test_oneshot(model, N, trials, Xval, val_classes, "val", verbose=True))
    #train_accs.append(ut.test_oneshot(model, N, trials, Xtrain, train_classes, "train", verbose=True))
    #print ("train accuracy = ", train_accs, " validation accuracy = ", val_accs)
    print("---------------------------------------------------------------------------------------------------------------")