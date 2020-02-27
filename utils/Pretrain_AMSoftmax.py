# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 16:50:55 2019

@author: user
"""

import numpy as np
import cv2
from os import listdir

TrainPath = './DataSet'
Trainlist = listdir(TrainPath)

x = []
y = []

for e in range(len(Trainlist)):
    eDir = TrainPath + Trainlist[e] + '/'
    for filename in listdir(eDir):
        img = cv2.imread(eDir + filename)
        if type(img) == type(None):continue
        img = cv2.resize(img,(128,256))
        x.append(img)
        y.append(e)
        
x = np.array(x)
y = np.array(y)
# %%
from keras.models import Model
from keras.layers import Input,Flatten,Dense,concatenate,Dropout,Conv2D,MaxPooling2D,BatchNormalization,Lambda
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras import backend as K
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.layers import activations, initializers, regularizers, constraints, Lambda
from keras.engine import InputSpec
import tensorflow as tf
from keras.engine.topology import Layer

class DenseWithAMSoftmaxLoss(Layer):

    def __init__(self, num_classes, m=0.35, scale=30, **kwargs):
        self.output_dim = num_classes
        self.m = m
        self.scale = scale
        super(DenseWithAMSoftmaxLoss, self).__init__(**kwargs)

    def build(self, input_shape):

        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='glorot_normal',
                                      trainable=True)

    def call(self, inputs):
        self.inputs = inputs
        self.w_norm = K.tf.nn.l2_normalize(self.kernel, 0, 1e-10)
        self.x_norm = K.tf.nn.l2_normalize(self.inputs, 1, 1e-10)
        self.logits = K.dot(self.x_norm, self.w_norm)
        return self.logits

    def compute_output_shape(self, input_shape):
        return (input_shape[0],
                self.output_dim)

    def loss_dense(self, y_true, y_pred):
        y_true = K.cast(y_true, 'float32')
        thelta_m = y_pred - y_true * self.m  # cosine(thelta)-m ; y_true 就相当于mask

        return K.categorical_crossentropy(y_true, self.scale * thelta_m, from_logits=True)
# %%

def resnet(classes):
    net = ResNet50(include_top=False, weights='imagenet', input_tensor=None,
                   input_shape=(256,128,3))
    x = net.output
    x = BatchNormalization()(x)
    x = Flatten()(x)
#    out = Dense(classes,activation='softmax')(x)
    out = DenseWithAMSoftmaxLoss(classes,0.35,30)(x)
    model = Model(inputs=net.input,outputs=out)
    return model

model = resnet(max(y)+1)
# %%
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
model.compile(optimizer=Adam(lr=1e-5),loss=DenseWithAMSoftmaxLoss(max(y)+1,0.35,30).loss_dense, metrics=['accuracy'])
log_dir = './pretrain/'
checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='val_loss', save_weights_only=True, period=3)
X_train, X_test, y_train, y_test = train_test_split((x/.255).astype(np.float32), to_categorical(y), test_size=0.3, random_state=0)
model.fit(x=X_train,y=y_train,batch_size=8,epochs=99,validation_data=(X_test,y_test),callbacks=[checkpoint])
