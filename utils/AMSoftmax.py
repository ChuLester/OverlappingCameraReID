# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 01:32:17 2019

@author: user
"""

from keras import backend as K
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