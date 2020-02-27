# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 20:55:50 2019

@author: MB207
"""

from keras.models import Model
from keras.layers import Input,Flatten,Dense,concatenate,Dropout,Conv2D,MaxPooling2D,BatchNormalization,Lambda,GlobalAveragePooling2D
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras import backend as K
from keras.regularizers import l2

def resnet():
    net = ResNet50(include_top=False, weights='imagenet', input_tensor=None,
                   input_shape=(256,128,3))
    x = net.output
    x = BatchNormalization()(x)
    x = Flatten()(x)
    out = Dense(512)(x)
    
    model = Model(inputs=net.input,outputs=out)
    return model

def Aresnet():
    net = ResNet50(include_top=False, weights='imagenet', input_tensor=None,
                   input_shape=(256,128,3))
    x = net.output
    x = BatchNormalization()(x)
    x = GlobalAveragePooling2D('channels_last')(x)
    out = Dense(512)(x)
    model = Model(inputs=net.input,outputs=out)
    return model


def vgg19(in_dims):
    
    Inputs = Input(shape=(in_dims[0],in_dims[1],in_dims[2]),)
    conv1 = Conv2D(64,kernel_size=(3,3),padding='same',activation='relu')(Inputs)
    conv2 = Conv2D(64,kernel_size=(3,3),padding='same',activation='relu')(conv1)    
    pool = MaxPooling2D(pool_size=(3,3),padding='same')(conv2)    
    
    conv1 = Conv2D(128,kernel_size=(3,3),padding='same',activation='relu')(pool)
    conv2 = Conv2D(128,kernel_size=(3,3),padding='same',activation='relu')(conv1)    
    pool = MaxPooling2D(pool_size=(3,3),padding='same')(conv2)

    conv1 = Conv2D(256,kernel_size=(3,3),padding='same',activation='relu')(pool)
    conv2 = Conv2D(256,kernel_size=(3,3),padding='same',activation='relu')(conv1)    
    pool = MaxPooling2D(pool_size=(3,3),padding='same')(conv2)   

    conv1 = Conv2D(256,kernel_size=(3,3),padding='same',activation='relu')(pool)
    conv2 = Conv2D(256,kernel_size=(3,3),padding='same',activation='relu')(conv1)    
    pool = MaxPooling2D(pool_size=(3,3),padding='same')(conv2)   

    conv1 = Conv2D(256,kernel_size=(3,3),padding='same',activation='relu')(pool)
    conv2 = Conv2D(256,kernel_size=(3,3),padding='same',activation='relu')(conv1)    
    pool = MaxPooling2D(pool_size=(3,3),padding='same')(conv2)   
    
    Fla = Flatten()(pool)
    BN = BatchNormalization()(Fla)
    D = Dropout(0.6)(BN)
    Outputs = Dense(512)(D)
    model = Model(inputs=Inputs,outputs=Outputs)
    return model

def Aload_model(path):
    image_size = (256,128,3)
    anchor_input = Input(image_size, name='anchor_input')
    positive_input = Input(image_size, name='positive_input')
    negative_input = Input(image_size, name='negative_input')
    
    normalize = Lambda(lambda x:K.l2_normalize(x,axis=-1))
    Shared_DNN = Aresnet()
    encoded_anchor = normalize(Shared_DNN(anchor_input))
    encoded_positive = normalize(Shared_DNN(positive_input))
    encoded_negative = normalize(Shared_DNN(negative_input))
    
    merged_vector = concatenate([encoded_anchor, encoded_positive, encoded_negative], axis=-1, name='merged_layer')
    model = Model(inputs=[anchor_input,positive_input, negative_input], outputs=merged_vector)
    model.load_weights(path)
    reid = Model(anchor_input,encoded_anchor)
    return reid


def load_model(path):
    image_size = (256,128,3)
    anchor_input = Input(image_size, name='anchor_input')
    positive_input = Input(image_size, name='positive_input')
    negative_input = Input(image_size, name='negative_input')
    
    normalize = Lambda(lambda x:K.l2_normalize(x,axis=-1))
    Shared_DNN = resnet()
    encoded_anchor = normalize(Shared_DNN(anchor_input))
    encoded_positive = normalize(Shared_DNN(positive_input))
    encoded_negative = normalize(Shared_DNN(negative_input))
    
    merged_vector = concatenate([encoded_anchor, encoded_positive, encoded_negative], axis=-1, name='merged_layer')
    model = Model(inputs=[anchor_input,positive_input, negative_input], outputs=merged_vector)
    model.load_weights(path)
    reid = Model(anchor_input,encoded_anchor)
    return reid


