# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 16:33:39 2019

@author: MB207
"""
from keras.models import Model
from keras.applications.resnet50 import ResNet50
from keras.layers import Input,Flatten,Dense,concatenate,BatchNormalization,Lambda
from keras import backend as K
from keras.optimizers import Adam,SGD
from keras.callbacks import ModelCheckpoint
import numpy as np
from AMSoftmax import DenseWithAMSoftmaxLoss
def triplet_loss(y_true,y_pred,alpha = 1):
    total_lenght = y_pred.shape.as_list()[-1]
    anchor = y_pred[:,0:int(total_lenght*1/3)]
    positive = y_pred[:,int(total_lenght*1/3):int(total_lenght*2/3)]
    negative = y_pred[:,int(total_lenght*2/3):int(total_lenght*3/3)]

    # distance between the anchor and the positive
    pos_dist = K.sum(K.square(anchor-positive),axis=1)

    # distance between the anchor and the negative
    neg_dist = K.sum(K.square(anchor-negative),axis=1)

    # compute loss
    basic_loss = pos_dist-neg_dist+alpha
    loss = K.maximum(basic_loss,0)
 
    return loss

def resnet(classes,weight):
    net = ResNet50(include_top=False, weights='imagenet', input_tensor=None,
                   input_shape=(256,128,3))
    x = net.output
    x = BatchNormalization()(x)
    fla = Flatten()(x)
    out = DenseWithAMSoftmaxLoss(classes,0.35,30)(fla)
    model = Model(inputs=net.input,outputs=out)
    model.load_weights(weight)
    out = Dense(512)(fla)
    model = Model(inputs=net.input,outputs=out)
    return model

Weight_file = 'Weight_file_'
image_size = (256,128,3)
anchor_input = Input(image_size, name='anchor_input')
positive_input = Input(image_size, name='positive_input')
negative_input = Input(image_size, name='negative_input')

normalize = Lambda(lambda x:K.l2_normalize(x,axis=-1))
Shared_DNN = resnet(1041,Weight_file)
encoded_anchor = normalize(Shared_DNN(anchor_input))
encoded_positive = normalize(Shared_DNN(positive_input))
encoded_negative = normalize(Shared_DNN(negative_input))

merged_vector = concatenate([encoded_anchor, encoded_positive, encoded_negative], axis=-1, name='merged_layer')
model = Model(inputs=[anchor_input,positive_input, negative_input], outputs=merged_vector)

# %% Get DataSet
Anchor = np.load('ANCHOR_DATASET')
Positive = np.load('POSITIVE_DATASET')
Negative = np.load('NEGATIVE_DATASET')

# %%
log_dir = './923/'
checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='val_loss', save_weights_only=True, period=3)
Y_dummy = np.empty((Anchor.shape[0],10))

v = -1
model.compile(optimizer=Adam(1e-5),loss=triplet_loss)
model.fit([Anchor[:v],Positive[:v],Negative[:v]],y=Y_dummy[:v],batch_size=8,epochs=80,validation_split=0.2,callbacks=[checkpoint])
