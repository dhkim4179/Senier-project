# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 18:55:03 2021
"""

import os
import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers,optimizers
import tensorflow as tf
import matplotlib.pyplot as plt


def rotate_270(m):
    N = len(m)
    ret = [[0] * N for _ in range(N)]

    for r in range(N):
        for c in range(N):
            ret[N-1-c][r] = m[r][c]
    return np.array(ret)

def rotate_90(m):
    N = len(m)
    ret = [[0] * N for _ in range(N)]
    # 'ret = [[0] * N] * N' 과 다름

    for r in range(N):
        for c in range(N):
            ret[c][N-1-r] = m[r][c]
    return np.array(ret)
def parse_data(data):
    image_array = np.zeros(shape=(len(data), 48, 48, 1))
    image_label = np.array(list(map(int, data['emotion'])))
    
    for i, row in enumerate(data.index):
        image = np.fromstring(data.loc[row, 'pixels'], dtype=int, sep=' ')
        image = np.reshape(image, (48, 48, 1))
        image_array[i] = image
        
    return image_array, image_label



os.chdir('/home/bmllab/bml_pjh/face_data')
data=pd.read_csv('fer2013.csv')

data.drop(data.loc[data['emotion']==2].index,axis=0,inplace=True)

for i in range(len(data['Usage'])):
    if data.iloc[i,0]==1:
        data.iloc[i,0]=0
    elif data.iloc[i,0]==3:
        data.iloc[i,0]=1
    elif data.iloc[i,0]==4:
        data.iloc[i,0]=2
    elif data.iloc[i,0]==5:
        data.iloc[i,0]=3
    elif data.iloc[i,0]==6:
        data.iloc[i,0]=4



train_imgs, train_lbls = parse_data(data[data["Usage"] == "Training"])
val_imgs, val_lbls = parse_data(data[data["Usage"] == "PrivateTest"])
test_imgs, test_lbls = parse_data(data[data["Usage"] == "PublicTest"])
print(train_imgs.shape)
print(train_lbls.shape)

train_imgs=list(train_imgs)
train_lbls=list(train_lbls)
for i in range(len(data[data["Usage"] == "Training"])):
    train_imgs.append(rotate_90(train_imgs[i]))
    train_imgs.append(rotate_270(train_imgs[i]))
    train_lbls.append(train_lbls[i])
    train_lbls.append(train_lbls[i])
    
train_imgs=np.array(train_imgs)
train_lbls=np.array(train_lbls)

train_x=train_imgs/255.0
test_x=test_imgs/255.0
val_x=val_imgs/255
def build_last(input_shape,classes):
    inputs=layers.Input(shape=input_shape)
    
    x=layers.Conv2D(64,(3,3),padding='same')(inputs)
    x=layers.BatchNormalization()(x)
    x=layers.Conv2D(64,(3,3),padding='same')(x)
    x=layers.BatchNormalization()(x)
    x=layers.Conv2D(64,(3,3),padding='same')(x)
    x=layers.BatchNormalization()(x)
    x=layers.ReLU()(x)
   
    x=layers.Conv2D(64,(3,3),padding='same')(x)
    x=layers.BatchNormalization()(x)
    x=layers.Conv2D(64,(3,3),padding='same')(x)
    x=layers.BatchNormalization()(x)
    x=layers.Conv2D(64,(3,3),padding='same')(x)
    x=layers.BatchNormalization()(x)
    x=layers.ReLU()(x)

    
    x__=layers.Conv2D(64,(3,3),padding='same')(inputs)
    x__=layers.BatchNormalization()(x__)
    x__=layers.ReLU()(x__)
    x_=tf.add(x,x__)
    x_=layers.Dropout(0.5)(x_)

   
    
    x=layers.Conv2D(128,(3,3),strides=(2,2))(x_)
    x=layers.BatchNormalization()(x)
    x=layers.Conv2D(128,(3,3),padding='same')(x)
    x=layers.BatchNormalization()(x)
    x=layers.ReLU()(x)
    
   
    x=layers.Conv2D(128,(3,3),padding='same')(x)
    x=layers.BatchNormalization()(x)
    x=layers.Conv2D(128,(3,3),padding='same')(x)
    x=layers.BatchNormalization()(x)
    x=layers.ReLU()(x)    
    
    x_=layers.Conv2D(128,(3,3),strides=(2,2))(x_)
    x_=layers.BatchNormalization()(x_)
    x_=layers.ReLU()(x_)
    x_=tf.add(x,x_)
    x_=layers.Dropout(0.5)(x_)
    
    x=layers.Conv2D(256,(3,3),strides=(2,2))(x_)
    x=layers.BatchNormalization()(x)
    x=layers.ReLU()(x)    
    
    x=layers.ZeroPadding2D()(x)    
    x=layers.Conv2D(256,(3,3))(x)
    x=layers.BatchNormalization()(x)
    x=layers.ReLU()(x)   
    
    x_=layers.Conv2D(256,(3,3),strides=(2,2))(x_)
    x_=layers.BatchNormalization()(x_)
    x_=layers.ReLU()(x_)
    x_=tf.add(x,x_)
    
    x=layers.Conv2D(256,(3,3),strides=(2,2))(x_)
    x=layers.BatchNormalization()(x)
    x=layers.ReLU()(x)    
    
    x=layers.ZeroPadding2D()(x)    
    x=layers.Conv2D(256,(3,3))(x)
    x=layers.BatchNormalization()(x)
    x=layers.ReLU()(x)   
    
    x_=layers.Conv2D(256,(3,3),strides=(2,2))(x_)
    x_=layers.BatchNormalization()(x_)
    x_=layers.ReLU()(x_)
    x_=tf.add(x,x_)
 
    
    x=layers.GlobalAveragePooling2D()(x_)
    
    pred=layers.Dense(classes,activation="softmax")(x)
    
    model=tf.keras.Model(inputs=inputs,outputs=pred)
    
    return model



def build_last2(input_shape,classes):
    inputs=layers.Input(shape=input_shape)
    
    x=layers.Conv2D(64,(3,3),padding='same')(inputs)
    x=layers.BatchNormalization()(x)
    x=layers.Conv2D(64,(3,3),padding='same')(x)
    x=layers.BatchNormalization()(x)
    x=layers.Conv2D(64,(3,3),padding='same')(x)
    x=layers.BatchNormalization()(x)
    x=layers.ReLU()(x)
   
    x=layers.Conv2D(64,(3,3),padding='same')(x)
    x=layers.BatchNormalization()(x)
    x=layers.Conv2D(64,(3,3),padding='same')(x)
    x=layers.BatchNormalization()(x)
    x=layers.Conv2D(64,(3,3),padding='same')(x)
    x=layers.BatchNormalization()(x)
    x=layers.ReLU()(x)

    
    x__=layers.Conv2D(64,(3,3),padding='same')(inputs)
    x__=layers.BatchNormalization()(x__)
    x__=layers.ReLU()(x__)
    x_=tf.add(x,x__)
    x_=layers.Dropout(0.5)(x_)

   
    
    x=layers.Conv2D(128,(3,3),strides=(2,2))(x_)
    x=layers.BatchNormalization()(x)
    x=layers.Conv2D(128,(3,3),padding='same')(x)
    x=layers.BatchNormalization()(x)
    x=layers.ReLU()(x)
    
   
    x=layers.Conv2D(128,(3,3),padding='same')(x)
    x=layers.BatchNormalization()(x)
    x=layers.Conv2D(128,(3,3),padding='same')(x)
    x=layers.BatchNormalization()(x)
    x=layers.ReLU()(x)    
    
    x_=layers.Conv2D(128,(3,3),strides=(2,2))(x_)
    x_=layers.BatchNormalization()(x_)
    x_=layers.ReLU()(x_)
    x_=tf.add(x,x_)
    x_=layers.Dropout(0.5)(x_)
    
    x=layers.Conv2D(256,(3,3),strides=(2,2))(x_)
    x=layers.BatchNormalization()(x)
    x=layers.ReLU()(x)    
    
    x=layers.ZeroPadding2D()(x)    
    x=layers.Conv2D(256,(3,3))(x)
    x=layers.BatchNormalization()(x)
    x=layers.ReLU()(x)   
    
    x_=layers.Conv2D(256,(3,3),strides=(2,2))(x_)
    x_=layers.BatchNormalization()(x_)
    x_=layers.ReLU()(x_)
    x_=tf.add(x,x_)
    x_=layers.Dropout(0.5)(x_)
    
    x=layers.Conv2D(256,(3,3),strides=(2,2))(x_)
    x=layers.BatchNormalization()(x)
    x=layers.ReLU()(x)    
    
    x=layers.ZeroPadding2D()(x)    
    x=layers.Conv2D(256,(3,3))(x)
    x=layers.BatchNormalization()(x)
    x=layers.ReLU()(x)   
    
    x_=layers.Conv2D(256,(3,3),strides=(2,2))(x_)
    x_=layers.BatchNormalization()(x_)
    x_=layers.ReLU()(x_)
    x_=tf.add(x,x_)
 
    
    x=layers.GlobalAveragePooling2D()(x_)
    
    pred=layers.Dense(classes,activation="softmax")(x)
    
    model=tf.keras.Model(inputs=inputs,outputs=pred)
    
    return model



class_names=['angry','happy','sad','surprise','neutral']
print("model build start")
model=build_last(input_shape=(48,48,1), classes=len(class_names))

print("model build complete")
optimizer=optimizers.Adam()
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

batch_size=80
epochs=20
for i in range(epochs):
    model.fit(train_x,train_lbls,batch_size=batch_size,epochs=1,validation_data=(val_x,val_lbls))

    os.chdir("/home/bmllab/bml_pjh/face_model")
    test_loss,test_acc=model.evaluate(test_x,test_lbls)

    print("test_acc: %f"%test_acc)

    model.save("face_emotion_model.h5"%i)


















