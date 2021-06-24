# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 07:05:55 2021

@author: sunil
"""


# Importing libraries

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Data Preprocessing

train_datagenerator = ImageDataGenerator(rescale = 1/255,
                                        shear_range = 0.2,
                                       zoom_range = 0.2,
                                       rotation_range=40,
                                       width_shift_range=0.2,
                                       horizontal_flip = True
                                       )

train_set = train_datagenerator.flow_from_directory("../input/american-sign-language-recognition/training_set",
                                                    target_size = (64,64),
                                                    batch_size = 32,color_mode="grayscale"
                                                    )



test_datagenerator = ImageDataGenerator(rescale = 1/255)

test_set = test_datagenerator.flow_from_directory("../input/american-sign-language-recognition/test_set",
                                                 target_size = (64,64),
                                                 batch_size = 32,color_mode="grayscale"
                                                 )





# CNN model

model = keras.Sequential([
    keras.layers.Conv2D(32, (3,3), input_shape = (64,64,1), activation = 'relu'),
    keras.layers.MaxPool2D((2,2)),
    keras.layers.Dropout(0.2),
                        
    keras.layers.Conv2D(64, (3,3), activation = 'relu'),
    keras.layers.MaxPool2D((2,2)),
    keras.layers.Dropout(0.2),
                        
    keras.layers.Conv2D(128, (3,3), activation = 'relu'),
    keras.layers.MaxPool2D((2,2)),
    keras.layers.Dropout(0.2), 
    
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation = 'relu'),
    keras.layers.Dropout(0.2),
    
    keras.layers.Dense(512, activation = 'relu'),
    keras.layers.Dropout(0.2),
                        
    keras.layers.Dense(40, activation = 'softmax')
                        
])


model.compile(
    optimizer = 'adam',
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)


model.fit(train_set,epochs =10, validation_data = test_set)


# Saving model

model.save("aslr.h5")

