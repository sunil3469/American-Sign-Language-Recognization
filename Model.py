# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 07:05:55 2021

@author: sunil
"""


# Importing libraries

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator   # ImageDataFenerator is used to read images from folders

# Data Preprocessing


# Data Augmentation on train images

train_datagenerator = ImageDataGenerator(rescale = 1/255,        # rescaling values between 0 and 1
                                        shear_range = 0.2,       # distirting image along one axis
                                        zoom_range = 0.2,        # zooming image by 0.2
                                        rotation_range=40,       # rotating image
                                        width_shift_range=0.2,   # streching image
                                        horizontal_flip = True   # random horizontal flip
                                       )

train_set = train_datagenerator.flow_from_directory("../input/american-sign-language-recognition/training_set",
                                                    target_size = (64,64),                # setting target size to 64x64
                                                    batch_size = 32,
                                                    color_mode="grayscale"     # grayscale for B&W image
                                                    )

# Data Augmentation on test images

test_datagenerator = ImageDataGenerator(rescale = 1/255)    # rescaling values between 0 and 1

test_set = test_datagenerator.flow_from_directory("../input/american-sign-language-recognition/test_set",
                                                 target_size = (64,64),      # setting target size to 64x64
                                                 batch_size = 32,
                                                 color_mode="grayscale"   # grayscale for B&W image
                                                 )





# CNN model

model = keras.Sequential([   # Keras Sequential model
    
    #convolution layers
    
    keras.layers.Conv2D(32, (3,3), input_shape = (64,64,1), activation = 'relu'),  # filter of size 32 with kernel size (3,3)
    keras.layers.MaxPool2D((2,2)),   # Max Pooling Layer with size (2,2)
    keras.layers.Dropout(0.2),   # Dropuout layer to avoid overfitting
                        
    keras.layers.Conv2D(64, (3,3), activation = 'relu'),   # filter of size 64 with kernel size (3,3)
    keras.layers.MaxPool2D((2,2)),   # Max Pooling Layer with size (2,2)
    keras.layers.Dropout(0.2),  # Dropuout layer to avoid overfitting
                        
    keras.layers.Conv2D(128, (3,3), activation = 'relu'),  # filter of size 128 with kernel size (3,3)
    keras.layers.MaxPool2D((2,2)),   # Max Pooling Layer with size (2,2)
    keras.layers.Dropout(0.2),   # Dropuout layer to avoid overfitting
    
    keras.layers.Flatten(),   # Flatten layer to convert input to 1 dimentional vetor
    keras.layers.Dense(128, activation = 'relu'),   # Dense layer with 128 perceptrons
    keras.layers.Dropout(0.2),   # Dropuout layer to avoid overfitting
    
    keras.layers.Dense(512, activation = 'relu'),   # Dense layer with 512 perceptrons
    keras.layers.Dropout(0.2),   # Dropuout layer to avoid overfitting
                        
    keras.layers.Dense(40, activation = 'softmax')   # Dense layer with 40 perceptrons, because we have 40 classes to predict
                        
])

# Compiling model
model.compile(
    optimizer = 'adam',     # 'adam' optimizer
    loss = 'categorical_crossentropy',   # categorical crossentropy loss
    metrics = ['accuracy']    # accuracy metrics as performance measure
)

# fitting model
model.fit(train_set,epochs =10, validation_data = test_set)


# Saving model

model.save("aslr.h5")
