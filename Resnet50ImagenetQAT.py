#!/usr/bin/env python
# coding: utf-8

# ### Quantize Keras Resnet50V1 Model
# 
# #### A data set that is a subset of Imagenet that has only 10 classes is used for training and validation 

# A different training regime could yield different results from  that which is achieved in this file. Please install
# tensorflow and tensorflow-model-optimization as shown below
# 
# pip install  tf-nightly
# 
# pip install tensorflow-model-optimization
# 
# 
# The method used here has not been officially released and is experimental. It may
# never make it to a release.  Sometimes I have run into problems where a particular release
# does not work
# 

# In[1]:


import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Softmax
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing import image
import tensorflow.keras.backend as K
from tensorflow.python.keras.utils import layer_utils
from tensorflow.python.keras.utils.data_utils import get_file
from tensorflow.python.keras.applications.imagenet_utils import decode_predictions
from tensorflow.python.keras.applications.imagenet_utils import preprocess_input
from tensorflow.python.keras.applications.imagenet_utils import obtain_input_shape
from tensorflow.keras.utils import get_source_inputs
#from roundInf import *
from tensorflow.keras import models
import matplotlib.pyplot as plt
from tensorflow.keras.initializers import glorot_uniform
import scipy.misc
from tensorflow.keras.utils import to_categorical
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow_model_optimization as tfmot
from tensorflow.keras import regularizers
from tensorflow.keras.applications import ResNet50

# ### Change numBits to select either 8 bit or 4 bit quantization

# In 8 bit quantization the activations are 8 bit and the weights 5 bit
# In 4 bit quantization both the activations and weights are 4 bit.  In 4 bit quantization changing the dense layer to 8 bit and possibly the input layer will yield better results.

 

numBits = 8
if(numBits ==8):
   import quant8
else:
   import quant4

 


batch_size=8
train_datagen = ImageDataGenerator(
                                     horizontal_flip=True,
                                     vertical_flip=False,
                                     rotation_range=20,
                                     preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
        # '/home/ubuntu/work/data.imagenet/train', 
         
        target_size=(224, 224), 
        batch_size=batch_size,
        shuffle=True,
        class_mode='categorical'
        )


# In[4]:


validation_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
validation_generator = validation_datagen.flow_from_directory(
        # '/home/ubuntu/work/data.imagenet/val',  
       
        target_size=(224, 224),
        batch_size=batch_size,
        shuffle=True,
        class_mode='categorical'
        )


# In[5]:

print(train_generator,validation_generator)
NUM_EPOCHS = 5
INIT_LR = 1e-4
"""class LossAndErrorPrintingCallback(tf.keras.callbacks.Callback):
    
    def on_epoch_end(self, epoch, logs=None):
      print('The average loss for epoch {} is {:7.2f} and the accuracy is {:7.2f}.'.format(epoch, logs['loss'], logs['accuracy']))
      print('The average validation loss for epoch {} is {:7.2f} and the validation accuracy is {:7.2f}.'.format(epoch, logs['val_loss'], logs['val_accuracy']))

"""
# In[6]:


opt1 =  tf.keras.optimizers.SGD(lr=INIT_LR,momentum=0.9, nesterov=True)


 
model = ResNet50()

model.summary()

 


quantize_annotate_layer = tfmot.quantization.keras.quantize_annotate_layer
quantize_annotate_model = tfmot.quantization.keras.quantize_annotate_model
quantize_scope = tfmot.quantization.keras.quantize_scope
if(numBits ==8):
    with quantize_scope(
      {'Conv2DQuantizeConfig':quant8.Conv2DQuantizeConfig,
      'ActivationQuantizeConfig': quant8.ActivationQuantizeConfig,
      'DenseQuantizeConfig': quant8.ActivationQuantizeConfig}):
 
       def apply_quantization(layer):
          if isinstance(layer, tf.keras.layers.Conv2D):
            return tfmot.quantization.keras.quantize_annotate_layer(layer,quant8.Conv2DQuantizeConfig())
          elif isinstance(layer, tf.keras.layers.Activation):
            return tfmot.quantization.keras.quantize_annotate_layer(layer,quant8.ActivationQuantizeConfig()) 
          elif isinstance(layer, tf.keras.layers.Dense):
            return tfmot.quantization.keras.quantize_annotate_layer(layer,quant8.DenseQuantizeConfig())  
          return layer
        
       annotated_model = tf.keras.models.clone_model(
       model,clone_function= apply_quantization)
       quant_aware_model = tfmot.quantization.keras.quantize_apply(annotated_model)
       quant_aware_model.summary()
else:
    with quantize_scope(
      {'Conv2DQuantizeConfig':quant4.Conv2DQuantizeConfig,
      'ActivationQuantizeConfig': quant4.ActivationQuantizeConfig,
      'DenseQuantizeConfig': quant4.ActivationQuantizeConfig}):
 
       def apply_quantization(layer):
          if isinstance(layer, tf.keras.layers.Conv2D):
            return tfmot.quantization.keras.quantize_annotate_layer(layer,quant4.Conv2DQuantizeConfig())
          elif isinstance(layer, tf.keras.layers.Activation):
            return tfmot.quantization.keras.quantize_annotate_layer(layer,quant4.ActivationQuantizeConfig()) 
          elif isinstance(layer, tf.keras.layers.Dense):
            return tfmot.quantization.keras.quantize_annotate_layer(layer,quant4.DenseQuantizeConfig())  
          return layer
    
       annotated_model = tf.keras.models.clone_model(
       model,clone_function= apply_quantization)
       quant_aware_model = tfmot.quantization.keras.quantize_apply(annotated_model)
       quant_aware_model.summary()


# In[10]:


opt1 =  tf.keras.optimizers.SGD(lr=INIT_LR,momentum=0.9, nesterov=True)
quant_aware_model.compile(optimizer=opt1,
             loss='categorical_crossentropy',
             metrics=['accuracy'])



from datetime import datetime
os.makedirs('logs',exist_ok=True)
current_train_dir = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
checkpoint_path = os.path.join('logs',current_train_dir,'checkpoints','model-{epoch:02d}-{val_accuracy:.2f}.hdf5')
 
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                  verbose=1,
                                                  save_best_only=True,
                                                  monitor='val_accuracy',
                                                  mode='max',
                                                  save_weights_only=True
                                                  )


quant_aware_model.fit(train_generator,
                  batch_size=batch_size, 
                  steps_per_epoch=40000//batch_size, 
                  epochs=NUM_EPOCHS,
                  validation_data = validation_generator,
                  validation_steps=10000//batch_size,
                  callbacks =[cp_callback],
                  verbose=1)


save_path = os.path.join('logs',current_train_dir,'saved_model')

model.save(os.path.join(save_path,'model-{epoch:02d}-{val_accuracy:.2f}.h5'))
