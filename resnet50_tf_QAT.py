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

import os
import argparse
import numpy as np
import scipy.misc
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import layers,regularizers
from tensorflow.keras.layers import Input,Dense,Activation,Flatten,Conv2D,MaxPooling2D,GlobalMaxPooling2D,ZeroPadding2D
from tensorflow.keras.layers import AveragePooling2D,GlobalAveragePooling2D,BatchNormalization,Softmax
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing import image
from tensorflow.python.keras.utils import layer_utils
from tensorflow.keras.utils import get_source_inputs
from tensorflow.python.keras.utils.data_utils import get_file
from tensorflow.python.keras.applications.imagenet_utils import decode_predictions,preprocess_input,obtain_input_shape
from tensorflow.keras import models
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow_model_optimization as tfmot
from tensorflow.keras.applications import ResNet50

 




# ### Change numBits to select either 8 bit or 4 bit quantization

# In 8 bit quantization the activations are 8 bit and the weights 5 bit
# In 4 bit quantization both the activations and weights are 4 bit.  In 4 bit quantization changing the dense layer to 8 bit and possibly the input layer will yield better results.

 

numBits = 8
if(numBits ==8):
   import quant8
else:
   import quant4

 


batch_size=32
train_datagen = ImageDataGenerator(
                                     horizontal_flip=True,
                                     vertical_flip=False,
                                     rotation_range=20,
                                     preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
        '/home/ubuntu/work/data.imagenet_nilakshan/train', 
        target_size=(224, 224), 
        batch_size=batch_size,
        shuffle=True,
        class_mode='categorical'
        )


# In[4]:


validation_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
validation_generator = validation_datagen.flow_from_directory(
        '/home/ubuntu/work/data.imagenet_nilakshan/val',  
       
        target_size=(224, 224),
        batch_size=batch_size,
        shuffle=True,
        class_mode='categorical'
        )


NUM_EPOCHS = 5
INIT_LR = 1e-4


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
          if isinstance(layer,  Conv2D):
            return tfmot.quantization.keras.quantize_annotate_layer(layer,quant8.Conv2DQuantizeConfig())
          elif isinstance(layer,  Activation):
            return tfmot.quantization.keras.quantize_annotate_layer(layer,quant8.ActivationQuantizeConfig()) 
          elif isinstance(layer, Dense):
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
          if isinstance(layer,  Conv2D):
            return tfmot.quantization.keras.quantize_annotate_layer(layer,quant4.Conv2DQuantizeConfig())
          elif isinstance(layer,  Activation):
            return tfmot.quantization.keras.quantize_annotate_layer(layer,quant4.ActivationQuantizeConfig()) 
          elif isinstance(layer,  Dense):
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
os.makedirs(os.path.join('logs',current_train_dir,'checkpoints'),exist_ok=True)
checkpoint_path = os.path.join('logs',current_train_dir,'checkpoints','model-{epoch:02d}-{val_accuracy:.2f}')
 
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

os.makedirs(os.path.join('logs',current_train_dir,'saved_model'),exist_ok=True)
save_path = os.path.join('logs',current_train_dir,'saved_model')

model.save(os.path.join(save_path,'model-{epoch:02d}-{val_accuracy:.2f}.h5'))

def init_arg_parser():
  """
  Common QAT-CLI arguments
  """
  parser = argparse.ArgumentParser("ResNet50 classifier model QAT")

  parser.add_argument('--data',metavar='DATASET_DIR', help='path to dataset')

  parser.add_argument('--epochs', type=int, metavar='N', default=20,
                        help='number of total epochs to run (default: 20')
  parser.add_argument('-b', '--batch-size', default=326, type=int,
                        metavar='N', help='mini-batch size (default: 32)')

  parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on test set')

  parser.add_argument('--seed', type=int, default=0,
                        help='seed the PRNG for CPU, CUDA, numpy, and Python')
  
  parser.add_argument('--out-dir', '-o', dest='output_dir', default='logs', help='Path to dump logs and checkpoints')
  parser.add_argument('--validation-split', '--valid-size', '--vs', dest='validation_split',
                        type=float_range(exc_max=True), default=0.1,
                        help='Portion of training dataset to set aside for validation')
  



if __name__ =='__main__':

  main()