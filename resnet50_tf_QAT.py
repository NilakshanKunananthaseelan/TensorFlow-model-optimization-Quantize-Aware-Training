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

import qunat8
import quant4

from datetime import datetime


 




# ### Change numBits to select either 8 bit or 4 bit quantization

# In 8 bit quantization the activations are 8 bit and the weights 5 bit
# In 4 bit quantization both the activations and weights are 4 bit.  In 4 bit quantization changing the dense layer to 8 bit and possibly the input layer will yield better results.

 
def data_generator(mode,data_dir,batch_size,val_split=0.,size=(224,224),augment=False):

    datagen_kwargs = dict(rescale=1./255, validation_split=val_split)
    dataflow_kwargs = dict(target_size=size, 
                           batch_size=batch_size,
                           interpolation="bilinear",
                           class_mode='categorical')


    if augment:
      datagen = ImageDataGenerator(rotation_range=20,
                                    horizontal_flip=True,
                                    vertical_flip=False,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    shear_range=0.2,
                                    **datagen_kwargs)

    else:
      datagen = ImageDataGenerator(**datagen_kwargs)




    generator = datagen.flow_from_directory(data_dir,
                                            subset=mode,
                                            shuffle=True,
                                            **dataflow_kwargs)
    return generator 

def apply_quantization(layer):
  if isinstance(layer,  Conv2D):
    return tfmot.quantization.keras.quantize_annotate_layer(layer,quant8.Conv2DQuantizeConfig())
  elif isinstance(layer,  Activation):
    return tfmot.quantization.keras.quantize_annotate_layer(layer,quant8.ActivationQuantizeConfig()) 
  elif isinstance(layer, Dense):
    return tfmot.quantization.keras.quantize_annotate_layer(layer,quant8.DenseQuantizeConfig())  
  return layer

def custom_callbacks(path):

  checkpoint_path = os.path.join('logs',current_train_dir,'checkpoints','model-{epoch:02d}-{val_accuracy:.2f}')
 


  cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                  verbose=1,
                                                  save_best_only=True,
                                                  monitor='val_accuracy',
                                                  mode='max',
                                                  save_weights_only=True
                                                  )

  tb_callback = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(path,'scalars'))
  return [cp_callback,tb_callback]



def quantize_aware_model(model,nbits=8):

  quantize_annotate_layer = tfmot.quantization.keras.quantize_annotate_layer
  quantize_annotate_model = tfmot.quantization.keras.quantize_annotate_model
  quantize_scope = tfmot.quantization.keras.quantize_scope


  if(nbits ==8):
    with quantize_scope(
      {'Conv2DQuantizeConfig':quant8.Conv2DQuantizeConfig,
      'ActivationQuantizeConfig': quant8.ActivationQuantizeConfig,
      'DenseQuantizeConfig': quant8.ActivationQuantizeConfig}):
 
       
        
       annotated_model = tf.keras.models.clone_model(
       model,clone_function= apply_quantization)
       quant_aware_model = tfmot.quantization.keras.quantize_apply(annotated_model)
        
  else:
      with quantize_scope(
        {'Conv2DQuantizeConfig':quant4.Conv2DQuantizeConfig,
        'ActivationQuantizeConfig': quant4.ActivationQuantizeConfig,
        'DenseQuantizeConfig': quant4.ActivationQuantizeConfig}):
   
        
      
         annotated_model = tf.keras.models.clone_model(
         model,clone_function= apply_quantization)
         quant_aware_model = tfmot.quantization.keras.quantize_apply(annotated_model)
      
  return quant_aware_model

def train(qat_model,train_generator,batch_size,validation_generator=None,epochs=10,learning_rate=0.001):

  optim =  tf.keras.optimizers.SGD(lr=learning_rate,momentum=0.9, nesterov=True)

  qat_model.compile(optimizer=optim,
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

  hist = quant_aware_model.fit(train_generator,
                  batch_size=train_generator.batch_size, 
                  steps_per_epoch=train_generator.samples//batch_size, 
                  epochs=epochs,
                  validation_data = validation_generator,
                  validation_steps=validation_generator.samples//batch_size,
                  callbacks =custom_callbacks(),
                  verbose=1).history

  return hist






 


 
def quantize_aware_training(args):
  

  fp_model = ResNet50()

  fp_model.summary()

  qat_model = quant_aware_model(model,args.n)
  mode = 'train'
 
  train_generator = data_generator(mode,
                                   args.data,
                                   args.batch_size,
                                   args.validation_split,
                                   augment=True):

  #for checking
  mode = 'val'
  validation_generator = data_generator(mode,args.data,args.batch_size)

  train(qat_model,train_generator,batch_size,validation_generator,
        epochs=args.epochs,learning_rate=arg.learn)










 





 
def init_arg_parser():
  """
  Common QAT-CLI arguments
  """
  parser = argparse.ArgumentParser("ResNet50 classifier model QAT")

  parser.add_argument('--data',metavar='DATASET_DIR', help='path to dataset')

  parser.add_argument('--epochs', type=int, metavar='N', default=20,
                        help='number of total epochs to run (default: 20')
  parser.add_argument('-b', '--batch-size', dest='batch_size',default=32, type=int,
                        metavar='B', help='mini-batch size (default: 32)')

  parser.add_argument('-n', '--num-bits', dest='num_bits',default=8, type=int,
                        metavar='N', help='activation bit size (default: 8)')

  parser.add_argument('-lr', '--learning-rate', default=0.001,dest='learning_rate' type=float,
                        metavar='LR', help='initial learning rate (default: 0.001)')



  parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on test set')

  parser.add_argument('--seed', type=int, default=0,
                        help='seed the PRNG for CPU, CUDA, numpy, and Python')
  
  parser.add_argument('--out-dir', '-o', dest='output_dir', default='logs', help='Path to dump logs and checkpoints')
  parser.add_argument('--validation-split', '--valid-size', '--vs', dest='validation_split',
                        type=float_range(exc_max=True), default=0.1,
                        help='Portion of training dataset to set aside for validation')
  

  # parser.add_argument('--confusion', dest='save_confusion', default=False, action='store_true',
  #                       help='save the confusion matrix')

  return parser

def main():
  args = init_arg_parser().parse_args()

  os.makedirs(args.output_dir,exist_ok=True)
  current_train_dir = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
  os.makedirs(os.path.join(args.output_dir,current_train_dir),exist_ok=True)
 

  quantize_aware_training(args)



if __name__ =='__main__':

  main()