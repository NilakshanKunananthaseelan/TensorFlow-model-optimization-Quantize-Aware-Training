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

import quant8
import quant4

import h5py

from datetime import datetime


 




# ### Change numBits to select either 8 bit or 4 bit quantization

# In 8 bit quantization the activations are 8 bit and the weights 5 bit
# In 4 bit quantization both the activations and weights are 4 bit.  In 4 bit quantization changing the dense layer to 8 bit and possibly the input layer will yield better results.

 

def data_generator(mode,data_dir,batch_size,val_split=0.,size=(224,224),augment=False,subset=None):

    if (mode=='train'):
      datagen_kwargs = dict(rescale=1./255, validation_split=val_split)
      dataflow_kwargs = dict(target_size=size, 
                           batch_size=batch_size,
                           interpolation="bilinear",
                           class_mode='categorical')
    else:
      datagen_kwargs = dict(rescale=1./255)
      dataflow_kwargs = dict(batch_size=batch_size,class_mode=None)


    


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





    data_path = os.path.join(data_dir,mode)
    if(mode=='train'):
      generator = datagen.flow_from_directory(data_path,
                                            subset=subset, 
                                            shuffle=True,
                                            **dataflow_kwargs)
    else:
      generator = datagen.flow_from_directory(data_path,

                                            shuffle=False,
                                            **dataflow_kwargs)
    return generator 

def save_model(model,path,name='resnet50_QAT'):
  model_name = os.path.join(path,name)+'.h5'
  model.save(model_name)



def load_model(path):

  if(h5py.is_hdf5(path)):
    print(path)
    model = tf.keras.models.load_model(path)
  else:
    model = ResNet50(weights=None)
    model.load_weights(path)

  return model

def lr_schedule(epoch,lr):
  """
  Returns a custom learning rate that decreases as epochs progress.
  """
  learning_rate = lr
  if epoch > 10:
    learning_rate *= 0.1
  if epoch > 20:
    learning_rate *= 0.1
  if epoch > 50:
    learning_rate *= 0.01

  # tf.summary.scalar('learning rate', data=learning_rate, step=epoch)
  return learning_rate
 
def apply_quantization(layer):
  if isinstance(layer,  Conv2D):
    return tfmot.quantization.keras.quantize_annotate_layer(layer,quant8.Conv2DQuantizeConfig())
  elif isinstance(layer,  Activation):
    return tfmot.quantization.keras.quantize_annotate_layer(layer,quant8.ActivationQuantizeConfig()) 
  elif isinstance(layer, Dense):
    return tfmot.quantization.keras.quantize_annotate_layer(layer,quant8.DenseQuantizeConfig())  
  return layer

def log_callback(path,args):
  log_callback = tf.keras.callbacks.CSVLogger(os.path.join(path,'train.log'), separator=',', append=False)

  return [log_callback]



def custom_callbacks(path,args):

  checkpoint_path = os.path.join(path,'model-{epoch:02d}-{val_accuracy:.2f}.ckpt')

  cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=path,
                                                  verbose=1,
                                                  save_best_only=True,
                                                  monitor='val_accuracy',
                                                  mode='max',
                                                  save_weights_only=True,
                                                  
                                                  )

  tb_callback = tf.keras.callbacks.TensorBoard(log_dir=path)
  lr_callback =  tf.keras.callbacks.LearningRateScheduler(lr_schedule)

  log_callback = tf.keras.callbacks.CSVLogger(os.path.join(path), separator=',', append=False)

  return [cp_callback,tb_callback,lr_callback]



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

def train(qat_model,train_generator,batch_size,validation_generator=None,
          epochs=10,learning_rate=0.001,sgd=True,ckpt_path=None,args=None):

 

  if sgd: #futture usage if Adam is need with full set
    optim =  tf.keras.optimizers.SGD(lr=learning_rate,momentum=0.9, nesterov=True)

  qat_model.compile(optimizer=optim,
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
   

  hist = qat_model.fit(train_generator,
                  batch_size=train_generator.batch_size, 
                  steps_per_epoch=train_generator.samples//batch_size, 
                  epochs=epochs,
                  validation_data = validation_generator,
                  validation_steps=validation_generator.samples//batch_size,
                  callbacks =custom_callbacks(ckpt_path,args),
                  verbose=1).history

  return hist
def evaluate(path,args=None):
  model = load_model(path)

  model.summary()

  test_generator = data_generator('val',args.data)
  test_generator.reset()

  results = model.evaluate(test_generator,verbose=1,
                              steps=test_generator.samples//args.batch_size,
                              return_dict=True,
                              callbacks=[log_callback(datetime.now().strftime('%Y-%m-%d_%H-%M-%S')+'.log')])
  # for k,v in results.items():
  #   print(k,'\t\t',v)

  print(results)

  #prediction

  """
   python resnet50_tf_QAT.py --data ~/work/data.imagenet_nilakshan/ -e --model-dir check/2020-09-28_07-08-50/saved_model/resnet50_QAT.h5 

  """

  #
  # preds = model.predict_generator(test_generator,verbose=1,
  #                             steps=test_generator.samples//args.batch_size)

  # pred_class = np.argmax(prediction,axis=1)

  # gt_class = test_generator.classes

  # filenames = test_generator.filenames






 
def quantize_aware_training(args):

  if not(args.evaluate):
  
    if args.pretrained  :
      fp_model = ResNet50()

    #Weight should load to QAT model
    # else:
    #   fp_model = ResNet50(weights=None)
    #   if(args.model_dir is None):
    #     raise Exception('Checkpoin path is not given')
    #   else:
    #     fp_model.load_weights(args.model_dir)

    fp_model.summary()

    qat_model = quantize_aware_model(fp_model,args.num_bits)

    qat_model.summary()
    mode = 'train'
   
    train_generator = data_generator(mode,
                                     args.data,
                                     args.batch_size,
                                     args.validation_split,
                                     subset='training',
                                     augment=True)

    #Note : validation data should be used from train data.
    #       This must be test generator for evaluation
    
    validation_generator = data_generator(mode,
                                          args.data,
                                          args.batch_size,
                                          args.validation_split,
                                          subset='validation')
    # validation_generator =None


     ##Need to find a good way
    current_train_dir = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    current_path = os.path.join(args.output_dir,current_train_dir)
    if not os.path.exists(current_path):
      os.makedirs(current_path)

    ckpt_path = os.path.join(current_path,'checkpoints')
    if not os.path.exists(ckpt_path):
      os.makedirs(ckpt_path)

    train(qat_model,train_generator,args.batch_size,validation_generator,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            ckpt_path=ckpt_path,
            args=args)

    if True:
      model_path = os.path.join(current_path,'saved_model')

      if not os.path.exists(model_path):
        os.makedirs(model_path)

      save_model(qat_model,model_path)



  else:
    evaluate(args.model_dir,args)

   

    
 
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

  parser.add_argument('-lr', '--learning-rate', default=0.001,dest='learning_rate',type=float,
                        metavar='LR', help='initial learning rate (default: 0.001)')

  parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='Use pretrained weights or trained checkpoint')

  parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on test set')

  parser.add_argument('--save_model', dest='save_model', action='store_true',
                        help='save model in proto file')

  parser.add_argument('--name', dest='name', action='store_true',
                        help='Name of the model')

  parser.add_argument('--seed', type=int, default=0,
                        help='seed the PRNG for CPU, CUDA, numpy, and Python')
  
  parser.add_argument('--out-dir', '-o', dest='output_dir', default='logs', help='Path to dump logs and checkpoints')
  parser.add_argument('--model-dir', dest='model_dir', default=None,help='Path to load checkpoints')

  parser.add_argument('--validation-split', '--valid-size', '--vs', dest='validation_split',
                          default=0,type=float,
                        help='Portion of training dataset to set aside for validation')
  

  # parser.add_argument('--confusion', dest='save_confusion', default=False, action='store_true',
  #                       help='save the confusion matrix')

  return parser

def main():
  args = init_arg_parser().parse_args()

  if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
 
  
 

  quantize_aware_training(args)



if __name__ =='__main__':

  main()


  """
  python resnet50_tf_QAT.py --data ~/work/data.imagenet_nilakshan/ --epochs 1 -b 32 -n 8 --pretrained --out-dir 'check' --vs 0.1

  """