 
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from os import makedirs
from os.path import join, exists, expanduser
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing import image

from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from model_TF import Tf_ResNet50

if not os.path.exists('./output'):
        os.makedirs('./output')


data_path = '/home/ubuntu/work/data.imagenet_nilakshan/'

batch_size= 32
do_train=False



def lr_schedule(epoch,lr):
      
    learning_rate = lr
    if epoch > 10:
        learning_rate *= 0.1
    if epoch > 20:
        learning_rate *= 0.1
    if epoch > 50:
        learning_rate *= 0.01

     
    return learning_rate

def custom_callbacks(path):

    checkpoint_path = os.path.join(path,'model-{epoch:02d}-{val_accuracy:.2f}')

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                  verbose=1,
                                                  save_best_only=True,
                                                  monitor='val_accuracy',
                                                  mode='max',
                                                  save_weights_only=False,

                                                  )

    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=path)
    lr_callback =  tf.keras.callbacks.LearningRateScheduler(lr_schedule)

    log_callback = tf.keras.callbacks.CSVLogger(os.path.join(path,'train.log'), separator=',', append=False)

    return [cp_callback,tb_callback,lr_callback,log_callback]


def train(model,train_generator,validation_generator,batch_size=32,
          epochs=2,learning_rate=0.001,sgd=True,ckpt_path=None,args=None):

 

    if sgd: #futture usage if Adam is need with full set
        optim =  tf.keras.optimizers.SGD(lr=learning_rate,momentum=0.9, nesterov=True)
	
    model.compile(optimizer=optim,
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
 
    model.fit(train_generator,epochs=epochs,batch_size=train_generator.batch_size,
        steps_per_epoch=train_generator.samples//batch_size, 


        validation_data=validation_generator,
        validation_steps=validation_generator.samples//batch_size,
        callbacks =custom_callbacks(ckpt_path),verbose=1)


    if True:
        model_path = os.path.join(ckpt_path,'saved_model')

        if not os.path.exists(model_path):
            os.makedirs(model_path)

        model_name = os.path.join(model_path,'resnet50')+'.h5'
        model.save(model_name)


 

if do_train:

    resnet=ResNet50(weights='imagenet') #To use pretrained model
    # resnet = Tf_ResNet50._resnet50(input_shape = (224, 224, 3), classes = 1000) #scratchmodel
     
    #resnet = tf.keras.models.load_model('out/model-05-0.63/') #to resume
    for layer in resnet.layers:
        layer.use_bias=True 
    _bias=[l.use_bias for l in resnet.layers]
    

    train_datagen_kwargs = dict(rescale=1./255, validation_split=0.3)
    train_dataflow_kwargs = dict(target_size=(224,224), 
                               batch_size=batch_size,
                               interpolation="bilinear",
                               class_mode='categorical')

    train_datagen = ImageDataGenerator(rotation_range=20,
                                        horizontal_flip=True,
                                        vertical_flip=False,
                                        width_shift_range=0.2,
                                        height_shift_range=0.2,
                                        shear_range=0.2,
                                        **train_datagen_kwargs)
    train_generator = train_datagen.flow_from_directory(data_path+'train',
                                                subset='training', 
                                                shuffle=True,
                                                **train_dataflow_kwargs)


    val_datagen = ImageDataGenerator(**train_datagen_kwargs)

    val_generator = val_datagen.flow_from_directory(data_path+'train',
                                                    subset='validation',
                                                    shuffle=False,
                                                    **train_dataflow_kwargs)


    # In[ ]:


    train(resnet,train_generator,val_generator,batch_size=batch_size,epochs=5,learning_rate=0.001,ckpt_path='./output')


else:
    datagen_kwargs = dict(rescale=1./255,validation_split=0)#for testing
    dataflow_kwargs = dict(target_size=(224,224), batch_size=batch_size,
                                class_mode='categorical')

    test_datagen = ImageDataGenerator(**datagen_kwargs)

    test_generator = test_datagen.flow_from_directory(data_path+'val',
                                                shuffle=False,
                                       #         subset='validation',
                                            **dataflow_kwargs)

    #loaded_model = tf.keras.models.load_model('out_nobias/saved_model/resnet50.h5')
    loaded_model=tf.keras.models.load_model('output/saved_model/resnet50.h5')
    for layer in loaded_model.layers:
      layer.use_bias=True
 
   
    with open('report.txt','w') as fh:
        loaded_model.summary(print_fn=lambda x:fh.write(x+'\n'))
     
    eval= loaded_model.evaluate(test_generator,verbose=1,
                             steps=test_generator.samples//batch_size,
                             return_dict=True)
    print(eval)
    # pred = loaded_model.predict(test_generator,batch_size=32,verbose=1,steps=test_generator.samples//b)
    # print([np.argmax(i)for i in pred])
    # for data in test_generator:
    #       print([np.argmax(i) for i in data[1]])
    #       break
    # exit()
    # for ele in test_generator:
    #         print(ele[0],'\n',ele[0].shape,'\n',ele[0][0][:].T.shape,'\n',ele[0].T)
    #         break





#{'loss': 2.7669942378997803, 'accuracy': 0.4643999934196472}

