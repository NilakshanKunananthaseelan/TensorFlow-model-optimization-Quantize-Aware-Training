import tensorflow_model_optimization as tfmot
LastValueQuantizer = tfmot.quantization.keras.quantizers.LastValueQuantizer
MovingAverageQuantizer = tfmot.quantization.keras.quantizers.MovingAverageQuantizer


class DefaultConv2DQuantizeConfig(tfmot.quantization.keras.QuantizeConfig):
   
    def get_weights_and_quantizers(self, layer):
      return [(layer.kernel, LastValueQuantizer(num_bits=8, symmetric=True, narrow_range=False, per_axis=False))]

    def get_activations_and_quantizers(self, layer):
      return [(layer.activation, MovingAverageQuantizer(num_bits=8, symmetric=False, narrow_range=False, per_axis=False))]

    def set_quantize_weights(self, layer, quantize_weights):
      layer.kernel = quantize_weights[0]

    def set_quantize_activations(self, layer, quantize_activations):
      layer.activation = quantize_activations[0]
    
    def get_output_quantizers(self, layer):
      return []

    def get_config(self):
      return {}
      

class Conv2DQuantizeConfig(DefaultConv2DQuantizeConfig):
    # Change number of bits as needed
    def get_weights_and_quantizers(self, layer):
      return [(layer.kernel, LastValueQuantizer(num_bits=4, symmetric=True, narrow_range=False, per_axis=False))]



#Activations in Convolution Layers

class DefaultActivationQuantizeConfig(tfmot.quantization.keras.QuantizeConfig):
    
    def get_activations_and_quantizers(self, layer):
      return [(layer.activation, MovingAverageQuantizer(num_bits=8, symmetric=False, narrow_range=False, per_axis=False))] 

    def set_quantize_activations(self, layer, quantize_activations):
      layer.activation = quantize_activations[0]
    
    def get_output_quantizers(self, layer):
      return []

    def get_config(self):
      return {}


      

class ActivationQuantizeConfig(DefaultActivationQuantizeConfig):
    #def __init__(self,numBits):
    #  super().__init__(tfmot.quantization.keras.QuantizeConfig)
    #  self.numBits=numBits
    
    def get_weights_and_quantizers(self, layer):
      return []
    def set_quantize_weights(self, layer, quantize_weights):
      return
    
    #Change number of bits as needed. Here same as default                                      
    def get_activations_and_quantizers (self, layer):
      return [(layer.activation, MovingAverageQuantizer(num_bits=4, symmetric=True, narrow_range=False, per_axis=False))]

                                                
#Dense Layers

class DefaultDenseQuantizeConfig(tfmot.quantization.keras.QuantizeConfig):
   
    def get_weights_and_quantizers(self, layer):
      return [(layer.kernel, LastValueQuantizer(num_bits=8, symmetric=True, narrow_range=False, per_axis=False))]

    def get_activations_and_quantizers(self, layer):
      return [(layer.activation, MovingAverageQuantizer(num_bits=8, symmetric=False, narrow_range=False, per_axis=False))]

    def set_quantize_weights(self, layer, quantize_weights):
      layer.kernel = quantize_weights[0]

    def set_quantize_activations(self, layer, quantize_activations):
      layer.activation = quantize_activations[0]
    
    def get_output_quantizers(self, layer):
      return []

    def get_config(self):
      return {}
      

class DenseQuantizeConfig(DefaultConv2DQuantizeConfig):

    #Change number of bits as needed
    def get_weights_and_quantizers(self, layer):
      return [(layer.kernel, LastValueQuantizer(num_bits=4, symmetric=True, narrow_range=False, per_axis=False))]
    #Change number of bits as needed. Here same as default
    def get_activations_and_quantizers(self, layer):
      return [(layer.activation, MovingAverageQuantizer(num_bits=4, symmetric=False, narrow_range=False, per_axis=False))]