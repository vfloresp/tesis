from os import name
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D

class Autoencoder:
    """
        
    """
    
    def __init__(self, 
                 input_shape,
                 conv_filters,
                 conv_kernels,
                 conv_strides,
                 latent_space_dim):
        self.input_shape = input_shape #[28,28,1]
        self.conv_filters = conv_filters #[2,4,8]
        self.conv_kernels = conv_kernels #[3,5,3]
        self.conv_strides = conv_strides #[1,2,2]
        self.latent_space_dim = latent_space_dim #2
        
        self.encoder = None
        self.decoder = None
        self.model = None
        
        self._num_conv_layers = len(conv_filters)
        
        self._build()
        
    def _build(self):
        self._build_encoder()
        self._build_dencoder()
        self._build_autoencoder()
        
    def _build_encoder(self):
        encoder_input = self._add_encoder_input()
        conv_layers = self._add_conv_layers(encoder_input)
        bottleneck = self._add_bottleneck(conv_layers)
        self.encoder = Model(encoder_input, bottleneck, name="encoder")
        
    def _add_encoder_input(self):
        return Input(shape=self.input_shape, name="encoder_imput")
    
    def _add_conv_layers(self, encoder_input):
        """Create all convolutionals blocks in encoder

        Args:
            encoder_input ([type]): [description]
        """
        x = encoder_input
        for layer_index in range(self._num_conv_layers):
            x = self._add_conv_layers(layer_index,x)
        return x
        
    def _add_conv_layer(self, layer_index,x):
        """Adds a convolutional block to a graph of layers, consisting og
            conv 2d + ReLU + batch normalization
        Args:
            layer_index ([type]): [description]
            x ([type]): [description]
        """
        layer_number = layer_index + 1
        conv_layer = Conv2D(
            filters=self.conv_filters[layer_index],
            kernel_size=self.conv_kernels[layer_index],
            strides= self.conv_strides[layer_index],
            padding="same",
            name=f"encoder_conv_layer_{layer_number}"
        )
        x = conv_layer(x)