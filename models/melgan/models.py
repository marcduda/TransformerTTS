import tensorflow as tf

from models.melgan.layers import PaddedWNConv1D, Upscale1D, ResidualStack, DiscriminatorBlock


class Generator(tf.keras.models.Model):
    def __init__(self, mel_channels: int, leaky_alpha: float = .2, debug: bool = False, **kwargs):
        super(Generator, self).__init__(**kwargs)
        self.model_layers = []
        self.model_layers += [PaddedWNConv1D(channels=512, kernel_size=7, dilation=1)]
        self.model_layers += [tf.keras.layers.LeakyReLU(alpha=leaky_alpha)]
        self.model_layers += [Upscale1D(output_channels=256, scale=8, kernel_size=16)]
        self.model_layers += [ResidualStack(n_layers=3, channels=256)]
        self.model_layers += [tf.keras.layers.LeakyReLU(alpha=leaky_alpha)]
        self.model_layers += [Upscale1D(output_channels=128, scale=8, kernel_size=16)]
        self.model_layers += [ResidualStack(n_layers=3, channels=128)]
        self.model_layers += [tf.keras.layers.LeakyReLU(alpha=leaky_alpha)]
        self.model_layers += [Upscale1D(output_channels=64, scale=2, kernel_size=4)]
        self.model_layers += [ResidualStack(n_layers=3, channels=64)]
        self.model_layers += [tf.keras.layers.LeakyReLU(alpha=leaky_alpha)]
        self.model_layers += [Upscale1D(output_channels=32, scale=2, kernel_size=4)]
        self.model_layers += [ResidualStack(n_layers=3, channels=32)]
        self.model_layers += [tf.keras.layers.LeakyReLU(alpha=leaky_alpha)]
        self.model_layers += [PaddedWNConv1D(channels=1, kernel_size=7, dilation=1)]
        self.model_layers += [tf.keras.activations.tanh()]
        input_signature = [tf.TensorSpec(shape=[None, None, mel_channels], dtype=tf.float32)]
        self.debug = debug
        self.call = self._apply_signature(self._call, signature=input_signature)
    
    def _apply_signature(self, function, signature):
        if self.debug:
            return function
        else:
            return tf.function(input_signature=signature)(function)
    
    def _call(self, x):
        for layer in self.model_layers:
            x = layer(x)
        return x


class MultiScaleDiscriminator(tf.keras.models.Model):
    def __init__(self, debug: bool = False, **kwargs):
        super(MultiScaleDiscriminator, self).__init__(**kwargs)
        self.pooling1 = tf.keras.layers.AvgPool1D(pool_size=4, strides=2, padding='valid')
        self.pooling2 = tf.keras.layers.AvgPool1D(pool_size=4, strides=2, padding='valid')
        self.d1 = DiscriminatorBlock()
        self.d2 = DiscriminatorBlock()
        self.d3 = DiscriminatorBlock()
        input_signature = [tf.TensorSpec(shape=[None, None, 1], dtype=tf.float32)]
        self.debug = debug
        self.call = self._apply_signature(self._call, signature=input_signature)
    
    def _apply_signature(self, function, signature):
        if self.debug:
            return function
        else:
            return tf.function(input_signature=signature)(function)
        
    def _call(self, x):
        scaled1 = self.pooling1(x)
        scaled2 = self.pooling2(x)
        out1, feats1 = self.d1(x)
        out2, feats2 = self.d2(scaled1)
        out3, feats3 = self.d1(scaled2)
        return [out1, out2, out3], [feats1, feats2, feats3]