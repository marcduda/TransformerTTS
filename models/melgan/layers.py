import tensorflow as tf
import tensorflow_addons as tfa


class ReplicationPadding(tf.keras.layers.Layer):
    def __init__(self, pad_size, **kwargs):
        super().__init__(**kwargs)
        self.padding = [[0, 0], [pad_size, pad_size], [0, 0]]
    
    def __call__(self, x, **kwargs):
        return tf.pad(x, self.padding, mode="SYMMETRIC")


class PaddedWNConv1D(tf.keras.layers.Layer):
    def __init__(self, channels: int, kernel_size: int, dilation=1, padding: int = None, strides=1,
                 groups=1, **kwargs):
        super(PaddedWNConv1D, self).__init__(**kwargs)
        assert kernel_size % 2 == 1, 'Kernel size must be odd.'
        if padding is None:
            if strides > 1:
                raise ValueError('Need to specify padding dimension if strides > 1.')
            padding = (kernel_size - 1) // 2 * dilation
        
        self.layers = []
        if padding > 0:
            self.layers += [ReplicationPadding(pad_size=padding)]
        self.layers += [
            tfa.layers.WeightNormalization(
                tf.keras.layers.Conv1D(channels, padding='valid', kernel_size=kernel_size, dilation_rate=dilation,
                                       groups=groups, kernel_initializer=get_initializer(42)))]
        # if groups ==1: # TF ADDONS not compatible with tf2.3 (groups are only in 2.3)
        #     self.layers += [tfa.layers.WeightNormalization(
        #          tf.keras.layers.Conv1D(channels, padding='valid', kernel_size=kernel_size, dilation_rate=dilation, groups=groups))]
        # else:
        #     self.layers += [tf.keras.layers.Conv1D(channels, padding='valid', kernel_size=kernel_size, dilation_rate=dilation,
        #                                groups=groups)]
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, channels: int, dilation: int, kernel_size: int, leaky_alpha: float, **kwargs):
        super().__init__(**kwargs)
        self.leaky_relu1 = tf.keras.layers.LeakyReLU(alpha=leaky_alpha)
        self.norm_conv1 = PaddedWNConv1D(channels=channels, kernel_size=kernel_size, dilation=dilation)
        
        self.leaky_relu2 = tf.keras.layers.LeakyReLU(alpha=leaky_alpha)
        self.norm_conv2 = PaddedWNConv1D(channels=channels, kernel_size=1, dilation=1)
        
        self.residual = PaddedWNConv1D(channels=channels, kernel_size=1, dilation=1)
    
    def _call_block(self, x):
        x = self.leaky_relu1(x)
        x = self.norm_conv1(x)
        x = self.leaky_relu2(x)
        x = self.norm_conv2(x)
        return x
    
    def __call__(self, x, **kwargs):
        residual = self.residual(x)
        block = self._call_block(x)
        return block + residual


class ResidualStack(tf.keras.layers.Layer):
    def __init__(self, n_layers: int, channels: int, kernel_size: int = 3, leaky_alpha: float = 0.2, **kwargs):
        super().__init__(**kwargs)
        self.n_layers = n_layers
        self.channels = channels
        self.residual_blocks = [
            ResidualBlock(channels=channels, kernel_size=kernel_size, dilation=kernel_size ** i,
                          leaky_alpha=leaky_alpha)
            for i in range(n_layers)
        ]
    
    def __call__(self, x, **kwargs):
        for block in self.residual_blocks:
            x = block(x)
        return x


class Upscale1D(tf.keras.layers.Layer):
    def __init__(self, output_channels: int, scale: int, kernel_size: int, **kwargs):
        super().__init__(**kwargs)
        self.scale = scale
        # self.conv = PaddedWNConv1D(channels=output_channels * scale, kernel_size=kernel_size, dilation=1)
        self.conv = tfa.layers.WeightNormalization(
                tf.keras.layers.Conv1D(output_channels*scale, padding='same', kernel_size=kernel_size, dilation_rate=1, kernel_initializer=get_initializer(42)))
    
    def _pixel_shuffle(self, x):
        x = tf.transpose(a=x, perm=[2, 1, 0])
        x = tf.batch_to_space(input=x, block_shape=[self.scale], crops=[[0, 0]])
        x = tf.transpose(a=x, perm=[2, 1, 0])
        return x
    
    def __call__(self, x, **kwargs):
        x = self.conv(x)
        return self._pixel_shuffle(x)

def get_initializer(initializer_seed=42):
    """Creates a `tf.initializers.glorot_normal` with the given seed.
    Args:
        initializer_seed: int, initializer seed.
    Returns:
        GlorotNormal initializer with seed = `initializer_seed`.
    """
    return tf.keras.initializers.GlorotNormal(seed=initializer_seed)


class TFConvTranspose1d(tf.keras.layers.Layer):
    """Tensorflow ConvTranspose1d module."""

    def __init__(
        self,
        filters,
        kernel_size,
        strides,
        is_weight_norm,
        initializer_seed,
        **kwargs
    ):
        """Initialize TFConvTranspose1d( module.
        Args:
            filters (int): Number of filters.
            kernel_size (int): kernel size.
            strides (int): Stride width.
            padding (str): Padding type ("same" or "valid").
        """
        super().__init__(**kwargs)
        self.conv1d_transpose = tf.keras.layers.Conv2DTranspose(
            filters=filters,
            kernel_size=(kernel_size, 1),
            strides=(strides, 1),
            padding="same",
            kernel_initializer=get_initializer(initializer_seed),
        )
        # if is_weight_norm:
        #     self.conv1d_transpose = tfa.layers.WeightNormalization(self.conv1d_transpose)

    def call(self, x):
        """Calculate forward propagation.
        Args:
            x (Tensor): Input tensor (B, T, C).
        Returns:
            Tensor: Output tensor (B, T', C').
        """
        x = tf.expand_dims(x, 2)
        x = self.conv1d_transpose(x)
        x = tf.squeeze(x, 2)
        return x


class DiscriminatorBlock(tf.keras.layers.Layer):
    def __init__(self, leaky_alpha: float = 0.2, **kwargs):
        super(DiscriminatorBlock, self).__init__(**kwargs)
        groups = [4, 16, 64, 256, 64]
        f_b_a = False  # features_before_activation
        # groups = [1,1,1,1,1] #TODO : restore, CPU testing only
        self.model_layers = []
        # model_layers have two elements: keras layer, is_output_feature bool
        self.model_layers += [(PaddedWNConv1D(channels=16, kernel_size=15), f_b_a)]
        self.model_layers += [(tf.keras.layers.LeakyReLU(alpha=leaky_alpha), not f_b_a)]
        self.model_layers += [
            (PaddedWNConv1D(channels=64, kernel_size=41, groups=groups[0], strides=4, padding=20), f_b_a)]
        self.model_layers += [(tf.keras.layers.LeakyReLU(alpha=leaky_alpha), not f_b_a)]
        self.model_layers += [
            (PaddedWNConv1D(channels=256, kernel_size=41, groups=groups[1], strides=4, padding=20), f_b_a)]
        self.model_layers += [(tf.keras.layers.LeakyReLU(alpha=leaky_alpha), not f_b_a)]
        self.model_layers += [
            (PaddedWNConv1D(channels=1024, kernel_size=41, groups=groups[2], strides=4, padding=20), f_b_a)]
        self.model_layers += [(tf.keras.layers.LeakyReLU(alpha=leaky_alpha), not f_b_a)]
        self.model_layers += [
            (PaddedWNConv1D(channels=1024, kernel_size=41, groups=groups[3], strides=4, padding=20), f_b_a)]
        self.model_layers += [(tf.keras.layers.LeakyReLU(alpha=leaky_alpha), not f_b_a)]
        self.model_layers += [(PaddedWNConv1D(channels=1024, kernel_size=5, groups=groups[4]), f_b_a)]
        self.model_layers += [(tf.keras.layers.LeakyReLU(alpha=leaky_alpha), not f_b_a)]
        self.model_layers += [(PaddedWNConv1D(channels=1, kernel_size=3), f_b_a)]
    
    def __call__(self, x, **kwargs):
        features = []
        for i, layer in enumerate(self.model_layers):
            x = layer[0](x)
            if layer[1] is True:
                features += [x]
        return x, features
