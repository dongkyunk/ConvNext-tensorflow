import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras.layers as layers


kernel_initial = tf.keras.initializers.TruncatedNormal(stddev=0.2)
bias_initial = tf.keras.initializers.Constant(value=0)


class Downsampling(tf.keras.Sequential):
    def __init__(self, out_dim):
        super(Downsampling, self).__init__([
            layers.LayerNormalization(),
            layers.Conv2D(
                out_dim, kernel_size=2, strides=2, padding='same',
                kernel_initializer=kernel_initial, bias_initializer=bias_initial
            )
        ])


class ConvNextBlock(layers.Layer):
    """ConvNeXt Block using implementation 1
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_prob=0, layer_scale_init_value=1e-6):
        super().__init__()
        self.layers = tf.keras.Sequential([
            layers.Conv2D(dim, kernel_size=7, padding="same", groups=dim,
                          kernel_initializer=kernel_initial, bias_initializer=bias_initial),
            layers.LayerNormalization(),
            layers.Conv2D(dim*4, kernel_size=1, padding="valid",
                          kernel_initializer=kernel_initial, bias_initializer=bias_initial),
            layers.Activation('gelu'),
            layers.Conv2D(dim, kernel_size=1, padding="valid",
                          kernel_initializer=kernel_initial, bias_initializer=bias_initial),
        ])
        if layer_scale_init_value > 0:
            self.layer_scale_gamma = tf.Variable(
                initial_value=layer_scale_init_value*tf.ones((dim)))
        else:
            self.layer_scale_gamma = None
        self.stochastic_depth = tfa.layers.StochasticDepth(drop_prob)

    def call(self, x):
        skip = x
        x = self.layers(x)
        if self.layer_scale_gamma is not None:
            x = x * self.layer_scale_gamma
        x = self.stochastic_depth([skip, x])
        return x


class ConvNext(tf.keras.Model):
    """ A Tensorflow impl of : `A ConvNet for the 2020s`
        https://arxiv.org/pdf/2201.03545.pdf
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, num_classes=1000,
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0.,
                 layer_scale_init_value=1e-6):
        super().__init__()
        self.downsample_layers = []
        self.downsample_layers.append(tf.keras.Sequential([
            layers.Conv2D(dims[0], kernel_size=4, strides=4, padding="valid"),
            layers.LayerNormalization()
        ]))
        self.downsample_layers += [Downsampling(dim) for dim in dims[1:]]
        self.convnext_blocks = [tf.keras.Sequential([ConvNextBlock(dim, drop_path_rate, layer_scale_init_value) for _ in range(
            depths[i])]) for i, dim in enumerate(dims)]
        self.head = layers.Dense(
            num_classes, kernel_initializer=kernel_initial, bias_initializer=bias_initial)
        self.gap = layers.GlobalAveragePooling2D()
        self.norm = layers.LayerNormalization()

    def call_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.convnext_blocks[i](x)
        x = self.gap(x)
        return self.norm(x)

    def call(self, x):
        x = self.call_features(x)
        x = self.head(x)
        return x
