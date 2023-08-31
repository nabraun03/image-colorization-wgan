import tensorflow as tf
from tensorflow import keras
from keras import layers
from custom_layers.ResBlock import ResBlock

class Generator(keras.Model):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.config = config
        self.image_size = config['IMAGE_SIZE']
    
    def build(self, input_shapes):
        inputs = layers.Input(shape = [self.image_size, self.image_size, 1])

        # Adjusted filter sizes for consistency
        down_stack = [
            self.downsample(256, 4, apply_batchnorm = False),
            self.downsample(512, 4),
            self.downsample(512, 4),
            self.downsample(1024, 4),
        ]

        up_stack = [
            
            self.upsample_res(512, 4),
            self.upsample_res(512, 4, apply_dropout=True),
            self.upsample_res(256, 4, apply_dropout=True),
        ]

        # Used a 3x3 kernel for the final convolution
        last_conv = layers.Conv2D(2, 3, strides=1, padding='same',
                                kernel_initializer=tf.random_normal_initializer(0., 0.02),
                                activation='tanh')

        x = inputs
        skips = []
        for down in down_stack:
            x = down(x)
            skips.append(x)

        skips = reversed(skips[:-1])

        for up, skip in zip(up_stack, skips):
            print(x.shape)
            x = up(x)
            print(x.shape)
            x = layers.Concatenate()([x, skip])

        x = layers.UpSampling2D(size=2)(x)
        x = last_conv(x)

        self.model = keras.Model(inputs=inputs, outputs=x)

    def call(self, inputs, training = True):
        return self.model(inputs, training = training)


    def downsample(self, num_filters, size, apply_batchnorm = True):
        initializer = tf.random_normal_initializer(0, 0.02)

        model = keras.Sequential()
        model.add(layers.Conv2D(num_filters, size, strides = 2, padding = 'same', kernel_initializer = initializer, use_bias = False))

        if apply_batchnorm:
            model.add(layers.BatchNormalization())
            model.add(layers.LeakyReLU())

        return model

        
    def upsample_res(self, num_filters, size, apply_dropout=False):
        model = keras.Sequential()
        model.add(layers.Conv2DTranspose(num_filters, size, strides=2, padding='same', use_bias=False)) 
        model.add(ResBlock(num_filters, num_filters, stride=1))

        if apply_dropout:
            model.add(layers.Dropout(0.2))

        return model