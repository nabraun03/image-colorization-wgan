import datetime
import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
from PIL import Image
from skimage.color import gray2rgb
from sklearn.model_selection import train_test_split
import cv2
import os
from matplotlib import pyplot as plt
import time
from IPython import display
from google.colab import drive

drive.mount('/content/drive')

FOLDER_PATH = '/content/drive/MyDrive/Image_colorizer/'
# Create a dataset from the generator
l_file_path = FOLDER_PATH + 'l/l_processed.npy'
ab_file_path = FOLDER_PATH + f'ab/ab_processed.npy'

IMAGE_SIZE = 32
LAMBDA = 100
LAMBDA_GP = 10
batch_size = 16
discriminator_ratio = 5

def normalize(lab):
    lab = tf.cast(lab, tf.float32)
    L = lab[..., 0] / 255.0
    a = (lab[..., 1] / 127.5) - 1.0
    b = (lab[..., 2] / 127.5) - 1.0

    # Expand dimensions to match the shape for concatenation
    L = tf.expand_dims(L, axis=-1)
    a = tf.expand_dims(a, axis=-1)
    b = tf.expand_dims(b, axis=-1)

    return tf.concat([L, a, b], axis=-1)


def denormalize(color):

    gray_norm = tf.cast(color[..., 0], tf.float32) * 100.0
    ab_norm = (tf.cast(color[..., 1:], tf.float32) + 1.0) * 127.5 - 128.0
    gray_norm = tf.expand_dims(gray_norm, axis=-1)
    return gray_norm, tf.concat([gray_norm, ab_norm], axis=-1)

def is_grayscale(image, threshold=20):
    # Calculate the absolute difference between color channels
    diff_rg = np.abs(image[..., 0] - image[..., 1])
    diff_rb = np.abs(image[..., 0] - image[..., 2])
    diff_gb = np.abs(image[..., 1] - image[..., 2])

    # Calculate the mean difference
    mean_diff = (diff_rg + diff_rb + diff_gb) / 3.0

    # If the mean difference is below the threshold, it's likely a grayscale image
    return np.mean(mean_diff) < threshold

def generate_image(model, test_input, tar):

    denorm_test_input, denorm_tar = denormalize(tar)


    display_test_input = np.array(denorm_test_input)
    display_test_input = np.squeeze(display_test_input, axis = -1)
    display_test_input = Image.fromarray((display_test_input).astype(np.uint8), mode = 'L')

    rgb_target = cv2.cvtColor(np.array(denorm_tar), cv2.COLOR_LAB2RGB)
    rgb_target = (rgb_target * 255.0).astype(np.uint8)
    display_tar = Image.fromarray(rgb_target)


    # Ensure the input image is 4D: (batch_size, height, width, channels)
    if len(test_input.shape) != 4:
        test_input = tf.expand_dims(test_input, axis=0)

    prediction = model(test_input, training=False)
    full_prediction = tf.concat([test_input[0], prediction[0]], axis = -1)
    temp, denorm_prediction = denormalize(full_prediction)
    rgb_prediction = cv2.cvtColor(np.array(denorm_prediction), cv2.COLOR_LAB2RGB)
    rgb_prediction = (rgb_prediction * 255.0).astype(np.uint8)
    prediction = Image.fromarray(rgb_prediction)

    plt.figure(figsize=(15, 15))


    display_list = [display_test_input, display_tar, prediction]
    title = ['Input Image', 'Target', 'Predicted Image']
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        # Convert the image data to uint8 before displaying
        img = tf.cast(display_list[i], tf.uint8)
        if i == 0:
            plt.imshow(img, cmap = 'gray')
        else:
            plt.imshow(img)
        plt.axis('off')
    plt.show()



def downsample(num_filters, size, apply_layernorm = False, apply_batchnorm = False):
    initializer = tf.random_normal_initializer(0, 0.02)

    model = keras.Sequential()
    model.add(layers.Conv2D(num_filters, size, strides = 2, padding = 'same', kernel_initializer = initializer, use_bias = False))
    if apply_layernorm:
      model.add(layers.LayerNormalization())
    elif apply_batchnorm:
      model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    return model

def upsample(num_filters, size, apply_dropout = False):
    initializer = tf.random_normal_initializer(0, 0.02)

    model = keras.Sequential()
    model.add(layers.UpSampling2D(size=2))
    model.add(layers.Conv2D(num_filters, size, strides = 1, padding = 'same', kernel_initializer = initializer, use_bias = False))

    model.add(layers.LayerNormalization())

    if apply_dropout:
        model.add(layers.Dropout(0.2))

    model.add(layers.ReLU())

    return model

class AddGaussianNoise(keras.layers.Layer):
    def __init__(self, stddev, **kwargs):
        super(AddGaussianNoise, self).__init__(**kwargs)
        self.stddev = stddev

    def call(self, inputs, training = None):
        if training:
            noise = tf.random.normal(shape=tf.shape(inputs), mean=0., stddev=self.stddev)
            return inputs + noise
        return inputs

class ResBlock(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()

        # Main pathway
        self.conv1 = layers.Conv2D(out_channels, kernel_size=3, strides=stride,
                                   padding='same', use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.ReLU()

        self.conv2 = layers.Conv2D(out_channels, kernel_size=3, strides=1,
                                   padding='same', use_bias=False)
        self.bn2 = layers.BatchNormalization()

        # Residual pathway
        # Only apply convolution if the spatial dimensions or channel count change
        if stride != 1 or in_channels != out_channels:
            self.identity_map = layers.Conv2D(out_channels, kernel_size=1, strides=stride)
        else:
            self.identity_map = None

        self.relu3 = layers.ReLU()

    def call(self, inputs, training=True):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)

        if self.identity_map is not None:
            residual = self.identity_map(inputs)
        else:
            residual = inputs

        x += residual
        return self.relu3(x)


def downsample_res(num_filters, size, apply_dropout=False):
    model = keras.Sequential()
    model.add(layers.Conv2D(num_filters, size, strides=2, padding='same', use_bias=False))  # Replaced MaxPooling with strided convolution
    model.add(ResBlock(num_filters, num_filters))

    if apply_dropout:
        model.add(layers.Dropout(0.2))

    return model


def upsample_res(num_filters, size, apply_dropout=False):
    model = keras.Sequential()
    model.add(layers.Conv2DTranspose(num_filters, size, strides=2, padding='same', use_bias=False))  # Replaced UpSampling2D with Conv2DTranspose
    model.add(ResBlock(num_filters, num_filters, stride=1))

    if apply_dropout:
        model.add(layers.Dropout(0.2))

    return model

def build_generator():
    inputs = layers.Input(shape=[IMAGE_SIZE, IMAGE_SIZE, 1])

    # Adjusted filter sizes for consistency
    down_stack = [
        downsample(256, 4, apply_batchnorm = True),
        downsample(512, 4, apply_batchnorm = True),
        downsample(512, 4, apply_batchnorm = True),
        downsample(1024, 4, apply_batchnorm = True),
    ]

    up_stack = [
        upsample_res(512, 4),
        upsample_res(512, 4, apply_dropout=True),
        upsample_res(256, 4, apply_dropout=True),
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
        x = up(x)
        x = layers.Concatenate()([x, skip])

    x = layers.UpSampling2D(size=2)(x)
    x = last_conv(x)

    return keras.Model(inputs=inputs, outputs=x)

class MiniBatchDiscrimination(tf.keras.layers.Layer):
    def __init__(self, B, C):
        super(MiniBatchDiscrimination, self).__init__()
        self.B = B
        self.C = C

    def build(self, input_shape):
        self.T = self.add_weight(name="T", shape=(input_shape[-1], self.B, self.C), initializer="uniform")

    def call(self, x):
        M_i = tf.tensordot(x, self.T, axes=[[3], [0]])
        M_i = tf.expand_dims(M_i, axis=0)
        M_j = tf.transpose(M_i, perm=[1, 0, 2, 3, 4, 5]) # Fixing the transpose operation
        norm = tf.reduce_sum(tf.abs(M_i - M_j), axis=5)  # L1 norm
        expnorm = tf.math.exp(-norm)
        o_b = tf.reduce_sum(expnorm, axis=1)

        return tf.concat([x, tf.reshape(o_b, (-1, x.shape[1], x.shape[2], self.B))], axis=3)



def build_discriminator():
    B, C = 10, 5  # You can adjust these parameters
    initializer = tf.random_normal_initializer(0., 0.02)

    input = layers.Input(shape=[IMAGE_SIZE, IMAGE_SIZE, 1])
    target = layers.Input(shape=[IMAGE_SIZE, IMAGE_SIZE, 2])

    target = AddGaussianNoise(stddev=0.01)(target)
    x = layers.concatenate([input, target])

    down1 = downsample(64, 4, apply_layernorm = True)(x)
    down2 = downsample(128, 4, apply_layernorm= True)(down1)
    down3 = downsample(256, 4, apply_layernorm= True)(down2)

    x = layers.Conv2D(256, (4, 4,), strides=(1, 1), padding='same', kernel_initializer=initializer)(down3)
    #x = MiniBatchDiscrimination(B, C)(x)  # Add MiniBatchDiscrimination layer here

    zero_pad1 = layers.ZeroPadding2D()(x)
    conv = layers.Conv2D(512, 4, strides=1, kernel_initializer=initializer, use_bias=False)(zero_pad1)

    norm_layer = tf.keras.layers.LayerNormalization()(conv)
    leaky_relu = layers.LeakyReLU()(norm_layer)
    zero_pad2 = layers.ZeroPadding2D()(leaky_relu)
    last = layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)(zero_pad2)

    return keras.Model(inputs=[input, target], outputs=last)


def compute_gradient_penalty(D, real_samples, fake_samples, real_input):
    alpha = tf.random.normal([real_samples.shape[0], 1, 1, 1], 0.0, 1.0)
    interpolated_samples = alpha * real_samples + (1 - alpha) * fake_samples

    with tf.GradientTape() as t:
        t.watch(interpolated_samples)
        validity = D([real_input, interpolated_samples])

    gradients = t.gradient(validity, interpolated_samples)
    gradient_penalty = tf.reduce_mean((tf.norm(gradients, axis=1) - 1.0) ** 2)

    return gradient_penalty


loss_obj = keras.losses.BinaryCrossentropy(from_logits = True)
def discriminator_loss(disc_real_output, disc_generated_output):
    # BCE Loss (Vanilla GAN Loss)
    real_loss_bce = loss_obj(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss_bce = loss_obj(tf.zeros_like(disc_generated_output), disc_generated_output)
    total_bce_loss = real_loss_bce + generated_loss_bce

    # WGAN Loss
    w_gan_real_loss = tf.reduce_mean(disc_real_output)
    w_gan_generated_loss = tf.reduce_mean(disc_generated_output)
    total_wgan_loss = w_gan_generated_loss - w_gan_real_loss

    return total_bce_loss, total_wgan_loss



def train_step(input_image, target, step, epoch):
    with tf.GradientTape(persistent = True) as gen_tape, tf.GradientTape(persistent = True) as disc_tape:
        gen_output = generator(input_image, training=True)

        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        # Compute the WGAN loss for the discriminator
        disc_real_loss = -tf.reduce_mean(disc_real_output)
        disc_fake_loss = tf.reduce_mean(disc_generated_output)
        disc_loss = disc_real_loss + disc_fake_loss

        # Compute the gradient penalty
        gradient_penalty = compute_gradient_penalty(discriminator, target, gen_output, input_image)
        disc_loss += LAMBDA_GP * gradient_penalty  # typically gradient_penalty_weight = 10

        # WGAN generator loss
        l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
        gen_loss = -tf.reduce_mean(disc_generated_output) + l1_loss * LAMBDA


    gen_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gen_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))

    for i in range(discriminator_ratio):
      disc_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
      disc_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))

    writer_step = step + len(train_ds) * epoch
    with summary_writer.as_default():
        tf.summary.scalar('gen_loss', gen_loss, step=writer_step)
        tf.summary.scalar('disc_real_loss', disc_real_loss, step=writer_step)
        tf.summary.scalar('disc_fake_loss', disc_fake_loss, step=writer_step)
        tf.summary.scalar('disc_loss', disc_loss, step=writer_step)
        # If you wish to monitor the gradient penalty separately
        tf.summary.scalar('gradient_penalty', gradient_penalty, step=writer_step)
        tf.summary.scalar('l1_loss', l1_loss, step = writer_step)

import heapq

def fit(train_ds, test_ds, epochs):
    train_end = len(train_ds)
    start = time.time()
    DISPLAY_N = 200  # Number of images to display from test set

    print("Training...")
    for epoch in range(epochs):
        print(f"Epoch {epoch}")
        train_ds_iter = iter(train_ds)
        print(train_end)

        for step in range(train_end):

            input, target = next(train_ds_iter)

            if (step) == 0:
                display.clear_output(wait=False)
                print(f'Time taken for epoch {epoch}:  {time.time() - start} sec\n')
                start = time.time()
                test_ds_iter = iter(test_ds.shuffle(buffer_size=len(test_ds)))

                for n in range(DISPLAY_N // batch_size):
                    example_input, example_target = next(test_ds_iter)
                    for i in range(batch_size):
                      generate_image(generator, example_input[i], example_target[i])



            noisy_input = AddGaussianNoise(stddev = 0.005)(target[..., 0])

            noisy_input = tf.clip_by_value(noisy_input, clip_value_min = 0., clip_value_max = 1.)

            train_step(noisy_input, target[..., 1:], step, epoch)

            if (step + 1) % 10 == 0:
                print('.')
                print(f'Epoch {epoch}: {step} / {train_end}')

            if step == train_end - 1:
                checkpoint.save(FOLDER_PATH + 'weights' + 'lambda100_bs16_strongergen' + '.ckpt')


def pretrain_step(input_image, target):
    with tf.GradientTape() as gen_tape:
        noisy_input = AddGaussianNoise(stddev = 0.005)(input_image)

        noisy_input = tf.clip_by_value(noisy_input, clip_value_min = 0., clip_value_max = 1.)

        gen_output = generator(noisy_input, training=True)
        loss = tf.reduce_mean(tf.abs(target[..., 1:] - gen_output))

    gradients = gen_tape.gradient(loss, generator.trainable_variables)
    gen_optimizer.apply_gradients(zip(gradients, generator.trainable_variables))

    return loss

def pretrain_generator(train_ds, test_ds, epochs):
    example_target = next(iter(test_ds))
    example_input = tf.expand_dims(example_target[..., 0], -1)
    for i in range(example_input.shape[0]):
        generate_image(generator, example_input[i], example_target[i])
    step = 0
    for epoch in range(epochs):
        print(f"Pretraining Epoch {epoch + 1}")
        for target in train_ds:
          input_image = target[..., 0]
          loss = pretrain_step(input_image, target)
          step += 1
          with summary_writer.as_default():
            tf.summary.scalar('gen_pretrain_loss', loss, step=step)
        print(f"Epoch {epoch + 1} Pre-training Loss: {loss:.4f}")
        display.clear_output(wait = False)
        example_target = next(iter(test_ds))
        example_input = tf.expand_dims(example_target[..., 0], -1)

        for i in range(example_input.shape[0]):
          generate_image(generator, example_input[i], example_target[i])

from keras.datasets import cifar10


def load_cifar():
    # Load the CIFAR-10 dataset
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()


    # Convert the filtered lists to LAB color space
    train_lab = [cv2.cvtColor(image, cv2.COLOR_RGB2LAB) for image in train_images]
    test_lab = [cv2.cvtColor(image, cv2.COLOR_RGB2LAB) for image in test_images]


    # Convert the filtered lists back to NumPy arrays
    train_lab = np.array(train_lab)
    test_lab = np.array(test_lab)

    train_lab = normalize(train_lab)
    test_lab = normalize(test_lab)

    train_gray = [np.expand_dims(img[:,:,0], axis=-1) for img in train_lab]
    test_gray = [np.expand_dims(img[:,:,0], axis=-1) for img in test_lab]


    # Create TensorFlow datasets
    train_ds = tf.data.Dataset.from_tensor_slices((train_gray, train_lab))
    test_ds = tf.data.Dataset.from_tensor_slices((test_gray, test_lab))

    return train_ds, test_ds


def get_data():

    l = np.load(l_file_path)
    print("Loaded L data!")
    ab = np.load(ab_file_path)
    print("Loaded AB data!")

    print("Concatenating...")
    return np.concatenate((l, ab), axis = -1)

def lab_to_rgb(lab):
    L = lab[..., 0:1]
    a = lab[..., 1:2]
    b = lab[..., 2:3]

    L = L * 100.0
    a = (a + 1.0) * 127.5 - 128.0
    b = (b + 1.0) * 127.5 - 128.0

    Y = (L + 16.) / 116.
    X = a / 500. + Y
    Z = Y - b / 200.

    Y3 = tf.where(Y > (6. / 29.) ** 3, Y ** 3, (Y - 4.0 / 29.0) / 7.787) * 100.0

    X3 = X * Y3 / 100.0
    Z3 = (1.0 - b / 200. - Y) * Y3

    # D65 illuminant scaling factors
    X = X3 * 0.95047
    Y = Y3
    Z = Z3 * 1.08883

    RGB = tf.concat([
        X * 3.2406 + Y * -1.5372 + Z * -0.4986,
        X * -0.9689 + Y * 1.8758 + Z * 0.0415,
        X * 0.0557 + Y * -0.2040 + Z * 1.0570], axis=-1)

    condition = RGB > 0.0031308
    positive_branch = 1.055 * tf.pow(tf.where(condition, RGB, 0.0031308), 1.0 / 2.4) - 0.055
    negative_branch = tf.where(condition, 0.0031308, RGB) * 12.92
    RGB = tf.where(condition, positive_branch, negative_branch)


    # Scale RGB from [0, 1] to [0, 255]
    rgb = tf.clip_by_value(RGB, 0.0, 1.0)

    return keras.applications.vgg19.preprocess_input(rgb)



"""
# Load the VGG19 model
vgg = keras.applications.VGG19(include_top=False, weights='imagenet', input_shape=[IMAGE_SIZE, IMAGE_SIZE, 3])

# Choose the layers you want to use for the perceptual loss (you can experiment with different layers)
vgg_layers = [vgg.get_layer(name).output for name in ['block3_conv3', 'block4_conv3']]

# Create a model that will return these outputs given the VGG inputs
vgg_model = keras.models.Model(inputs=vgg.input, outputs=vgg_layers)

# Make sure the VGG model is not trainable
vgg_model.trainable = False

def vgg_loss(y_true, y_pred):

    # Convert the grayscale input and AB channels to RGB (if necessary)
    y_true_rgb = lab_to_rgb(y_true) # Implement this conversion based on your data format
    y_pred_rgb = lab_to_rgb(y_pred) # Implement this conversion based on your data format

    # Extract the features
    y_true_features = vgg_model(y_true_rgb)
    y_pred_features = vgg_model(y_pred_rgb)
    # Compute the perceptual loss
    loss = 0
    for true_feat, pred_feat in zip(y_true_features, y_pred_features):
        loss += tf.reduce_mean(tf.abs(true_feat - pred_feat))

    return loss
"""


# Convert the training and testing data into TensorFlow datasets
train_ds, test_ds = load_cifar()

# Apply normalization and batching
train_ds = train_ds.batch(batch_size)
test_ds = test_ds.batch(batch_size)

print("Creating generator...")
generator = build_generator()
print("Creating discriminator...")
discriminator = build_discriminator()
print("Creating optimizers + checkpoints...")
gen_optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0002, beta_1 = 0.5, beta_2 = 0.99)
disc_optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0002, beta_1 = 0.5, beta_2 = 0.99)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer= gen_optimizer, discriminator_optimizer= disc_optimizer, generator=generator, discriminator=discriminator)
log_dir = FOLDER_PATH + "logs/"

summary_writer = tf.summary.create_file_writer(
  log_dir + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))



weights_path = FOLDER_PATH + "weightslambda100_bs16.ckpt-5"
print(weights_path)

checkpoint.restore(weights_path)
#pretrain_generator(train_ds, test_ds, 1)
fit(train_ds, test_ds, epochs = 20)

#feature_matching(train_ds, test_ds, 20)