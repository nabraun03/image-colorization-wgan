import tensorflow as tf
from PIL import Image
import cv2
import numpy as np
from matplotlib import pyplot as plt

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


    print(test_input.shape)
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
