from keras.datasets import cifar10
import tensorflow as tf
import numpy as np
import cv2
from utils.image_processing import normalize

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