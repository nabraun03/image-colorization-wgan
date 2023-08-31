import tensorflow as tf
from PIL import Image
import cv2
import numpy as np
from matplotlib import pyplot as plt


def normalize(lab):
    """
    Normalize the LAB color space values.

    Args:
        lab (tf.Tensor): A tensor containing the LAB color data from the dataset, scaled from [0, 255]

    Returns:
        tf.Tensor: A tensor containing the normalized LAB color values. The L channel is scaled to [0, 1], and the A and B channels are scaled to [-1, 1]
    """
    lab = tf.cast(lab, tf.float32)
    L = lab[..., 0] / 255.0
    a = (lab[..., 1] / 127.5) - 1.0
    b = (lab[..., 2] / 127.5) - 1.0

    # Expand dimensions to match the shape for concatenation
    L, a, b = [tf.expand_dims(x, axis=-1) for x in [L, a, b]]

    return tf.concat([L, a, b], axis=-1)


def denormalize(color):
    """
    Denormalize the LAB color space values.

    Args:
        color (tf.Tensor): A tensor containing the normalized LAB color values, with the L channel scaled from [0, 1] and the A and B channels scaled to [-1, 1]

    Returns:
        tuple: A tuple containing the denormalized grayscale and LAB color values. The L channel is scaled from [0, 100], and the A and B channels are scaled from [-128, 127]
    """
    gray_norm = tf.cast(color[..., 0], tf.float32) * 100.0
    ab_norm = (tf.cast(color[..., 1:], tf.float32) + 1.0) * 127.5 - 128.0
    gray_norm = tf.expand_dims(gray_norm, axis=-1)
    return gray_norm, tf.concat([gray_norm, ab_norm], axis=-1)


def generate_image(model, test_input, tar):
    """
    Generate and display the input, target, and predicted images.

    Args:
        model (tf.keras.Model): The trained model.
        test_input (tf.Tensor): The input grayscale image.
        tar (tf.Tensor): The target LAB color image.
    """
    # Rescale image data
    denorm_test_input, denorm_tar = denormalize(tar)

    # Convert tensors to PIL images for display
    display_test_input = Image.fromarray(
        (np.squeeze(np.array(denorm_test_input), axis=-1)).astype(np.uint8), mode="L"
    )
    display_tar = Image.fromarray(
        (cv2.cvtColor(np.array(denorm_tar), cv2.COLOR_LAB2RGB) * 255.0).astype(np.uint8)
    )

    # Ensure the input image is 4D: (batch_size, height, width, channels)
    test_input = (
        tf.expand_dims(test_input, axis=0) if len(test_input.shape) != 4 else test_input
    )

    # Generate image representing generator prediction given grayscale data
    prediction = model(test_input, training=False)
    full_prediction = tf.concat([test_input[0], prediction[0]], axis=-1)
    _, denorm_prediction = denormalize(full_prediction)
    prediction = Image.fromarray(
        (cv2.cvtColor(np.array(denorm_prediction), cv2.COLOR_LAB2RGB) * 255.0).astype(
            np.uint8
        )
    )

    # Plotting
    plt.figure(figsize=(15, 15))
    display_list = [display_test_input, display_tar, prediction]
    title = ["Input Image", "Target", "Predicted Image"]

    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(title[i])
        plt.imshow(
            display_list[i] if i else display_list[i].convert("L"),
            cmap="gray" if i == 0 else None,
        )
        plt.axis("off")
    plt.show()
