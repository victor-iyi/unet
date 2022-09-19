# Copyright 2022 Victor I. Afolabi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
import matplotlib.pyplot as plt


def visualize_training(
    history: tf.keras.callbacks.History,
    title: str = 'Training History',
) -> None:
    """Visualize training history.

    Args:
        history (tf.keras.callbacks.History): Training history.
        title (str, optional): Title of the plot.
            Defaults to 'Training History'.
    """
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure()
    plt.title(title)

    plt.plot(history.epoch, loss, label='Training Loss')
    plt.plot(history.epoch, val_loss, label='Validation Loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim([0, 1])

    plt.legend()
    plt.show()


def display(
    images: list[tf.TensorArray],
    figsize: tuple[int, int] = (10, 10)
) -> None:
    """Display images in a grid.

    Args:
        images (list[tf.TensorArray]): List of image(s) to display.
    """
    plt.figure(figsize=figsize)

    title = ['Input Image', 'True Mask', 'Predicted Mask']
    n_image = len(images)

    for i, image in enumerate(images):
        plt.subplot(1, n_image, i + 1)
        plt.title(title[i])
        plt.imshow(tf.keras.utils.array_to_img(image))
        plt.axis('off')

    plt.show()
