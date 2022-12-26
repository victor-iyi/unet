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
import matplotlib.pyplot as plt
import tensorflow as tf


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
    figsize: tuple[int, int] = (10, 10),
) -> None:
    """Display images in a grid.

    Args:
        images (list[tf.TensorArray]): List of image(s) to display.
        figsize (tuple[int, int], optional): Figure size.
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


def create_mask(pred_mask: tf.TensorArray) -> tf.TensorArray:
    """Create image mask with predicted output.

    Args:
        pred_mask (tf.TensorArray): Predicted image mask with shape
            [batch_size, img_height, img_width, img_channel].

    Returns:
        tf.TensorArray - Prediction mask with shape
            [img_height, img_width, img_channel]
    """
    pred_mask = tf.math.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]

    return pred_mask[0]


def predict(images: tf.TensorArray, model: tf.keras.Model) -> tf.TensorArray:
    """Predict image mask.

    Args:
        data (tf.TensorArray): Image to predict with shape
            [batch_size, img_height, img_width, img_channel].
        model (tf.keras.Model): Model to use for prediction.

    Returns:
        tf.TensorArray - Prediction mask with shape
            [img_height, img_width, img_channel]
    """
    pred_mask = model.predict(images)
    pred_mask = create_mask(pred_mask)
    return pred_mask


def show_predictions(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    num: int = 1,
) -> None:
    """Display image, image mask and predicted mask.

    Args:
        model (tf.keras.Model, optional): Model to use for prediction.
        dataset (tf.data.Dataset, optional): Dataset containing
            image & image mask.
            Each with shape [batch_size, img_height, img_width, img_channel].
        num (int): Number of images triples to display. Defaults to 1.
    """
    for image, mask in dataset.take(num):
        pred_mask = model.predict(image)
        display([image[0], mask[0], create_mask(pred_mask)])
