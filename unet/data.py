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

IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL = 128, 128, 3
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)
IMG_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL)
OUTPUT_CLASSES, VAL_SUBSPLITS = 3, 5


class Augment(tf.keras.layers.Layer):
    """Randomly flip both image and image mask."""

    def __init__(self, seed: int = 42) -> None:
        """Augment the images and labels by flipping them randomly.

        Args:
            seed (int, optional): Random seed number. Defaults to 42.
        """
        super(Augment, self).__init__()

        # Both use the same seed, so they'll make the same random changes.
        self.augment_inputs = tf.keras.layers.RandomFlip(
            mode='horizontal', seed=seed,
        )
        self.augment_labels = tf.keras.layers.RandomFlip(
            mode='horizontal', seed=seed,
        )

    def call(
        self, inputs: tf.Tensor, labels: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Apply augmentation to both image & label (image mask).

        Args:
            inputs (tf.Tensor): Original image.
            labels (tf.Tensor): Image mask.

        Returns:
            tuple[tf.Tensor, tf.Tensor]: Augmented image and image mask.
        """

        inputs = self.augment_inputs(inputs)
        labels = self.augment_labels(labels)

        return inputs, labels


def normalize(
    input_image: tf.TensorArray,
    input_mask: tf.TensorArray
) -> tuple[tf.TensorArray, tf.TensorArray]:
    """Normalize image & image mask.

    Arguments:
        input_image (tf.TensorArray): Input image in range [0, 255].
        input_mask (tf.TensorArray): Input mask in range [1, 3].

    Returns:
        tuple[tf.TensorArray, tf.TensorArray]:
            Normalized image in range [0, 1] and normalized image
            mask in range [0, 2].
    """
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask -= 1

    return input_image, input_mask


def load_image(
    datapoint: dict[str, tf.Tensor]
) -> tuple[tf.TensorArray, tf.TensorArray]:
    """Resize and normalize image and image mask.

    Arguments:
        datapoint (dict[str, Tf.Tensor]): Each image data.

    Returns:
        tuple[tf.TensorArray, tf.TensorArray]:
            Resized & normalized image & image mask.
    """
    input_image = tf.image.resize(datapoint['image'], IMG_SIZE)
    input_mask = tf.image.resize(datapoint['segmentation_mask'], IMG_SIZE)

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask
