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

from unet.data import IMG_SHAPE, OUTPUT_CLASSES


class Encoder(tf.keras.layers.Layer):
    def __init__(
        self, input_shape: tuple[int, int, int] = IMG_SHAPE,
    ) -> None:
        """Encode the image with a pre-trained model.

        Args:
          input_shape (tuple[int, int, int]): Image input shape.
        """
        super(Encoder, self).__init__()

        model = tf.keras.applications.MobileNetV2(
            input_shape=input_shape, include_top=False,
        )

        # Use the activations of these layers.
        layer_names = [
          'block_1_expand_relu',   # 64x64
          'block_3_expand_relu',   # 32x32
          'block_6_expand_relu',   # 16x16
          'block_13_expand_relu',  # 8x8
          'block_16_project',      # 4x4
        ]

        model_output = [model.get_layer(name).output
                        for name in layer_names]
        self.encoder_model = tf.keras.Model(inputs=model.input,
                                            outputs=model_output)
        self.encoder_model.trainable = False

    def call(
        self, x: tf.TensorArray, training: bool = False,
    ) -> list[tf.TensorArray]:
        """Encode the image with an input.

        Args:
          x (tf.TensorArray): Image to encode with shape
            [batch_size, img_height, img_widht, img_channel].
          training (bool, optional): Training mode. Defaults to False.

        Returns:
          tf.TensorArray - List of encoded output from each layer of interest.
        """
        return self.encoder_model(x)


class UNet(tf.keras.Model):
    def __init__(
        self, input_shape: tuple[int, int, int] = IMG_SHAPE,
        output_channels: int = OUTPUT_CLASSES,
        dropout: float = 0.5,
    ) -> None:
        """U-Net: Convolutional Networks for Biomedical Image Segmentation.

        Source: <https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/>

        Args:
          input_shape (tuple[int, int, int]): Input shape.
            Defaults to IMG_SHAPE.
          output_channels (int): Number of output channels.
            Defaults to OUTPUT_CLASSES.
          dropout (float): Dropout rate. Defaults to 0.5.
        """
        super(UNet, self).__init__()
        self.encoder = Encoder(input_shape=input_shape)

        # Don't train the encoer.
        self.encoder.trainable = False

        decoder_filters = [512, 256, 128, 64]
        initializer = tf.random_normal_initializer(0., 0.02)
        self.concat = tf.keras.layers.Concatenate()

        # Decoder (upsampler).
        self.decoder_stack = [tf.keras.Sequential([
          tf.keras.layers.Conv2DTranspose(
            filters, kernel_size=3, strides=2,
            padding='same', use_bias=False,
            kernel_initializer=initializer,
          ),
          tf.keras.layers.BatchNormalization(),
          tf.keras.layers.Dropout(dropout),
          tf.keras.layers.ReLU(),
        ]) for filters in decoder_filters]

        # Final (output) layer.
        self.output_layer = tf.keras.layers.Conv2DTranspose(
            filters=output_channels, kernel_size=3,
            strides=2, padding='same',
        )

    def call(
        self, inputs: tf.TensorArray, training: bool = False,
    ) -> tf.TensorArray:
        """Call the U-Net model with input(s).

        Args:
          input (tf.TensorArray): Input image with shape
            [batch_size, img_height, img_width, img_channel].
          training (bool, optional): Training mode. Defaults to False.

        Returns:
          tf.TensorArray - Predicted output image mask.
        """
        # Downsampling.
        encoder_outputs = self.encoder(inputs)
        x = encoder_outputs[-1]

        # Skip connections.
        skips = reversed(encoder_outputs[:-1])

        # Upsampling and establishing the skip connections.
        for decoder, skip in zip(self.decoder_stack, skips):
            x = decoder(x)
            x = self.concat([x, skip])

        # This is the last layer of the model
        output = self.output_layer(x)
        return output
