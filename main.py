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
import os

import tensorflow as tf
from unet.data import BATCH_SIZE
from unet.data import load_data
from unet.data import VAL_SUBSPLITS
from unet.model import create_model
from unet.model import EPOCHS
from unet.model import SAVE_FREQ
from unet.model import save_model
from unet.model import save_model_as_tflite
try:
    from unet.visualize import show_predictions
    from unet.visualize import visualize_training
except ImportError:
    import sys

    sys.stderr.write('Error: Couldn\'t import matplotlib.\n')
    sys.stderr.write('Run: `pip install "unet[plot]"`')
    sys.stderr.write(' or `poetry install --with plot`\n')

    raise SystemExit("Error: Couldn't import matplotlib.")

# Base directories.
BASE_DIR = os.path.dirname(__file__)
RES_DIR = os.path.join(BASE_DIR, 'res')
SAVED_MODELS = os.path.join(BASE_DIR, 'saved_models')

# Data directory
DATA_DIR = os.path.join(BASE_DIR, 'data')
# os.makedirs(DATA_DIR, exist_ok=True)

# Model directories.
UNET_DIR = os.path.join(SAVED_MODELS, 'unet')

# Model checkpoint during training.
MODEL_CKPT = os.path.join(
    UNET_DIR,
    'train/unet-{epoch:03d}.ckpt',
)
# os.makedirs(os.path.dirname(MODEL_CKPT),
#             exist_ok=True)

# Tensorboard logs.
LOG_DIR = os.path.join(UNET_DIR, 'logs')
# os.makedirs(LOG_DIR, exist_ok=True)

# Saved model.
MODEL_PATH = os.path.join(UNET_DIR, 'weights')
# os.makedirs(MODEL_PATH, exist_ok=True)

# TF Lite model.
TF_LITE_MODEL = os.path.join(UNET_DIR, 'unet.tflite')
# os.makedirs(os.path.dirname(TF_LITE_MODEL), exist_ok=True)


def main() -> int:
    """Main function.

    Returns:
        int: Error code. 0 if successful.

    """
    # Load data.
    train_dataset, val_dataset, info = load_data(
        data_dir=DATA_DIR,
    )

    # Estimate how many train steps to take per epoch.
    train_length = info.splits['train'].num_examples
    val_steps = info.splits['test'].num_examples // BATCH_SIZE // VAL_SUBSPLITS
    steps_per_epoch = train_length // BATCH_SIZE

    # Build (and compile) model.
    model = create_model(summary=True)

    # Callbacks: ModelCheckpoint, TensorBoard, EarlyStopping.
    callbacks = [
        # create checkpoints during training.
        tf.keras.callbacks.ModelCheckpoint(
            filepath=MODEL_CKPT,
            save_best_only=True,
            save_weights_only=True,
            save_freq=SAVE_FREQ,  # pyright: reportGeneralTypeIssues=false
            verbose=1,
        ),
        # Log training metrics to tensorboard.
        tf.keras.callbacks.TensorBoard(
            log_dir=LOG_DIR,
        ),
        # stop if validation loss doesn't improve.
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
        ),
    ]

    # Train model.
    history = model.fit(
        train_dataset,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_dataset,
        validation_steps=val_steps,
        callbacks=callbacks,
    )

    # Visualize training & predictions.
    visualize_training(history)
    show_predictions(model, val_dataset, num=3)

    # Save full model.
    save_model(model, MODEL_PATH)

    # Save model as TF Lite.
    save_model_as_tflite(model, TF_LITE_MODEL)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
