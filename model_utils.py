"""
model_utils.py
Core utilities for the CNN Image Classifier:
- CNN architecture (Keras)
- Data loading: built-in quick-demo dataset (CIFAR-10) + custom folder datasets
- Training helpers
- Grad-CAM explainability
"""

import io
import json

import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


# --------------------------------------------------------------------------
# DATA LOADING
# --------------------------------------------------------------------------
def load_cifar10_subset(samples_per_class: int = 300, img_size: int = 32):
    """Load a small, fast-to-train subset of the built-in CIFAR-10 dataset."""
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    y_train = y_train.flatten()
    y_test = y_test.flatten()

    def subsample(x, y, per_class):
        idxs = []
        for c in range(10):
            class_idxs = np.where(y == c)[0][:per_class]
            idxs.extend(class_idxs)
        idxs = np.array(idxs)
        np.random.shuffle(idxs)
        return x[idxs], y[idxs]

    x_train, y_train = subsample(x_train, y_train, samples_per_class)
    x_test, y_test = subsample(x_test, y_test, max(20, samples_per_class // 5))

    if img_size != 32:
        x_train = tf.image.resize(x_train, (img_size, img_size)).numpy()
        x_test = tf.image.resize(x_test, (img_size, img_size)).numpy()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    y_train_cat = keras.utils.to_categorical(y_train, 10)
    y_test_cat = keras.utils.to_categorical(y_test, 10)

    return (x_train, y_train_cat, y_train), (x_test, y_test_cat, y_test), CIFAR10_CLASSES


def build_array_generators(x_train, y_train, x_test, y_test, batch_size=32):
    """Build augmented train / plain validation generators from in-memory arrays."""
    train_datagen = ImageDataGenerator(
        rotation_range=15, width_shift_range=0.1, height_shift_range=0.1,
        zoom_range=0.1, horizontal_flip=True,
    )
    val_datagen = ImageDataGenerator()

    train_gen = train_datagen.flow(x_train, y_train, batch_size=batch_size)
    val_gen = val_datagen.flow(x_test, y_test, batch_size=batch_size, shuffle=False)
    return train_gen, val_gen


def build_directory_generators(train_dir, val_dir, img_size=96, batch_size=32, validation_split=None):
    """
    Build train/val generators from a folder-per-class directory structure:
        train_dir/classA/*.jpg, train_dir/classB/*.jpg, ...
    If val_dir is None, validation_split carves out a portion of train_dir instead.
    """
    if val_dir:
        train_datagen = ImageDataGenerator(
            rescale=1 / 255.0, rotation_range=20, width_shift_range=0.15,
            height_shift_range=0.15, shear_range=0.1, zoom_range=0.15,
            horizontal_flip=True,
        )
        val_datagen = ImageDataGenerator(rescale=1 / 255.0)

        train_gen = train_datagen.flow_from_directory(
            train_dir, target_size=(img_size, img_size), batch_size=batch_size, class_mode="categorical"
        )
        val_gen = val_datagen.flow_from_directory(
            val_dir, target_size=(img_size, img_size), batch_size=batch_size, class_mode="categorical", shuffle=False
        )
    else:
        datagen = ImageDataGenerator(
            rescale=1 / 255.0, rotation_range=20, width_shift_range=0.15,
            height_shift_range=0.15, shear_range=0.1, zoom_range=0.15,
            horizontal_flip=True, validation_split=validation_split or 0.2,
        )
        train_gen = datagen.flow_from_directory(
            train_dir, target_size=(img_size, img_size), batch_size=batch_size,
            class_mode="categorical", subset="training",
        )
        val_gen = datagen.flow_from_directory(
            train_dir, target_size=(img_size, img_size), batch_size=batch_size,
            class_mode="categorical", subset="validation", shuffle=False,
        )

    class_names = list(train_gen.class_indices.keys())
    return train_gen, val_gen, class_names


# --------------------------------------------------------------------------
# CNN MODEL
# --------------------------------------------------------------------------
def build_cnn_model(input_shape=(64, 64, 3), num_classes=10, conv_blocks=3, base_filters=32, dropout=0.4, lr=0.001):
    """Build and compile a configurable Convolutional Neural Network."""
    model = keras.Sequential(name="CNN_Image_Classifier")
    model.add(layers.Input(shape=input_shape))

    filters = base_filters
    for i in range(conv_blocks):
        model.add(layers.Conv2D(filters, (3, 3), padding="same", activation="relu", name=f"conv_{i+1}a"))
        model.add(layers.Conv2D(filters, (3, 3), padding="same", activation="relu", name=f"conv_{i+1}b"))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(dropout * 0.5))
        filters *= 2

    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(256, activation="relu", name="dense_1"))
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(num_classes, activation="softmax", name="predictions"))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_cnn(model, train_gen, val_gen, epochs=15):
    early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6)
    history = model.fit(
        train_gen, validation_data=val_gen, epochs=epochs,
        callbacks=[early_stop, reduce_lr], verbose=0,
    )
    return history


# --------------------------------------------------------------------------
# IMAGE PREPROCESSING
# --------------------------------------------------------------------------
def preprocess_image(pil_image: Image.Image, img_size: int) -> np.ndarray:
    """Convert a PIL image to a normalized (1, H, W, 3) float array."""
    img = pil_image.convert("RGB").resize((img_size, img_size))
    arr = np.array(img).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)


# --------------------------------------------------------------------------
# GRAD-CAM
# --------------------------------------------------------------------------
def find_last_conv_layer(model) -> str:
    for layer in reversed(model.layers):
        if isinstance(layer, layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found in model.")


def make_gradcam_heatmap(img_array: np.ndarray, model, last_conv_layer_name: str = None, pred_index: int = None):
    """Generate a Grad-CAM heatmap (values in [0,1]) highlighting influential image regions."""
    if last_conv_layer_name is None:
        last_conv_layer_name = find_last_conv_layer(model)

    grad_model = keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-9)
    return heatmap.numpy()


def overlay_heatmap(pil_image: Image.Image, heatmap: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Overlay a Grad-CAM heatmap onto the original PIL image using a jet-like colormap."""
    import matplotlib.cm as cm

    img = pil_image.convert("RGB")
    heatmap_resized = np.array(Image.fromarray((heatmap * 255).astype("uint8")).resize(img.size))

    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap_resized]
    jet_heatmap = (jet_heatmap * 255).astype("uint8")
    jet_heatmap_img = Image.fromarray(jet_heatmap).resize(img.size)

    blended = Image.blend(img, jet_heatmap_img, alpha=alpha)
    return blended


# --------------------------------------------------------------------------
# SAVE / LOAD HELPERS
# --------------------------------------------------------------------------
def class_names_to_bytes(class_names) -> bytes:
    return json.dumps(class_names).encode("utf-8")


def model_to_bytes(model) -> bytes:
    """Serialize a Keras model to bytes (via a temp .keras file) for download."""
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as tmp:
        tmp_path = tmp.name
    model.save(tmp_path)
    with open(tmp_path, "rb") as f:
        data = f.read()
    os.remove(tmp_path)
    return data
