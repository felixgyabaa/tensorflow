#%%
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
from keras import layers, regularizers
from keras.datasets import cifar10

#%%
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# %%
x_train = tf.cast(x_train, dtype=tf.float32) / 255.0
x_test = tf.cast(x_test, dtype=tf.float32) / 255.0

#%%
def my_model():
    """Functional API"""
    inputs = tf.keras.Input(shape=(32, 32, 3))
    x = layers.Conv2D(32, 3, padding="same", kernel_regularizer=regularizers.l2(0.01))(
        inputs
    )
    x = layers.BatchNormalization()(x)
    x = tf.keras.activations.relu(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", kernel_regularizer=regularizers.l2(0.01))(
        x
    )
    x = layers.BatchNormalization()(x)
    x = tf.keras.activations.relu(x)
    x = layers.Conv2D(128, 3, padding="same", kernel_regularizer=regularizers.l2(0.01))(
        x
    )
    x = layers.BatchNormalization()(x)
    x = tf.keras.activations.relu(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(10, activation="softmax")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


#%%
model = my_model()
# %%
print(model.summary())

# %%
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=["accuracy"],
)

#%%
model.fit(x_train, y_train, batch_size=64, epochs=150, verbose=2)

# %%
model.evaluate(x_test, y_test, batch_size=64, verbose=1)
# %%
