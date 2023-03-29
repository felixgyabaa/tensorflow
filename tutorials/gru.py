#%%
# import pakages
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
from keras import layers, regularizers
from keras.datasets import mnist

#%%
# Data Pre-processing
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = tf.cast(x_train, dtype=tf.float32) / 255.0
x_test = tf.cast(x_test, dtype=tf.float32) / 255.0


#%%
# Designing Model
model = tf.keras.Sequential()
model.add(tf.keras.Input(shape=(None, 28)))
model.add(layers.GRU(256, return_sequences=True, activation="tanh"))
model.add(layers.GRU(256, activation="tanh"))
model.add(layers.Dense(10))

print(model.summary())

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=["accuracy"],
)

#%%
# Training Model
model.fit(x_train, y_train, batch_size=64, epochs=10, verbose=2)

#%%
# Testing the model
model.evaluate(x_test, y_test, batch_size=64, verbose=2)
