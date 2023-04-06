#%%
# Import Packages
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.datasets import mnist

#%%
# Data Preprocessing
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = tf.cast(tf.reshape(x_train, (-1, 28 * 28)), dtype=tf.float32) / 255.0
x_test = tf.cast(tf.reshape(x_test, (-1, 28 * 28)), dtype=tf.float32) / 255.0

#%%
# Building the model

# Custom Layer
class CustomLayer(layers.Layer):
    def __init__(self, units):
        super(CustomLayer, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            name="w",
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,
        )
        self.b = self.add_weight(
            name="b",
            shape=(self.units,),
            initializer="zeros",
            trainable=True,
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b


class MyRelu(layers.Layer):
    def __init__(self):
        super(MyRelu, self).__init__()

    def call(self, x):
        return tf.math.maximum(x, 0)


class MyModel(tf.keras.Model):
    def __init__(self, num_classes=10):
        super(MyModel, self).__init__()
        self.dense1 = CustomLayer(64)
        self.dense2 = CustomLayer(num_classes)
        self.relu = MyRelu()

    # self.dense1 = layers.Dense(64)
    # self.dense2 = layers.Dense(num_classes)

    def call(self, input_tensor):
        x = self.relu(self.dense1(input_tensor))
        return self.dense2(x)


model = MyModel()

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=["accuracy"],
)
#%%
# Training the model
model.fit(x_train, y_train, batch_size=32, verbose=2, epochs=2)

#%%
# Testing the model
model.evaluate(x_test, y_test, batch_size=32, verbose=2)

# %%
