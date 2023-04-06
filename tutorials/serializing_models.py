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
x_train = tf.cast(tf.reshape(x_train, shape=(-1, 28 * 28)), dtype=tf.float32) / 255.0
x_test = tf.cast(tf.reshape(x_test, shape=(-1, 28 * 28)), dtype=tf.float32) / 255.0


#%%
# Building the Models

# Sequential Model
model1 = tf.keras.Sequential(
    [
        layers.Dense(64, activation="relu"),
        layers.Dense(10),
    ]
)

# Functional Model
inputs = tf.keras.Input(28 * 28)
x = layers.Dense(64, activation="relu")(inputs)
outputs = layers.Dense(10)(x)
model2 = tf.keras.Model(inputs=inputs, outputs=outputs)

# Model Subclassing
class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.dense1 = layers.Dense(64, activation="relu")
        self.dense2 = layers.Dense(10)

    def call(self, input_tensor):
        x = tf.nn.relu(self.dense1(input_tensor))
        return self.dense2(x)


model3 = MyModel()

model = model1

model.load_weights("saved_weights/")

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=["accuracy"],
)

#%%
# Train the Model
model.fit(x_train, y_train, batch_size=32, epochs=2, verbose=2)
model.save_weights("saved_weights/")

#%%
# Test the Model
model.evaluate(x_test, y_test, batch_size=32, verbose=2)
model.save("trained_model/")
# %%
