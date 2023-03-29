#%%
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.datasets import cifar10

#%%
(x_train,y_train) , (x_test,y_test) = cifar10.load_data()

# %%
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# %%
##Sequencial API
model = tf.keras.Sequential(
    [
        tf.keras.Input(shape=(32, 32, 3)),
        layers.Conv2D(32, 3,activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10)
    ] 
)


#%%
def my_model():
    """Functional API"""
    inputs = tf.keras.Input(shape=(32, 32, 3))
    x = layers.Conv2D(32, 3)(inputs)
    x = layers.BatchNormalization()(x)
    x = tf.keras.activations.relu(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = tf.keras.activations.relu(x)
    x = layers.Conv2D(128, 3)(x)
    x = layers.BatchNormalization()(x)
    x = tf.keras.activations.relu(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(10,activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs,outputs=outputs)
    return model

#%%
model =my_model()
# %%
print(model.summary())

# %%
model.compile(
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    optimizer = keras.optimizers.Adam(learning_rate =0.001),
    metrics=["accuracy"]
)

#%%
model.fit(x_train, y_train, batch_size = 64, epochs=10, verbose= 2)

# %%
model.evaluate(x_test, y_test, batch_size = 64, verbose = 1)
# %%
