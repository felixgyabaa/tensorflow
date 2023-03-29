#%%
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.datasets import mnist

#%%
(x_train,y_train) , (x_test,y_test) = mnist.load_data()

# %%
print(tf.shape(x_train))
print(tf.shape(x_test.shape))
print('#'*30)
print(y_train.shape)
print(y_test.shape)


# %%
x_train= tf.cast(tf.reshape(x_train,(-1,28*28)),dtype=tf.float32) / 255.0
x_test= tf.cast(tf.reshape(x_test,(-1,28*28)),dtype=tf.float32) / 255.0

# %%
#Sequential API
model= tf.keras.Sequential()
model.add(tf.keras.Input(shape=(28*28)))
model.add(layers.Dense(512,activation='relu'))
model.add(layers.Dense(256,activation='relu'))
model.add(layers.Dense(10))

#%%
print(model.summary())

# %%
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy']
)

#%%
#FUNCTIONAL API
inputs = tf.keras.Input(shape=(28*28))
x=layers.Dense(512,activation='relu')(inputs)
x=layers.Dense(256,activation='relu')(x)
outputs=layers.Dense(10,activation='softmax')(x)

#%%
model=tf.keras.Model(inputs=inputs,outputs=outputs)

#%%
print(model.summary())
# %%
model.fit(x_train,y_train,batch_size=32,epochs=5,verbose=2)

#%%
model.evaluate(x_test,y_test,batch_size=32,verbose=2)

# %%
