#%%
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

#%%
print(tf.__version__)


#%%
# Initializing a tensor
x = tf.constant(3)
print(x)
# %%
x = tf.constant([[1, 2, 3], [4, 5, 6]])
print(x)
# %%
x = tf.ones((3, 3))
print(x)
# %%
x = tf.eye(3)
print(x)
# %%
x = tf.random.uniform((3, 3), minval=0, maxval=10)
print(x)
# %%
y = tf.cast(x, dtype=tf.int32)
print(y)
# %%
# Mathematical Operations
x = tf.constant([1, 2, 3])
y = tf.constant([9, 8, 7])

#%%
z = tf.add(x, y)
print(z)

# %%
z = tf.subtract(y, x)
print(z)

# %%
# Element wise multiplication
z = tf.multiply(x, y)
print(z)

# %%
# Element wise division
z = tf.divide(y, x)
print(z)

# %%
# Dot product
z = tf.tensordot(x, y, axes=1)
print(z)
z = tf.reduce_sum(x * y, axis=0)
print(z)

# %%
x = tf.random.normal((2, 3))
y = tf.random.normal((3, 4))
z = tf.matmul(x, y)
print(z)


# %%
# Indexing


x = tf.constant([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
print(x[:])  # print all elements

# %%
print(x[::2])  # print elemets but step is two

# %%
print(x[6:])  # print all elements starting from the given index

# %%
print(
    x[2:7]
)  # print elements starting from the first index but are below the second index

# %%
print(x[::-1])  # print the elments in revers order


# %%
indicies = tf.constant([0, 5, 8])  # declear the indicies of values you want to retrive

x_ind = tf.gather(
    x, indicies
)  # select values from your tensor at the specified indicies

#%%
print(x_ind)

# %%
# Reshaping

x = tf.range(9)
print(x)

# %%
y = tf.reshape(x, (3, 3))
print(y)

x = tf.shape(y)
print(x)

# %%
