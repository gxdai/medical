"""
tf.data

There are three steps

Importing data: creating a Dataset instance from some data.
Generate an Iterator: By using the created dataset to make an iterator
Consuming data: by using the created iterator we can get the elements
"""


# importing data from numpy

import numpy as np
import tensorflow as tf


x = np.random.sample((100, 2))
# make a dataset from a numpy array
dataset_x = tf.data.Dataset.from_tensor_slices(x)
print(dataset_x)

# creates paired data (feature, label)
features, labels = np.random.rand(100,2), np.random.rand(100, 1)
dataset = tf.data.Dataset.from_tensor_slices((features, labels))
print(dataset)

# From a tensor
dataset = tf.data.Dataset.from_tensor_slices(tf.random_uniform([100, 2]))
print(dataset)

# use placeholder, we could dynamically change the data
x = tf.placeholder(tf.float32, shape=[None, 2])
dataset = tf.data.Dataset.from_tensor_slices(x)
print(dataset)

# create iterator

# One shot:		you can not feed any value to it
# reinitializable: Initialized from different Datasets for additional transformations.
# feedable: It can be used to select iterator to use.

"""
# make one shot iterator
iter = dataset_x.make_one_shot_iterator()
el = iter.get_next()

with tf.Session() as sess:
	for _ in range(1000):
		try:
			print(sess.run(el))
		except tf.errors.OutOfRangeError:
			break
"""
# initializable iterator
x = tf.placeholder(tf.float32, shape=[None, 2])
dataset = tf.data.Dataset.from_tensor_slices(x)

data = np.random.sample((100, 2))
iter = dataset.make_initializable_iterator()
el = iter.get_next()

with tf.Session() as sess:
	sess.run(iter.initializer, feed_dict={x: data})			# reinitialize the iterator
	for _ in range(1000):
		try:
			print(sess.run(el))
		except tf.errors.OutOfRangeError:
			break


# real example for iterator

epoch = 10
x, y = tf.placeholder(tf.float32, shape=[None, 2]), tf.placeholder(tf.float32, [None, 1])
# create dataset
dataset = tf.data.Dataset.from_tensor_slices((x,y))

train_data = (np.random.rand(100, 2), np.random.rand(100, 1))
test_data = (np.random.rand(100, 2), np.random.rand(100, 1))

iter = dataset.make_initializable_iterator()
features, labels = iter.get_next()


with tf.Session() as sess:
	# initialize iter
	sess.run(iter.initializer, feed_dict={x: train_data[0], y:train_data[1]})
	for _ in range(epoch):
		print(sess.run([features, labels]))
	print("switch to test data")
	sess.run(iter.initializer, feed_dict={x:test_data[0], y:test_data[1]})
	print(sess.run([features, labels]))


# Reinitializable iterator

# prepare data
train_data = (np.random.rand(100, 2), np.random.rand(100,1))
test_data = (np.random.rand(100, 2), np.random.rand(100,1))
# create dataset
train_dataset = tf.data.Dataset.from_tensor_slices(train_data)
test_dataset = tf.data.Dataset.from_tensor_slices(test_data)

# Here is the trick to create a generic iterator
iter = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)

# two init ops
train_init_op = iter.make_initializer(train_dataset)
test_init_op = iter.make_initializer(test_dataset)

# Get the next elements
features_, labels_ = iter.get_next()
with tf.Session() as sess:
	sess.run(train_init_op)
	for _ in range(epoch):
		print(sess.run([features_, labels_]))
	sess.run(test_init_op)
	print(sess.run([features_, labels_]))

