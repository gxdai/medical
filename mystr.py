import tensorflow as tf

x = tf.placeholder(tf.string)

y =  x + 'abc'

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer)

	for _ in range(5):
		print(sess.run(y, feed_dict={x: ['abc', 'cde']}))