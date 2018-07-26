import tensorflow as tf
import numpy as np

# def log1pexp(x):
#   return tf.log(1 + tf.exp(x))

@tf.custom_gradient
def log1pexp(x):
  e = tf.exp(x)
  def grad(dx):
    return dx * (1 - 1 / (1 + e))
  return tf.log(1 + e), grad

x_ = 100.

x = tf.placeholder(tf.float32)
y = log1pexp(x)


with tf.Session() as session:
    feed_dict = {x:x_}
    out = session.run(tf.gradients(y,x),feed_dict=feed_dict)
    gradient=out[0]

print(gradient)
