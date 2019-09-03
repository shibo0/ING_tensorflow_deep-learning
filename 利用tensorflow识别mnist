# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
#import scipy.misc
#import os
#import numpy as np

mnist= input_data.read_data_sets("D:项目dataset/tensorflow_mnist",one_hot=True)

x=tf.placeholder(tf.float32,[None,784])
y_=tf.placeholder(tf.float32,[None,10])

W=tf.Variable(tf.zeros([784,10]))
b=tf.Variable(tf.zeros([10]))
y=tf.nn.softmax(tf.matmul(x,W)+b)

cross=tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y)))
train_step=tf.train.GradientDescentOptimizer(0.01).minimize(cross)

#创建会话
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(1000):
        batch_xs,batch_ys=mnist.train.next_batch(100)
        sess.run(train_step,feed_dict={x:batch_xs,y_:batch_ys})
    print(sess.run(W))

    correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    print (sess.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels}))

