# -*- coding: utf-8 -*-


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#加载data集
mnist= input_data.read_data_sets("D:项目dataset/tensorflow_mnist",one_hot=True)

x=tf.placeholder(tf.float32,[None,784])
y_=tf.placeholder(tf.float32,[None,10])

#将单张图片从 784 维向量重新还原为 28×28 的矩阵图片 
x_image = tf.reshape(x,[-1, 28, 28, 1])

def weight_variable(shape):
    #tf.truncated_normal(shape, mean, stddev) :shape表示生成张量的维度，mean是均值，stddev是标准差。
    #这个函数产生正太分布，均值和标准差自己设定
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

"""
tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, name=None)
除去name参数用以指定该操作的name，与方法有关的一共五个参数：
第一个参数input：指需要做卷积的输入图像，它要求是一个Tensor，
具有[batch, in_height, in_width, in_channels]这样的shape，
具体含义是[训练时一个batch的图片数量, 图片高度, 图片宽度, 图像通道数]，
注意这是一个4维的Tensor，要求类型为float32和float64其中之一
第二个参数filter：相当于CNN中的卷积核，它要求是一个Tensor，
具有[filter_height, filter_width, in_channels, out_channels]这样的shape，
具体含义是[卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]，
要求类型与参数input相同，有一个地方需要注意，第三维in_channels，就是参数input的第四维
第三个参数strides：卷积时在图像每一维的步长，这是一个一维的向量，长度4
第四个参数padding：string类型的量，只能是"SAME","VALID"其中之一，这个值决定了不同的卷积方式（后面会介绍）
第五个参数：use_cudnn_on_gpu:bool类型，是否使用cudnn加速，默认为true
结果返回一个Tensor，这个输出，就是我们常说的feature map，shape仍然是[batch, height, width, channels]这种形式。"""
def conv2d(x,w):
    return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME')

"""
tf.nn.max_pool(value, ksize, strides, padding, name=None)
参数是四个，和卷积很类似：
第一个参数value：需要池化的输入，一般池化层接在卷积层后面，
所以输入通常是feature map，依然是[batch, height, width, channels]这样的shape
第二个参数ksize：池化窗口的大小，取一个四维向量，一般是[1, height, width, 1]，
因为我们不想在batch和channels上做池化，所以这两个维度设为了1
第三个参数strides：和卷积类似，窗口在每一个维度上滑动的步长，一般也是[1, stride,stride, 1]
第四个参数padding：和卷积类似，可以取'VALID' 或者'SAME'
返回一个Tensor，类型不变，shape仍然是[batch, height, width, channels]这种形式"""
def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],
                          strides=[1,2,2,1],padding='SAME')

#第一层卷积
w_conv1=weight_variable([5,5,1,32])
b_conv1=bias_variable([32])
h_conv1=tf.nn.relu(conv2d(x_image,w_conv1)+b_conv1)
h_pool1=max_pool_2x2(h_conv1)
    
#第二层卷积
w_conv2=weight_variable([5,5,32,64])
b_conv2=bias_variable([64])
h_conv2=tf.nn.relu(conv2d(h_pool1,w_conv2)+b_conv2)
h_pool2=max_pool_2x2(h_conv2)

#两层卷积之后是全连接
w_fc1=weight_variable([7*7*64,1024])
b_fc1=bias_variable([1024])
h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*64])
h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,w_fc1)+b_fc1)
#使用 Dropout, keep _prob 是－个占位符，训练时为 0.5，测试时为 1
#加入dropout,防止过拟合在全连接层中加入了 Dropout，
#它是防止神经网络过拟合的一种手段。在每一步训练时，
#以一定概率“去悼”网络中的某些连接，但这种去除不是永久性的，
#只是在当前步骤中去除，并且每一步去除的连接都是随机选择的 
keep_prob=tf.placeholder(tf.float32)
h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)

#把1024维的向量转换成10维，对应10个类别
w_fc2=weight_variable([1024,10])
b_fc2=bias_variable([10])
y_conv=tf.matmul(h_fc1_drop,w_fc2)+b_fc2

cross_entropy=tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y_conv))

train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction=tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

#会话
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(20000):
        batch=mnist.train.next_batch(50)
        #每100步报告一次在验证集上的准确率
        if i %100==0:
            train_accuracy=accuracy.eval(feed_dict={
                    x:batch[0],y_:batch[1],keep_prob:1.0})
            print("step %d, training accuracy %g"%(i,train_accuracy))
        train_step.run(feed_dict={x:batch[0],y_:batch[1],keep_prob:0.5})
    print("----------------------------")
    print("test accuracy %g" % accuracy.eval(feed_dict={
            x:mnist.test.images,y_:mnist.test.labels,keep_prob:1.0}))    












