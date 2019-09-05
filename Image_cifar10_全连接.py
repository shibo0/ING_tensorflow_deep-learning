from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import time
import pickle

# 加载和准备数据集
beginTime = time.time()
# 定义参数
batch_size = 100
# learning_rate = 0.005
learning_rate = 0.005
max_steps = 10000


# 读取一批次数据
def load_batch(batch):
    with open(batch, "rb") as fo:
        datadict = pickle.load(fo,encoding="latin1")
        data = datadict["data"]
        labels = datadict["labels"]
        # 拆分为10000组图像，3个通道
        # data = np.reshape(data, (10000, 3, 32, 32))
        data = np.reshape(data, (10000, 3072))
        labels = np.array(labels)
        return data, labels


# 读取整合数据
def load_data(src):
    datasets = {"images_train":[], "labels_train":[],"images_test":[], "labels_test":[]}
    for b in range(1, 6):
        data, labels = load_batch("{0}data_batch_{1}".format(src, b))
        datasets["images_train"].append(data)
        datasets["labels_train"].append(labels)
    datasets["images_train"] = np.concatenate(datasets["images_train"])
    datasets["labels_train"] = np.concatenate(datasets["labels_train"])
    datasets["images_test"], datasets["labels_test"]  = load_batch("{0}test_batch".format(src))
    return datasets
# 加载数据

# 数据集的目录，如果出现no file提示，多半是目录问题
# 在vscode下面目录是以工程根目录开始算的
data_sets = load_data("D:/dataset/21个项目-tensorflow/cifar-10-batches-py/")

# 定义占位符
images_placeholder = tf.placeholder(tf.float32, shape=[None, 3072])
labels_placeholder = tf.placeholder(tf.int64, shape=[None])

# 定义变量（希望优化的值）
weights = tf.Variable(tf.zeros([3072, 10]))
biases = tf.Variable(tf.zeros([10]))

# 定义分类的结果
logits = tf.matmul(images_placeholder, weights) + biases

# 定义损失函数
loss = tf.reduce_mean(
    tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels_placeholder))

# 定义训练操作
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# 新的函数
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# 迭代训练
for i in range(max_steps):
    # 从data_sets中随机抽取一批图片 batch_size = 100
    indices = np.random.choice(data_sets['images_train'].shape[0], batch_size)
    images_batch = data_sets['images_train'][indices]
    labels_batch = data_sets['labels_train'][indices]

    sess.run(train_step, feed_dict={
             images_placeholder: images_batch, labels_placeholder: labels_batch})

# 和正确的标签比较
correct_prediction = tf.equal(tf.argmax(logits, 1), labels_placeholder)
# 计算准确性
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(sess.run(accuracy, feed_dict={
    images_placeholder: data_sets['images_test'], labels_placeholder: data_sets['labels_test']
}))
