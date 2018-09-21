# -*- coding: utf-8 -*-
import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_inference
import numpy as np

# 配置神经网络参数
BATCH_SIZE = 100
LEARNING_RATE = 0.001
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99
# 模型保存路径及文件名
MODEL_SAVE_PATH = './model/'
MODEL_NAME = 'model.ckpt'

def train(mnist):
    # 定义输入数据和标签占位符
    x = tf.placeholder(tf.float32,shape=[None,mnist_inference.INPUT_NODE],name='x-input')                   
    y_ = tf.placeholder(tf.float32,shape=[None,mnist_inference.OUTPUT_NODE],name='y-input')
    reshaped_x = tf.reshape(x,[-1,mnist_inference.IMAGE_SIZE,mnist_inference.IMAGE_SIZE,mnist_inference.NUM_CHANNELS])
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    global_step = tf.Variable(0,trainable=False)
    y = mnist_inference.inference(reshaped_x,True,regularizer)
    
     # 滑动平均结果
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    
    # loss
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y_,1),logits=y)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean+tf.add_n(tf.get_collection('losses'))
    
    # 优化
    train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss,global_step=global_step)
    
    # 一次完成多个操作
#     train_op = tf.group(train_step,variable_averages_op)
    with tf.control_dependencies([train_step,variable_averages_op]):
        train_op = tf.no_op(name='train')
        
    # 初始化Tensorflow持久化类
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # 初始化
        sess.run(tf.global_variables_initializer())
        for i in range(TRAINING_STEPS):
            # 准备训练数据
            xs,ys = mnist.train.next_batch(BATCH_SIZE)   
            train_feed = {x:xs,y_:ys}
            _, loss_value, step = sess.run([train_op,loss,global_step],feed_dict=train_feed)
            print('After %d training steps, loss on training batch is %g' %(step, loss_value))
            # 每1000轮保存一次模型
            if i%1000 == 0:             
                saver.save(sess, os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step=global_step)

def main(argv=None):
    mnist = input_data.read_data_sets("../../data/MNIST_data/",one_hot=True)
    train(mnist)

if __name__=='__main__':
    tf.app.run()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    