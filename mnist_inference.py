# -*- coding: utf-8 -*-
import tensorflow as tf

# 配置神经网络结构
IMAGE_SIZE = 28
INPUT_NODE = 784
OUTPUT_NODE = 10
NUM_CHANNELS = 1
# 第一层卷积层
CONV1_SIZE = 5
CONV1_DEEP = 32
# 第二层卷积层
CONV2_SIZE = 5
CONV2_DEEP = 64
# 全连接层
FC_SIZE = 512

def inference(input_tensor, train, regularizer):
    # 28*28*1->28*28*32
    with tf.variable_scope('layer1-conv1'):
        weight = tf.get_variable('weight',[CONV1_SIZE,CONV1_SIZE,NUM_CHANNELS,CONV1_DEEP],initializer=tf.truncated_normal_initializer(stddev=0.1))
        bias = tf.get_variable('bias',[CONV1_DEEP],initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(input_tensor,weight,strides=[1,1,1,1],padding='SAME')
#         relu1 = tf.nn.relu(conv1+bias)
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1,bias))
        
    # 28*28*32->14*14*32
    with tf.name_scope('layer2-pool1'):
        pool1 = tf.nn.max_pool(relu1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
        
    # 14*14*32->14*14*64
    with tf.variable_scope('layer3-conv2'):
        weight = tf.get_variable('weight',[CONV2_SIZE,CONV2_SIZE,CONV1_DEEP,CONV2_DEEP],initializer=tf.truncated_normal_initializer(stddev=0.1))
        bias = tf.get_variable('bias',[CONV2_DEEP],initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(pool1,weight,strides=[1,1,1,1],padding='SAME')
#         relu2 = tf.nn.relu(conv2+bias)
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2,bias))
        
    # 14*14*64->7*7*64
    with tf.name_scope('layer4-pool2'):
        pool2 = tf.nn.max_pool(relu2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
        
    # 7*7*64->3136
    # 得到pool2每一维的大小:总共四维，第一维是batch size
    pool_shape = pool2.get_shape().as_list()
    nodes = pool_shape[1]*pool_shape[2]*pool_shape[3]
    reshaped = tf.reshape(pool2,[-1,nodes])
    
    # 3136->512
    with tf.variable_scope('layer5-fc1'):
        weight = tf.get_variable('weight',[nodes,FC_SIZE],initializer=tf.truncated_normal_initializer(stddev=0.1))
        # 只有全连接层的权重需要加入正则化
        if regularizer!=None:
            tf.add_to_collection('losses',regularizer(weight))
        bias = tf.get_variable('bias',[FC_SIZE],initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(reshaped,weight)+bias)
        if train:
            tf.nn.dropout(fc1,0.5)
            
    # 512->10
    with tf.variable_scope('layer6-fc2'):
        weight = tf.get_variable('weight',[FC_SIZE,OUTPUT_NODE],initializer=tf.truncated_normal_initializer(stddev=0.1))
        # 只有全连接层的权重需要加入正则化
        if regularizer!=None:
            tf.add_to_collection('losses',regularizer(weight))
        bias = tf.get_variable('bias',[OUTPUT_NODE],initializer=tf.constant_initializer(0.1))
        logit = tf.matmul(fc1,weight)+bias
    
    return logit
    
    