# -*- coding: utf-8 -*-
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_inference
import mnist_train
import numpy as np

# 每10秒加载一次最新的模型，并在测试数据上测试最新模型的正确率
EVAL_INTERVAL_SECS = 3

def evaluate(mnist, average=True):
    with tf.Graph().as_default() as g:
        # 定义输入数据和标签占位符
        x = tf.placeholder(tf.float32,shape=[None,mnist_inference.INPUT_NODE],name='x-input')                   
        y_ = tf.placeholder(tf.float32,shape=[None,mnist_inference.OUTPUT_NODE],name='y-input')
        reshaped_x = tf.reshape(x,[-1,mnist_inference.IMAGE_SIZE,mnist_inference.IMAGE_SIZE,mnist_inference.NUM_CHANNELS])
        # 准备验证数据
        vx = mnist.validation.images
        vy = mnist.validation.labels  
        validate_feed = {x:vx,y_:vy}
        # 计算前向传播结果
        y = mnist_inference.inference(x,False,None)
        # 计算正确率
        correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1)) #bool
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        if average==True:
            # 通过变量重命名的方式加载滑动平均模型：测试集准确率98.36%
            variable_averages = tf.train.ExponentialMovingAverage(mnist_train.MOVING_AVERAGE_DECAY)
            variables_to_restore = variable_averages.variables_to_restore()
            saver = tf.train.Saver(variables_to_restore)
        if average==False:
            # 加载不使用滑动平均的模型：测试集准确率98.22%
            saver = tf.train.Saver()
        # 每隔EVAL_INTERVAL_SECS秒计算一次正确率
        while True:
            with tf.Session() as sess:
                # 找到最新模型的文件名
                ckpt = tf.train.get_checkpoint_state(mnist_train.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    # 加载模型
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    # 通过文件名得到模型保存时迭代的轮数
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accuracy_score = sess.run(accuracy,feed_dict=validate_feed)
                    print('After %s training steps, validation accuracy = %g' %(global_step,accuracy_score))
                else:
                    print('No checkpoint file found')
                    return
                time.sleep(EVAL_INTERVAL_SECS)
                # 计算模型在测试集上的正确率
                if global_step==str(mnist_train.TRAINING_STEPS-1000+1):
                    tx = mnist.test.images
                    ty = mnist.test.labels
                    # 将输入的测试数据调整为一个四维矩阵              
                    test_feed = {x:tx,y_:ty}
                    accuracy_score = sess.run(accuracy,feed_dict=test_feed)
                    print('After %s training steps, test accuracy = %g' %(global_step,accuracy_score))
                    break
                          
def main(argv=None):
    mnist = input_data.read_data_sets("../../data/MNIST_data/",one_hot=True)
    evaluate(mnist)
                          
if __name__=='__main__':
    tf.app.run()
                              