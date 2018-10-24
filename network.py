# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import random


class trakerCnn:
    def __init__(self):
        self.LEARNING_RATE = 1e-4
        '''
        self.img1 = tf.placeholder(tf.float32,[1,720,1280, 3])
        self.img2 = tf.placeholder(tf.float32,[1,720,1280, 3])
        self.img1_box = tf.placeholder(tf.float32, [1,720,1280, 1])
        self.img2_box = tf.placeholder(tf.float32, [1,720,1280, 1])
        '''
        self.img1 = tf.placeholder(tf.float32,[None,None,None, None])
        self.img2 = tf.placeholder(tf.float32,[None,None,None, None])
        self.img1_box = tf.placeholder(tf.float32, [None,None,None, None])
        self.img2_box = tf.placeholder(tf.float32, [None,None,None, None])
        
        self.img2_box_pre = self.inf(self.img1,self.img2,self.img1_box)

        self.loss = tf.sqrt(tf.reduce_mean(tf.square(self.img2_box - self.img2_box_pre)))

        self.train_step = tf.train.AdamOptimizer(self.LEARNING_RATE).minimize(self.loss)

    def data_pre_train(self):
        img1_path = './data/img1/'
        img2_path = './data/img2/'
        img1_box_path = './data/img1_box/'
        img2_box_path = './data/img2_box/'
        
        
        
        print('loading data from dataset...data_pre_train')
        img_names = os.listdir(img1_path)
        img_num = len(img_names)

        data = []
        for i in range(img_num):
        
            img1 = img1_path + str(i) + '.jpg'
            img2 = img2_path + str(i) + '.jpg'
            img1_box = img1_box_path + str(i) + '.jpg'
            img2_box = img2_box_path + str(i) + '.jpg'
            
            img1 = np.array(Image.open(img1))
            img2 = np.array(Image.open(img2))
            img1_box = np.array(Image.open(img1_box))
            
            img2_box = Image.open(img2_box)
            #img2_box = img2_box.resize([img2_box.width/4,img2_box.height/4])
            img2_box = np.array(img2_box)
            data.append([img1, img2,img1_box,img2_box,img2_box_path + str(i) + '.jpg']) 
            
        return data
     

    def data_pre_test(self):
        img1_path = './data/img1/'
        img2_path = './data/img2/'
        img1_box_path = './data/img1_box/'
        img2_box_path = './data/img2_box/'
        
        
        
        print('loading data from dataset...data_pre_test')
        img_names = os.listdir(img1_path)
        img_num = len(img_names)

        data = []
        for i in range(100):
        
            img1 = img1_path + str(i) + '.jpg'
            img2 = img2_path + str(i) + '.jpg'
            img1_box = img1_box_path + str(i) + '.jpg'
            img2_box = img2_box_path + str(i) + '.jpg'

            img1 = np.array(Image.open(img1))
            img2 = np.array(Image.open(img2))
            img1_box = np.array(Image.open(img1_box))
            
            img2_box = Image.open(img2_box)
            img2_box = img2_box.resize([img2_box.width/4,img2_box.height/4])
            img2_box = np.array(img2_box)
            data.append([img1, img2,img1_box,img2_box,img2_box_path + str(i) + '.jpg']) 
            
        return data
    
    
    def conv2d(self, x, w):
        return tf.nn.conv2d(x, w, strides = [1, 1, 1, 1], padding = 'SAME')

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
    
    def inf(self,img1,img2,img1_box):
        
        #img1
        w_conv1_1 = tf.get_variable('w_conv1_1', [3, 3, 3, 24])
        b_conv1_1 = tf.get_variable('b_conv1_1', [24])
        h_conv1_1 = tf.nn.relu(self.conv2d(img1, w_conv1_1) + b_conv1_1)
        h_pool1_1 = self.max_pool_2x2(h_conv1_1)

        w_conv2_1 = tf.get_variable('w_conv2_1', [3, 3, 24, 48])
        b_conv2_1 = tf.get_variable('b_conv2_1', [48])
        h_conv2_1 = tf.nn.relu(self.conv2d(h_pool1_1, w_conv2_1) + b_conv2_1)
        h_pool2_1 = self.max_pool_2x2(h_conv2_1)

        w_conv3_1 = tf.get_variable('w_conv3_1', [3, 3, 48, 24])
        b_conv3_1 = tf.get_variable('b_conv3_1', [24])
        h_conv3_1 = tf.nn.relu(self.conv2d(h_pool2_1, w_conv3_1) + b_conv3_1)

        w_conv4_1 = tf.get_variable('w_conv4_1', [3, 3, 24, 12])
        b_conv4_1 = tf.get_variable('b_conv4_1', [12])
        h_conv4_1 = tf.nn.relu(self.conv2d(h_conv3_1, w_conv4_1) + b_conv4_1)
        
        #img2
        w_conv1_2 = tf.get_variable('w_conv1_2', [3, 3, 3, 24])
        b_conv1_2 = tf.get_variable('b_conv1_2', [24])
        h_conv1_2 = tf.nn.relu(self.conv2d(img2, w_conv1_2) + b_conv1_2)
        h_pool1_2 = self.max_pool_2x2(h_conv1_2)

        w_conv2_2 = tf.get_variable('w_conv2_2', [3, 3, 24, 48])
        b_conv2_2 = tf.get_variable('b_conv2_2', [48])
        h_conv2_2 = tf.nn.relu(self.conv2d(h_pool1_2, w_conv2_2) + b_conv2_2)
        h_pool2_2 = self.max_pool_2x2(h_conv2_2)

        w_conv3_2 = tf.get_variable('w_conv3_2', [3, 3, 48, 24])
        b_conv3_2 = tf.get_variable('b_conv3_2', [24])
        h_conv3_2 = tf.nn.relu(self.conv2d(h_pool2_2, w_conv3_2) + b_conv3_2)

        w_conv4_2 = tf.get_variable('w_conv4_2', [3, 3, 24, 12])
        b_conv4_2 = tf.get_variable('b_conv4_2', [12])
        h_conv4_2 = tf.nn.relu(self.conv2d(h_conv3_2, w_conv4_2) + b_conv4_2)
        
        #img1_box
        w_conv1_3 = tf.get_variable('w_conv1_3', [3, 3, 1, 24])
        b_conv1_3 = tf.get_variable('b_conv1_3', [24])
        h_conv1_3 = tf.nn.relu(self.conv2d(img1_box, w_conv1_3) + b_conv1_3)
        h_pool1_3 = self.max_pool_2x2(h_conv1_1)

        w_conv2_3 = tf.get_variable('w_conv2_3', [3, 3, 24, 48])
        b_conv2_3 = tf.get_variable('b_conv2_3', [48])
        h_conv2_3 = tf.nn.relu(self.conv2d(h_pool1_3, w_conv2_3) + b_conv2_3)
        h_pool2_3 = self.max_pool_2x2(h_conv2_3)
        

        w_conv3_3 = tf.get_variable('w_conv3_3', [3, 3, 48, 24])
        b_conv3_3 = tf.get_variable('b_conv3_3', [24])
        h_conv3_3 = tf.nn.relu(self.conv2d(h_pool2_3, w_conv3_3) + b_conv3_3)

        w_conv4_3 = tf.get_variable('w_conv4_3', [3, 3, 24, 12])
        b_conv4_3 = tf.get_variable('b_conv4_3', [12])
        h_conv4_3 = tf.nn.relu(self.conv2d(h_conv3_3, w_conv4_3) + b_conv4_3)
        
        
        print(h_conv4_1)
        print(h_conv4_2)
        print(h_conv4_3)
        #几个图层合并
        h_conv4_merge = tf.concat([h_conv4_1, h_conv4_2, h_conv4_3], 3)
        print(h_conv4_merge)
        
        w_conv5 = tf.get_variable('w_conv5', [3, 3, 36, 64])
        b_conv5 = tf.get_variable('b_conv5', [64])
        h_conv5 = tf.nn.relu(self.conv2d(h_conv4_merge, w_conv5) + b_conv5)
        
        #加入反卷积
        w_conv6 = tf.get_variable('w_conv6', [4, 4, 32, 64])
        deconv_6 = tf.nn.conv2d_transpose(value=h_conv5 , filter=w_conv6, output_shape=[1,tf.shape(img1_box)[1],tf.shape(img1_box)[2],32],
                                            strides=[1,4,4,1], padding='VALID')
        
        
        
        # box单独一个
        w_conv1_box = tf.get_variable('w_conv1_box', [3, 3, 1, 4])
        b_conv1_box = tf.get_variable('b_conv1_box', [4])
        h_conv1_box = tf.nn.relu(self.conv2d(img1_box, w_conv1_box) + b_conv1_box)
        
        w_conv2_box = tf.get_variable('w_conv2_box', [3, 3, 4, 1])
        b_conv2_box = tf.get_variable('b_conv2_box', [1])
        h_conv2_box = tf.nn.relu(self.conv2d(h_conv1_box, w_conv2_box) + b_conv2_box)
        
        
        print('deconv_6',deconv_6)
        print(h_conv2_box)
        #再合并原图的box                        
        h_conv7_merge = tf.concat([deconv_6, h_conv2_box], 3)
        print(h_conv7_merge)
        
        w_conv7 = tf.get_variable('w_conv7', [3, 3, 33, 16])
        b_conv7 = tf.get_variable('b_conv7', [16])
        h_conv7 = tf.nn.relu(self.conv2d(h_conv7_merge, w_conv7) + b_conv7)
        
        w_conv8 = tf.get_variable('w_conv8', [3, 3, 16, 8])
        b_conv8 = tf.get_variable('b_conv8', [8])
        h_conv8 = tf.nn.relu(self.conv2d(h_conv7, w_conv8) + b_conv8)
        
        w_conv9 = tf.get_variable('w_conv9', [1, 1, 8, 1])
        b_conv9 = tf.get_variable('b_conv9', [1])
        h_conv9 = tf.nn.relu(self.conv2d(h_conv8, w_conv9) + b_conv9)
        
        
        return h_conv9
        
    def train(self):    
        with tf.Session() as sess:
            
            try:
                saver = tf.train.Saver()
                saver.restore(sess, './model/model.ckpt')
                data_train = self.data_pre_test()
            except:
                sess.run(tf.global_variables_initializer())
            
            #sess.run(tf.global_variables_initializer())
            
            merged = tf.summary.merge_all()
            writer = tf.summary.FileWriter("logs/", sess.graph)
            
            data_train = self.data_pre_train()
           
            max_epoch = 1000
            random.shuffle(data_train)
            for epoch in range(max_epoch):
                for i in range(len(data_train)):
                
                    data = data_train[i]
                    
                    img1 = np.reshape(data[0], (1, data[0].shape[0], data[0].shape[1], 3))
                    img2 = np.reshape(data[1], (1, data[1].shape[0], data[1].shape[1], 3))
                    img1_box = np.reshape(data[2], (1, data[2].shape[0], data[2].shape[1], 1))
                    img2_box = np.reshape(data[3], (1, data[3].shape[0], data[3].shape[1], 1))
                    _, _loss = sess.run( [self.train_step, self.loss], 
                        feed_dict = {self.img1: img1, self.img2: img2, self.img1_box: img1_box, self.img2_box: img2_box})
                    print(epoch,i,_loss)
                    
                    if i % 500 == 0:        
                        print('epoch', epoch, 'step', i)
                        saver = tf.train.Saver()
                        saver.save(sess, './model/model.ckpt')
            writer.close()       
    def test(self):
        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, './model/model.ckpt')
            data_train = self.data_pre_test()
            
            for i in range(1):
                
                data = data_train[i]
                print(data[4])
                img1 = np.reshape(data[0], (1, data[0].shape[0], data[0].shape[1], 3))
                img2 = np.reshape(data[1], (1, data[1].shape[0], data[1].shape[1], 3))
                img1_box = np.reshape(data[2], (1, data[2].shape[0], data[2].shape[1], 1))
                img2_box = np.reshape(data[3], (1, data[3].shape[0], data[3].shape[1], 1))
                
                y_p_den = sess.run(self.img2_box_pre, feed_dict = {self.img1: img1, self.img2: img2, self.img1_box: img1_box})
                
                y_p_den = np.reshape(y_p_den, (y_p_den.shape[1], y_p_den.shape[2]))
                
                '''
                # 对于小于0的值 安排一下
                for x in range(0, y_p_den.shape[0]):
                    for y in range(0, y_p_den.shape[1]):
                        if y_p_den[x, y] < 0:
                            y_p_den[x, y] = 0
                '''            
                img = Image.fromarray(y_p_den.astype('uint8')).convert('L')
                
                img.show()
                img2 = Image.fromarray(data[1].astype('uint8')).convert('L')
                img1 = Image.fromarray(data[0].astype('uint8')).convert('L')
                result = Image.fromarray((np.asarray(img)+np.asarray(img1)).astype('uint8')).convert('L') 
                
                result.show()


                
    def testone(self,index):
        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, './model/model.ckpt')
            
            img1 = Image.open('./data/img1/' + str(index) + '.jpg','r')
            img1 = np.asarray(img1)
            img1_ori = img1
            img2 = Image.open('./data/img2/' + str(index) + '.jpg','r')
            img2 = np.asarray(img2)
            img2_ori = img2
            img1_box = Image.open('./data/img1_box/' + str(index) + '.jpg','r')
            img1_box = np.asarray(img1_box)
            
            img1 = np.reshape(img1, (1, img1.shape[0], img1.shape[1], 3))
            img2 = np.reshape(img2, (1, img2.shape[0], img2.shape[1], 3))
            img1_box = np.reshape(img1_box, (1, img1_box.shape[0], img1_box.shape[1], 1))
            
            y_p_den = sess.run(self.img2_box_pre, feed_dict = {self.img1: img1, self.img2: img2, self.img1_box: img1_box})
                
            y_p_den = np.reshape(y_p_den, (y_p_den.shape[1], y_p_den.shape[2]))
            
            # 对于小于0的值 安排一下
            for x in range(0, y_p_den.shape[0]):
                for y in range(0, y_p_den.shape[1]):
                    if y_p_den[x, y] < 0:
                        y_p_den[x, y] = 0
                        
            img = Image.fromarray(y_p_den.astype('uint8')).convert('L')
            
            img.show()
            img2 = Image.fromarray(img2_ori.astype('uint8')).convert('L')
            result = Image.fromarray((np.asarray(img)+np.asarray(img2)/3.0).astype('uint8')).convert('L') 
            result.show()
