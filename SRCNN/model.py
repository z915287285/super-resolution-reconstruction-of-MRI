from utils import (
  read_data, 
  input_setup, 
  imsave,
  merge,
  merge_test,
  imread,
  merge_test_RGB
)

import time
import os
import matplotlib.pyplot as plt
import scipy.misc
import scipy.ndimage
from skimage.measure._structural_similarity import compare_ssim
from skimage.measure import compare_psnr
import numpy as np
import tensorflow as tf
import h5py
from skimage.color import ycbcr2rgb
from skimage.measure._structural_similarity import compare_ssim

try:
  xrange
except:
  xrange = range

class SRCNN(object):

  def __init__(self, 
               sess, 
               image_size=33,
               label_size=21, 
               batch_size=128,
               c_dim=1, 
               checkpoint_dir=None, 
               sample_dir=None,
               config=None):

    self.sess = sess
    self.is_grayscale = (c_dim == 1)
    self.image_size = image_size
    self.label_size = label_size
    self.batch_size = batch_size

    self.c_dim = c_dim
    self.config =config
    self.checkpoint_dir = checkpoint_dir
    self.sample_dir = sample_dir
    self.build_model()

  def build_model(self):
    self.images = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.c_dim], name='images')
    self.labels = tf.placeholder(tf.float32, [None, self.label_size, self.label_size, self.c_dim], name='labels')
    
    self.weights = {
      'w1': tf.Variable(tf.random_normal([9, 9, 1, 64], stddev=1e-3), name='w1'),
      'w2': tf.Variable(tf.random_normal([1, 1, 64, 32], stddev=1e-3), name='w2'),
      'w3': tf.Variable(tf.random_normal([5, 5, 32, 1], stddev=1e-3), name='w3')
    }
    self.biases = {
      'b1': tf.Variable(tf.zeros([64]), name='b1'),
      'b2': tf.Variable(tf.zeros([32]), name='b2'),
      'b3': tf.Variable(tf.zeros([1]), name='b3')
    }

    # with tf.name_scope('hidden_layer_one'):
    #   tf.summary.histogram('hidden_layer_one/weight', self.weights["w1"])
    #   tf.summary.histogram('hidden_layer_one/weight', self.biases["b1"])
    # with tf.name_scope('hidden_layer_two'):
    #   tf.summary.histogram('hidden_layer_one/weight', self.weights["w2"])
    #   tf.summary.histogram('hidden_layer_one/weight', self.biases["b1"])
    # with tf.name_scope('hidden_layer_three'):
    #   tf.summary.histogram('hidden_layer_one/weight', self.weights["w3"])
    #   tf.summary.histogram('hidden_layer_one/weight', self.biases["b3"])

    self.pred = self.model()

    # Loss function (MSE)
    self.loss = tf.reduce_mean(tf.square(self.labels - self.pred))

    # 用于psnr计算
    #self.psnr = tf.Variable(0.0)

    tf.summary.scalar('loss', self.loss)
    #tf.summary.scalar('PSNR', self.psnr)
    self.saver = tf.train.Saver()

  def train(self, config):
    #os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # if config.is_train:
    #   input_setup(self.sess, config)
    # else:
    #   nx, ny,loc = input_setup(self.sess, config)

    if config.is_train:     
      data_dir = os.path.join('./{}'.format(config.checkpoint_dir), "data_train_cmq.h5")
      train_data, train_label = read_data(data_dir,["train_inpt","train_labl"])
    else:
      data_dir = os.path.join('./{}'.format(config.checkpoint_dir), "data_test_cmq.h5")
      train_data, train_label, test_mask= read_data(data_dir,["test_train","test_label","test_mask"])


    # Stochastic gradient descent with the standard backpropagation
    self.train_op = tf.train.GradientDescentOptimizer(config.learning_rate).minimize(self.loss)



    # tensorboard
    merged = tf.summary.merge_all()  # 将图形、训练过程等数据合并在一起
    writer = tf.summary.FileWriter(os.getcwd() + '/logs', self.sess.graph)  # 将训练日志写入到logs文件夹下
    tf.initialize_all_variables().run()
    counter = 0

    start_time = time.time()

    if self.load(self.checkpoint_dir):
      print(" [*] Load SUCCESS")
    else:
      print(" [!] Load failed...")

    if config.is_train:
      print("Training...")
      #准备测试psnr的数据
      # data_dir = os.path.join('./{}'.format(config.checkpoint_dir), "data_test.h5")
      # test_data, test_label = read_data(data_dir, ["test_train", "test_label"])
      # with h5py.File(os.getcwd()+"/checkpoint/data_origin_246.h5", 'r') as hf:
      #   origin_image_246 = np.array(hf.get("origin_data_246"))
      # # 找到无关数据
      # zorelist = []
      # for i in range(origin_image_246.shape[0]):
      #   for j in range(origin_image_246.shape[1]):
      #     if origin_image_246[i][j] == 0.0:
      #       zorelist.append([i, j])


      for ep in xrange(config.epoch):
        # Run by batch images
        batch_idxs = len(train_data) // config.batch_size
        for idx in xrange(0, batch_idxs):
          batch_images = train_data[idx * config.batch_size: (idx + 1) * config.batch_size]
          batch_labels = train_label[idx * config.batch_size: (idx + 1) * config.batch_size]

          counter += 1
          _, err = self.sess.run([self.train_op, self.loss],
                                 feed_dict={self.images: batch_images, self.labels: batch_labels})
          if counter % 10 == 0:
            print("Epoch: [%2d], step: [%2d], time: [%4.4f], loss: [%.8f]" \
                  % ((ep + 1), counter, time.time() - start_time, err))

          if counter % 500 == 0:
            self.save(config.checkpoint_dir, counter)
            #计算psnr
            # self.image_size=258
            # self.label_size=246
            # result= self.pred.eval({self.images:test_data, self.labels:test_label})
            # self.image_size=33
            # self.label_size=21
            # result = result.reshape([246, 246])
            # for i in range(result.shape[0]):
            #   for j in range(result.shape[1]):
            #     if [i, j] in zorelist:
            #       result[i][j] = 0.0
            # self.psnr = tf.assign(self.psnr, compare_psnr(origin_image_246, result))
            # self.sess.run(self.psnr)
            summary, acc = self.sess.run([merged, self.loss],feed_dict={self.images: batch_images, self.labels: batch_labels})
            writer.add_summary(summary, counter)

    else:
      print("Testing...")

      #使用makeimagebychoose函数制作的数据，直接计算psnr
      # with h5py.File(os.getcwd()+"/checkpoint/data_test.h5", 'r') as hf:
      #     test_train = np.array(hf.get("test_train"))
      #     test_label = np.array(hf.get("test_label"))
      #     test_psnr = np.array(hf.get("test_psnr"))
      #
      # result = self.pred.eval({self.images: test_train, self.labels: test_label})
      # cont = 0
      # for i in range(result.shape[0]):
      #   label = test_label[i]
      #   label = label.reshape([21,21])
      #   train = result[i]
      #   train = train.reshape([21,21])
      #   srcnn_psnr=compare_psnr(label, train)
      #   if(srcnn_psnr > test_psnr[i]):
      #     cont=cont+1
      #     print("**********psnr srcnn:%.3f | bicubic:%.3f**********" % (srcnn_psnr,test_psnr[i]))
      #   else:
      #     print("psnr srcnn:%.3f | bicubic:%.3f" % (srcnn_psnr,test_psnr[i]))
      # print("超过数：%d/%d" % (cont, result.shape[0]))
      #结束

      #使用原始模型的测试方式 makeimage33()
      # with h5py.File(os.getcwd()+"/checkpoint/data_test.h5", 'r') as hf:
      #     test_train = np.array(hf.get("test_train"))
      #     test_label = np.array(hf.get("test_label"))
      #     mask = np.array(hf.get("mask"))
      #     loc = np.array((hf.get("loc")))
      #     nx = np.array((hf.get("nx")))
      #     ny = np.array((hf.get("ny")))
      # result = self.pred.eval({self.images: test_train, self.labels: test_label})
      # result = merge_test(result, [nx, ny, loc])
      # result = result.squeeze()
      # #result = result*mask
      # # 使用matplotlib
      # fig = plt.figure()
      # plt.suptitle("Test")
      #
      # # 原图
      # h, w = loc[len(loc) - 1][0] + 21, loc[len(loc) - 1][1] + 21
      # img = np.zeros((h, w, 1))
      # for i, image in enumerate(train_label):
      #     x, y = loc[i]
      #     img[x:x + 21, y:y + 21] = image
      # img = img.squeeze()
      #
      # a = fig.add_subplot(1, 3, 2)
      # plt.imshow(result, cmap="gray")
      # a.set_title("srcnn:%.4f" % compare_psnr(img, result))
      # print("SRCNN:%.4f ssim:%.4f" % (compare_psnr(img, result), compare_ssim(img, result)))
      #
      # a = fig.add_subplot(1, 3, 1)
      # plt.imshow(img, cmap="gray")
      # a.set_title("origin")
      #
      # imgorigin = img[2:, 2:]
      # # bicubic
      # img = scipy.ndimage.interpolation.zoom(img[2:, 2:], (1. / 3), prefilter=False)
      # img = scipy.ndimage.interpolation.zoom(img, (3 / 1.), prefilter=False)
      #
      # print("bicubic:%.4f ssim:%.4f" % (compare_psnr(imgorigin, img), compare_ssim(imgorigin, img)))
      #
      # a = fig.add_subplot(1, 3, 3)
      # plt.imshow(img, cmap="gray")
      # a.set_title("bicubic:%.4f" % compare_psnr(imgorigin, img))
      # plt.show()
      # print("getResult...")
      #结束


      #单个测试，去除背景输入网络
      # with h5py.File(os.getcwd()+"/checkpoint/singleimage.h5", 'r') as hf:
      #     trian_data = np.array(hf.get("trian_data"))
      #     train_label = np.array(hf.get("train_label"))
      #     bicubicpsinr = np.array((hf.get("bicubicpsinr")))
      # result = self.pred.eval({self.images: trian_data, self.labels: train_label})
      # result = result.reshape([result.shape[1], result.shape[2]])
      # train_label = train_label.reshape([train_label.shape[1], train_label.shape[2]])
      # print("srcnn%.4f  bicubic%.4f" % (compare_psnr(train_label, result),bicubicpsinr))
      # plt.figure()
      # plt.imshow(result, cmap="gray")
      # plt.show()
      # 单个测试结束


      #针对makeimageCutByHand函数的测试代码
      # with h5py.File(os.getcwd()+"/checkpoint/data_origin_246.h5", 'r') as hf:
      #   origin_image_246 = np.array(hf.get("origin_data_246"))
      # with h5py.File(os.getcwd()+"/checkpoint/data_origin_258.h5", 'r') as hf:
      #   origin_image_258 = np.array(hf.get("origin_data_258"))
      # count =0
      # sum =0
      # for i in range(train_data.shape[0]):
      #     result = train_data[i]
      #     result_label = train_label[i]
      #     maskimage = test_mask[i]
      #     result = result.reshape([1,result.shape[0], result.shape[1], 1])
      #     result_label = result_label.reshape([1,result_label.shape[0],result_label.shape[1], 1])
      #     result = self.pred.eval({self.images: result, self.labels: result_label})
      #     result = result.reshape([result.shape[1], result.shape[2]])
      #     result =result*maskimage
      #     train_image = train_data[i].reshape([train_data[i].shape[0], train_data[i].shape[1]])
      #     psnrsrcnn = compare_psnr(origin_image_246[i],result)
      #     psnrorigin = compare_psnr(origin_image_258[i],train_image)
      #
      #     plt.figure()
      #     plt.subplot(1, 3, 1)
      #     plt.imshow(origin_image_258[i], cmap="gray")
      #     plt.title("orgin:%d"% i)
      #
      #     plt.subplot(1, 3, 2)
      #     plt.imshow(result, cmap="gray")
      #     plt.title("srcnn:%.4f" % psnrsrcnn)
      #
      #
      #     plt.subplot(1, 3, 3)
      #     plt.imshow(train_image, cmap="gray")
      #     plt.title("bicubic:%.4f" % psnrorigin)
      #     plt.show()
      #
      #     if psnrsrcnn > psnrorigin:
      #       print("**************第%d张 srcnn:%.4f|bicubic:%.4f**************" %  (i, psnrsrcnn, psnrorigin))
      #       count= count+1
      #     else:
      #       print("第%d张 srcnn:%.4f|bicubic:%.4f "  % (i,psnrsrcnn,psnrorigin))
      #     sum=sum+1
      # print("超过数：",count,"/",sum)
      #测试代码结束


      #针对makeimageCut的测试代码
      with h5py.File(os.getcwd()+"/checkpoint/data_origin_339.h5", 'r') as hf:
        origin_image_246 = np.array(hf.get("origin_data_339"))
      with h5py.File(os.getcwd()+"/checkpoint/data_origin_351.h5", 'r') as hf:
        origin_image_258 = np.array(hf.get("origin_data_352"))
      count =0
      sum = 0
      bh_psnr =[]
      sh_psnr =[]
      bh_ssim = []
      sh_ssim = []
      for i in range(train_data.shape[0]):
          result = train_data[i]
          result_label = train_label[i]
          maskimage = test_mask[i]
          result = result.reshape([1,351, 351, 1])
          result_label = result_label.reshape([1,339,339,1])
          result = self.pred.eval({self.images: result, self.labels: result_label})
          result = result.reshape([339, 339])
          # for j in range(maskimage.shape[0]):
          #   for k in range(maskimage.shape[1]):
          #     if maskimage[j][k] == 0:
          #       result[j][k] = 0.0
          result =result*maskimage
          train_image = train_data[i].reshape([351, 351])
          psnrsrcnn = compare_psnr(origin_image_246[i],result)
          psnrorigin = compare_psnr(origin_image_258[i],train_image)
          ssimsrcnn = compare_ssim(origin_image_246[i], result)
          ssimorigin = compare_ssim(origin_image_258[i], train_image)
          sh_psnr.append(psnrsrcnn)
          sh_ssim.append(ssimsrcnn)
          bh_psnr.append(psnrorigin)
          bh_ssim.append(ssimorigin)
          # num = num + psnrsrcnn
          # plt.figure()
          # plt.subplot(1, 3, 1)
          # plt.imshow(origin_image_258[i], cmap="gray")
          # plt.title("orgin:%d" % i)
          #
          # plt.subplot(1, 3, 2)
          # plt.imshow(result, cmap="gray")
          # plt.title("srcnn:%.4f" % psnrsrcnn)
          #
          # plt.subplot(1, 3, 3)
          # plt.imshow(train_image, cmap="gray")
          # plt.title("bicubic:%.4f" % psnrorigin)
          # plt.show()

          if psnrsrcnn > psnrorigin:
            print("**************第%d张 srcnn:%.4f|bicubic:%.4f**************" %  (i, psnrsrcnn, psnrorigin))
            count= count+1
          else:
            print("第%d张 srcnn:%.4f|bicubic:%.4f "  % (i,psnrsrcnn,psnrorigin))
          sum=sum+1
      print("超过数：",count,"/",sum,"平均-psnr：",np.mean(sh_psnr),"平均-ssim：",np.mean(sh_ssim),"标准差:",np.std(sh_psnr,ddof=1),np.std(sh_ssim,ddof=1))
      print("bicubic平均-psnr：", np.mean(sh_psnr), "bicubic平均-ssim：", np.mean(sh_ssim),"标准差:",np.std(bh_psnr,ddof=1),np.std(bh_ssim,ddof=1))
      #结束


  def model(self):
    conv1 = tf.nn.relu(tf.nn.conv2d(self.images, self.weights['w1'], strides=[1,1,1,1], padding='VALID') + self.biases['b1'])
    conv2 = tf.nn.relu(tf.nn.conv2d(conv1, self.weights['w2'], strides=[1,1,1,1], padding='VALID') + self.biases['b2'])
    conv3 = tf.nn.conv2d(conv2, self.weights['w3'], strides=[1,1,1,1], padding='VALID') + self.biases['b3']
    return conv3

  def save(self, checkpoint_dir, step):
    model_name = "SRCNN.model"
    model_dir = "%s_%s" % ("srcnn", self.config.model)
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    self.saver.save(self.sess,
                    os.path.join(checkpoint_dir, model_name),
                    global_step=step)

  def load(self, checkpoint_dir):
    print(" [*] Reading checkpoints...")
    model_dir = "%s_%s" % ("srcnn", self.config.model)
    print(model_dir)
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
        return True
    else:
        return False
