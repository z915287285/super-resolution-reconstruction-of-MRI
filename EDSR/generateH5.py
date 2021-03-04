import numpy as np
import SimpleITK as sitk
import os
import h5py
import cv2
import random
import glob
import scipy.ndimage
import scipy.misc
data_dir="T1WINC"
img_files = os.listdir(data_dir)
path="data1w"
# if not os.path.exists(path):
#     os.makedirs(path)
# dataset = h5py.File(os.path.join(path, "data.h5"), 'w')
#
# image_size=99
# scale=3
#
# d_imgshape = (None, image_size/scale, image_size/scale, 1)
# d_labelshape = (None, image_size, image_size, 1)
#
# dataset.create_dataset('train_input', d_imgshape, dtype='f')
# dataset.create_dataset('train_label', d_labelshape, dtype='f')
# dataset.create_dataset('val_input', d_imgshape, dtype='f')
# dataset.create_dataset('val_label', d_labelshape, dtype='f')

#
# img_input=[]
# img_label=[]


isTrain=True

if isTrain:

    image_size = 66
    scale = 3

    img_input = []
    img_label = []
    # image = np.empty((1,33,33,1))
    # label = np.empty((1,99,99,1))


    for img in img_files:
        print("当前：{}".format(img))
        tmp_data = os.path.join(data_dir, img) + "\*_brain.nii.gz"
        tmp_mask = os.path.join(data_dir, img) + "\*_brain_mask.nii.gz"
        li_data = glob.glob(tmp_data)
        li_mask = glob.glob(tmp_mask)
        tmp = sitk.ReadImage(li_data[0])
        tmp1 = sitk.ReadImage(li_mask[0])
        data = sitk.GetArrayFromImage(tmp)
        m = sitk.GetArrayFromImage(tmp1)
        # data = np.transpose(data, (2, 1, 0))
        # data = data[0:300, 0:258, 0:258]
        # m = np.transpose(m, (1, 2, 0))
        # mask = m[0:300, 0:258, 0:258]
        max_inp = np.ones_like(data) * np.max(data)
        min_inp = np.min(data)
        inputs_1 = (data - min_inp) / (max_inp - min_inp)
        count = 0
        # get image from slices
        for i in range(data.shape[0]):
            if count>300 or i==data.shape[0]-1 and len(img_input)!=0:
                img_input1 = np.asarray(img_input)
                img_label1 = np.asarray(img_label)
                img_input = []
                img_label = []
                n = len(img_input1)
                train_size = n * 0.7
                print(n)
                print(train_size)
                train_data_input = img_input1[:int(train_size)]
                test_data_input = img_input1[int(train_size):]
                train_data_label = img_label1[:int(train_size)]
                test_data_label = img_label1[int(train_size):]
                permutation = np.random.permutation(train_data_input.shape[0])
                shuffled_input_train = train_data_input[permutation, :, :, :]
                shuffled_label_train = train_data_label[permutation, :, :, :]
                permutation1 = np.random.permutation(test_data_input.shape[0])
                shuffled_input_test = test_data_input[permutation1, :, :, :]
                shuffled_label_test = test_data_label[permutation1, :, :, :]

                with h5py.File(os.path.join('data1w', "data_train"+"_"+img+".h5"), 'w') as hf:
                    hf.create_dataset('train_input', data=shuffled_input_train)
                    hf.create_dataset('train_label', data=shuffled_label_train)
                    hf.create_dataset('val_input', data=test_data_input)
                    hf.create_dataset('val_label', data=test_data_label)
                # if img == img_files[0]:
                #     image = img_input1
                #     label = img_label1
                # else:
                #     image = np.concatenate((image,img_input1))
                #     label = np.concatenate((label,img_label1))
                    # image = image +img_input1
                    # label = label +img_label1
                break
            mask_1 = m[i:i + 1, :, :]
            mask_zero = (mask_1 == 0)
            num_zero = np.count_nonzero(mask_zero)
            rate_mask_zero = 1.0 * num_zero / (mask_1.shape[1] * mask_1.shape[2])
            if rate_mask_zero<1:
                index=np.where(mask_1)
                H_image = inputs_1[i:i + 1, :, :]
                count1 = 0
                for j in range(0,len(index[0]),40):
                    local1=int(index[1][j])
                    local2=int(index[2][j])
                    image_input=H_image[:,int(local1-image_size/2):int(local1+image_size/2), \
                                int(local2-image_size/2):int(local2+image_size/2)]
                    image_input = np.squeeze(image_input)
                    Iszero=(image_input==0)
                    num_zero=np.count_nonzero(Iszero)
                    rate_zero=1.0*num_zero/(image_size*image_size)
                    if rate_zero<0.4 and image_input.shape[0]==image_size and image_input.shape[1]==image_size:
                        if count1>4:
                            break
                        else:
                            print (image_input.shape[0],image_input.shape[1])
                            print("满足条件")
                            image1=image_input
                            # image1_inp = scipy.ndimage.zoom(image_input, zoom=(1.0/ scale), order=3)
                            image1_inp=cv2.resize(image1,(int(image_input.shape[0]/scale),int(image_input.shape[1]/scale)),interpolation=cv2.INTER_CUBIC)
                            image1_inp=image1_inp[:,:,np.newaxis]
                            image_input = image_input[:,:,np.newaxis]
                            img_input.append(image1_inp)
                            img_label.append(image_input)
                            count1 = count1 + 1
                            count =count +1
                            continue

    # img_input1=np.asarray(img_input)
    # img_label1 = np.asarray(img_label)
# else:
#     for img in a:
#             tmp = sitk.ReadImage(data_dir + "/" + img + "/brain.nii.gz")
#             tmp1 = sitk.ReadImage(data_dir + "/" + img + "/brain_mask.nii.gz")
#             data = sitk.GetArrayFromImage(tmp)
#             m = sitk.GetArrayFromImage(tmp1)
#             data = np.transpose(data, (1, 2, 0))
#             data = data[0:300, 0:258, 0:258]
#             m = np.transpose(m, (1, 2, 0))
#             mask = m[0:300, 0:258, 0:258]
#
#             max_inp = np.ones_like(data) * np.max(data)
#             min_inp = np.min(data)
#             inputs_1 = (data - min_inp) / (max_inp - min_inp)
#
#             # get image from slices
#             for i in range(data.shape[0]):
#                 mask_1 = mask[i:i + 1, :, :]
#                 m1 = mask_1
#                 m2 = mask_1
#                 m_zero = m1[m1 == 0]
#                 m_one = m2[m2 == 0]
#                 print ("num_zero:", m_zero.size)
#                 print ("num_one:", m_one.size)
#                 rate = m_one.size / (float(m_one.size) + float(m_zero.size))
#                 print("img:", img, "rate:", rate)
#                 if rate >= 0.3:
#                     H_image = inputs_1[i:i + 1, :, :]
#                     H_image1 = H_image[:, :, 1]
#                     x, y, z = H_image.shape
#                     q = x / image_size
#                     p = y / image_size
#                     for m in range(q):
#                         for n in range(p):
#                             image = H_image[m * image_size:(m + 1) * image_size, n * image_size:(n + 1) * image_size, :]
#                             input = cv2.resize(image, (image_size / scale, image_size / scale, 1))
#                             img_input.append(input)
#                             img_label.append(image)
#
#             dataset['test_input'][int(img)] = list(img_input)
#             dataset['test_label'][int(img)] = list(img_label)

