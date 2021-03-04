import numpy as np
import SimpleITK as sitk
import os
import h5py
import cv2
import random
import glob
import scipy.ndimage
import scipy.misc

img = "guqinying"
image_size = 150
scale = 3
img_input = []
img_label = []

print("当前：{}".format(img))
tmp_data = img+"\*_brain.nii.gz"
tmp_mask = img+"\*_brain_mask.nii.gz"
li_data = glob.glob(tmp_data)
li_mask = glob.glob(tmp_mask)
tmp = sitk.ReadImage(li_data[0])
tmp1 = sitk.ReadImage(li_mask[0])
data = sitk.GetArrayFromImage(tmp)
m = sitk.GetArrayFromImage(tmp1)
max_inp = np.ones_like(data) * np.max(data)
min_inp = np.min(data)
inputs_1 = (data - min_inp) / (max_inp - min_inp)
count = 0
# get image from slices
for i in range(data.shape[0]):
    mask_1 = m[i:i + 1, :, :]
    mask_zero = (mask_1 == 0)
    num_zero = np.count_nonzero(mask_zero)
    rate_mask_zero = 1.0 * num_zero / (mask_1.shape[1] * mask_1.shape[2])
    if rate_mask_zero < 0.7:
        index = np.where(mask_1)
        H_image = inputs_1[i:i + 1, :, :]
        count1 = 0
        for j in range(0, len(index[0]), 50):
            j_index = np.array((index[0][j], index[1][j], index[2][j]))
            local1 = int(j_index[1])
            local2 = int(j_index[2])
            # print(local1-image_size/2,local1+image_size/2)
            # print(local2-image_size/2,local2+image_size/2)
            image_input = H_image[:, int(local1 - image_size / 2):int(local1 + image_size / 2), \
                          int(local2 - image_size / 2):int(local2 + image_size / 2)]
            # print j_index[1],j_index[2]
            Iszero = (image_input == 0)
            num_zero = np.count_nonzero(Iszero)
            rate_zero = 1.0 * num_zero / (image_size * image_size)
            # print rate_zero
            if rate_zero < 0.1 and image_input.shape[1] == image_size and image_input.shape[2] == image_size:
                print("满足条件")
                # print (image_input.shape[0],image_input.shape[1])
                image1 = np.squeeze(image_input)
                image1_inp = scipy.ndimage.zoom(image_input, zoom=(1.0 / scale), order=3)
                # image1_inp=cv2.resize(image1,(int(image_input.shape[0]/scale),int(image_input.shape[1]/scale)),interpolation=cv2.INTER_CUBIC)
                input_ = scipy.ndimage.zoom(image1, (1. / scale), order=3)
                image1_inp=image1_inp[:,:,np.newaxis]
                input_ = input_[:,:,np.newaxis]
                img_input.append(input_)
                img_label.append(image1)
                count1 = count1 + 1

img_input1=np.asarray(img_input)
img_label1 = np.asarray(img_label)
n = len(img_input)
print(n)

permutation = np.random.permutation(img_input1.shape[0])
shuffled_input = img_input1[permutation, :, :]
shuffled_label= img_label1[permutation, :, :]

with h5py.File(os.path.join('', "data_train"+"_"+img+"_150.h5"), 'w') as hf:
    hf.create_dataset('test_input1', data=shuffled_input)
    hf.create_dataset('test_label1', data=shuffled_label)