import scipy.misc
import random
import numpy as np
import os
import cv2
import SimpleITK as sitk
import h5py
import glob

train_set = []
test_set = []
batch_index = 0

"""
Load set of images in a directory.
This will automatically allocate a 
random 20% of the images as a test set

data_dir: path to directory containing images
"""
def load_dataset(data_dir, it,batch_size):
	"""img_files = os.listdir(data_dir)
	test_size = int(len(img_files)*0.2)
	test_indices = random.sample(range(len(img_files)),test_size)
	for i in range(len(img_files)):
		#img = scipy.misc.imread(data_dir+img_files[i])
		if i in test_indices:
			test_set.append(data_dir+"/"+img_files[i])
		else:
			train_set.append(data_dir+"/"+img_files[i])
	return"""
	x_list = []
	y_list = []
	num = 0
	tmp_data = data_dir + "\*.h5"
	# tmp_mask = data_dir + "\*_brain_mask.nii.gz"
	li_data = glob.glob(tmp_data)
	# li_mask = glob.glob(tmp_mask)
	for f in li_data:
		data_path_files = h5py.File(f, 'r')
		x = np.array(data_path_files['train_input'])
		y = np.array(data_path_files['train_label'])
		print(x.shape[0])
		x_list.append(x)
		y_list.append(y)
	for j in range(len(x_list)):
		inpt = x_list[j]
		num = num +inpt.shape[0]
		print(num)
	iter = (num) // batch_size
	return iter*it

"""
Get test set from the loaded dataset

size (optional): if this argument is chosen,
each element of the test set will be cropped
to the first (size x size) pixels in the image.

returns the test set of your data
"""
def get_test_set(original_size,shrunk_size):
	"""for i in range(len(test_set)):
		img = scipy.misc.imread(test_set[i])
		if img.shape:
			img = crop_center(img,original_size,original_size)		
			x_img = scipy.misc.imresize(img,(shrunk_size,shrunk_size))
			y_imgs.append(img)
			x_imgs.append(x_img)"""

	x_list = []
	y_list = []
	tmp_data = "data\*.h5"
	# tmp_mask = data_dir + "\*_brain_mask.nii.gz"
	li_data = glob.glob(tmp_data)
	# li_mask = glob.glob(tmp_mask)
	for f in li_data:
		data_path_files = h5py.File(f, 'r')
		x = np.array(data_path_files['val_input'])
		y = np.array(data_path_files['val_label'])
		x_list.append(x)
		y_list.append(y)
	input_val =  x_list[0]
	label_val = y_list[0]
	for j in range(1,len(x_list)):
		input_val = np.concatenate((input_val, x_list[j]))
		label_val = np.concatenate((label_val, y_list[j]))

	return input_val,label_val

def get_image(imgtuple,size):
	# img = scipy.misc.imread(imgtuple[0])
	# img=cv2.imread(imgtuple[0])
	# img = np.array(img, dtype=float)
	# max_sr_image = np.ones_like(img) * np.max(img)
	# min_sr_image = np.min(img)
	# img1 = (img - min_sr_image) / (max_sr_image - min_sr_image)
	tmp = sitk.ReadImage(imgtuple[0])
	data = sitk.GetArrayFromImage(tmp)
	data = np.transpose(data, (1, 2, 0))
	data = data[0:300, 0:258, 0:258]
	max_inp = np.ones_like(data) * np.max(data)
	min_inp = np.min(data)
	data1 = (data - min_inp) / (max_inp - min_inp)
	x,y = imgtuple[1]
	index=imgtuple[2]
	data1=data1[index:index+1,:,:]
	data2=np.squeeze(data1)
	img2 = data2[x*size:(x+1)*size,y*size:(y+1)*size]
	return img2
	

"""
Get a batch of images from the training
set of images.

batch_size: size of the batch
original_size: size for target images
shrunk_size: size for shrunk images

returns x,y where:
	-x is the input set of shape [-1,shrunk_size,shrunk_size,channels]
	-y is the target set of shape [-1,original_size,original_size,channels]
"""
def get_batch(batch_size,original_size,shrunk_size):
	global batch_index
	"""img_indices = random.sample(range(len(train_set)),batch_size)
	for i in range(len(img_indices)):
		index = img_indices[i]
		img = scipy.misc.imread(train_set[index])
		if img.shape:
			img = crop_center(img,original_size,original_size)
			x_img = scipy.misc.imresize(img,(shrunk_size,shrunk_size))
			x.append(x_img)
			y.append(img)"""
	# window = [x for x in range(counter*batch_size,(counter+1)*batch_size)]
	# imgs = [train_set[q] for q in window]
	# x = [cv2.resize(get_image(q,original_size),(shrunk_size,shrunk_size)) for q in imgs]#scipy.misc.imread(q[0])[q[1][0]*original_size:(q[1][0]+1)*original_size,q[1][1]*original_size:(q[1][1]+1)*original_size].resize(shrunk_size,shrunk_size) for q in imgs]
	# y = [get_image(q,original_size) for q in imgs]#scipy.misc.imread(q[0])[q[1][0]*original_size:(q[1][0]+1)*original_size,q[1][1]*original_size:(q[1][1]+1)*original_size] for q in imgs]
	x_list = []
	y_list = []
	num=0
	tmp_data = "data\*.h5"
	# tmp_mask = data_dir + "\*_brain_mask.nii.gz"
	li_data = glob.glob(tmp_data)
	# li_mask = glob.glob(tmp_mask)
	for f in li_data:
		data_path_files = h5py.File(f, 'r')
		x = np.array(data_path_files['train_input'])
		y = np.array(data_path_files['train_label'])
		x_list.append(x)
		y_list.append(y)
	input_train = x_list[0]
	label_train = y_list[0]
	for j in range(1, len(x_list)):
		input_train = np.concatenate((input_train, x_list[j]))
		label_train = np.concatenate((label_train, y_list[j]))
	max_counter = (input_train.shape[0]) // batch_size
	counter = batch_index % max_counter
	x3=[]
	y3=[]
	for i in range(batch_index*batch_size,(batch_index+1)*batch_size):
		x2=input_train[i,:,:,:]
		y2=label_train[i,:,:,:]
		x3.append(x2)
		y3.append(y2)
	batch_index = (batch_index+1)%max_counter
	return x3,y3

"""
Simple method to crop center of image

img: image to crop
cropx: width of crop
cropy: height of crop
returns cropped image
"""
def crop_center(img,cropx,cropy):
	y,x,_ = img.shape
	startx = random.sample(range(x-cropx-1),1)[0]#x//2-(cropx//2)
	starty = random.sample(range(y-cropy-1),1)[0]#y//2-(cropy//2)
	return img[starty:starty+cropy,startx:startx+cropx]





