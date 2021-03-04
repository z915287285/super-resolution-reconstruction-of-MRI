from model import EDSR
import scipy.misc
import argparse
import data
import cv2
import os
import SimpleITK as sitk
import glob
from skimage.measure._structural_similarity import compare_ssim
from skimage.measure import compare_psnr
import scipy.ndimage
import cv2
import numpy as np
import matplotlib.pyplot as plt
import h5py

def evatue(input):
	a = np.squeeze(input)

	max_inp = np.ones_like(a) * np.max(a)
	min_inp = np.min(a)
	inp = (a - min_inp) / (max_inp - min_inp)

	# tmp1=scipy.ndimage.zoom(input=inp,zoom=1/3,order=3)
	# x2=scipy.ndimage.zoom(input=tmp1,zoom=3,order=3)
	inp_size = inp.shape[0]
	tmp1 = cv2.resize(inp, (inp_size // args.scale, inp_size // args.scale))
	tmp1_size = tmp1.shape[0]
	x2 = cv2.resize(tmp1, (tmp1_size * args.scale, tmp1_size * args.scale))

	max_inp = np.ones_like(tmp1) * np.max(tmp1)
	min_inp = np.min(tmp1)
	x_in = (tmp1 - min_inp) / (max_inp - min_inp)
	x_in = x_in[:, :, np.newaxis]
	x2 = x2[:, :, np.newaxis]
	a_1 = inp[:, :, np.newaxis]

	x_in = np.reshape(x_in, [1, args.imgsize//3, args.imgsize//3, 1])

	outputs = network.predict(x_in)
	# #outputs=np.array(outputs,dtype=float)
	#
	# path='./out'
	# path1='./Test1'
	#
	# #input_image=cv2.imread(path+'/bc_image.png')
	# input_image1=np.array(x,dtype=float)
	# sr_image=outputs
	# H_image=cv2.imread(path1+'/7_180_0.305735833183102.png')
	# '''input_image=np.double(input_image)
	# sr_image=np.double(sr_image)
	# H_image=np.double(H_image)
	# '''
	# #bicubic
	# '''input_image1=scipy.ndimage.zoom(input_image, 3, order=3)'''
	#
	#
	# H_image=np.array(H_image,dtype=float)
	# #BN
	# '''
	# max_input_image1=np.ones_like(input_image1)*np.max(input_image1)
	# min_input_image1=np.min(input_image1)
	# input_image2=(input_image1-min_input_image1)/(max_input_image1-min_input_image1)
	#
	#
	# max_H_image=np.ones_like(H_image)*np.max(H_image)
	# min_H_image=np.min(H_image)
	# H_image=(H_image-min_H_image)/(max_H_image-min_H_image)
	# '''
	# max_sr_image=np.ones_like(outputs1)*np.max(outputs1)
	# min_sr_image=np.min(sr_image)
	# sr_image=(sr_image-min_sr_image)/(max_sr_image-min_sr_image)

	#
	# sr_image=sr_image[:,:,1]
	# H_image=H_image[:,:,1]
	# input_image2=input_image1[:,:,1]
	outputs = np.squeeze(outputs, axis=0)
	outputs[outputs>1]=1
	x2[x2>1]=1
	# max_in = np.ones_like(outputs) * np.max(outputs)
	# min_in = np.min(outputs)
	# outputs1 = (outputs - min_in) / (max_in - min_in)
	# SSIM
	a_1 = np.squeeze(a_1)
	x2 = np.squeeze(x2)
	outputs = np.squeeze(outputs)
	bh_ssim = compare_ssim(x2, a_1)
	# print('bh_ssim:', bh_ssim)
	sh_ssim = compare_ssim(outputs, a_1)
	# print('sh_ssim:', sh_ssim)
	# psnr
	bh_psnr = compare_psnr(a_1, x2)
	# print('bh_psnr:', bh_psnr)
	sh_psnr = compare_psnr(a_1, outputs)
	# print('sh_psnr:', sh_psnr)

	# print("**************第%d张psnr edsr:%.4f|bicubic:%.4f**************" % (i, sh_psnr, bh_psnr))
	# print("**************第%d张ssim edsr:%.4f|bicubic:%.4f**************" % (i, sh_ssim, bh_ssim))
	plt.figure()
	plt.subplot(1, 3, 1)
	plt.imshow(a_1, cmap="gray")
	plt.title("orgin" )

	plt.subplot(1, 3, 2)
	plt.imshow(outputs, cmap="gray")
	plt.title("edsr:%.4f" % sh_psnr)

	plt.subplot(1, 3, 3)
	plt.imshow(x2, cmap="gray")
	plt.title("bicubic:%.4f" % bh_psnr)
	plt.show()
	return sh_ssim,sh_psnr,bh_ssim,bh_psnr

parser = argparse.ArgumentParser()
parser.add_argument("--dataset",default="/Test1/6_72_0.3039480800432666.png")
parser.add_argument("--imgsize",default=300,type=int)
parser.add_argument("--scale",default=3,type=int)
parser.add_argument("--layers",default=32,type=int)
parser.add_argument("--featuresize",default=64,type=int)
parser.add_argument("--batchsize",default=32,type=int)
parser.add_argument("--savedir",default="save_model_200")
parser.add_argument("--iterations",default=1000,type=int)
parser.add_argument("--numimgs",default=5,type=int)
parser.add_argument("--outdir",default="out")
parser.add_argument("--image",default="guqinying/*_brain.nii.gz")
args = parser.parse_args()
if not os.path.exists(args.outdir):
	os.mkdir(args.outdir)
down_size = args.imgsize//args.scale
network = EDSR(down_size,args.layers,args.featuresize,scale=args.scale)
network.resume(args.savedir)
li_data = glob.glob(args.image)
x1=sitk.ReadImage(li_data[0])
# # x = cv2.imread(args.image)
# # x=np.array(x,dtype=float)
tmp=sitk.GetArrayFromImage(x1)
# # x_trans = np.transpose(tmp, (1, 2, 0))
x_1 = tmp[0:155, 0:351, 0:351]

# input=x_1[120,:,:]

sum_ssim=0
sum_psnr=0
bh_sum_ssim=0
bh_sum_psnr=0
count=0#stastic toltal
ssim_count=0
psnr_count=0
mean_psnr_list = []
mean_ssim_list = []
psnr_list = []
ssim_list = []
# pad = 100
# with h5py.File("data_train_guqinying_150.h5", 'r') as hf:
# 	input = np.array(hf['test_input1'])
# 	label = np.array(hf['test_label1'])
# num = input.shape[0]
# count_mean_list= [i*pad for i in range(input.shape[0]//pad)]
# local = 352/2
size = args.imgsize
size=300
local=352//2
for i in range(85,86):
	print("当前:",i)
	input1=x_1[i]
	input1 = np.squeeze(input1)
	input1 = input1[int(local - size / 2):int(local + size / 2), int(local - size / 2):int(local + size / 2)]
	#取出testimage 无背景
	# test = input1[int(local-size/2):int(local+size/2),int(local-size/2):int(local+size/2)]
	x_zero = (input1 == 0)
	num_zero = np.count_nonzero(x_zero)
	rate_x_zero = 1.0 * num_zero / (input1.shape[0] * input1.shape[1])
	if rate_x_zero>0.6:
		continue
	ssim1,psnr1,bh_ssim1,bh_psnr1=evatue(input1)
	if ssim1>bh_ssim1:
		ssim_count=ssim_count+1
	if psnr1>bh_psnr1:
		psnr_count=psnr_count+1
	print("ssim:",ssim1)
	print("psnr:",psnr1)
	sum_ssim=sum_ssim+ssim1
	sum_psnr=sum_psnr+psnr1
	bh_sum_ssim =bh_sum_ssim+bh_ssim1
	bh_sum_psnr =bh_sum_psnr+bh_psnr1
	count=count+1

	# psnr_list.append(psnr1)
	# ssim_list.append(ssim1)
	# if count%pad==0:
	# 	print("目前第",count)
	# 	mean_ssim = sum_ssim / count
	# 	mean_psnr = sum_psnr / count
	# 	print("ssim:", mean_ssim)
	# 	print("psnr:", mean_psnr)
	# 	bh_mean_ssim = bh_sum_ssim / count
	# 	bh_mean_psnr = bh_sum_psnr / count
	# 	print("bh_ssim:", bh_mean_ssim)
	# 	print("bh_psnr:", bh_mean_psnr)
	# 	mean_psnr_list.append(mean_psnr)
	# 	mean_ssim_list.append(mean_ssim)
# plt.title('Result PSNR')
# # plt.plot(list(range(count)), psnr_list, color='green', label='test psnr')
# plt.plot(count_mean_list, mean_psnr_list, color='red', label='mean psnr')
# plt.legend()  # 显示图例
#
# plt.xlabel('iteration')
# plt.ylabel('psnr')
# plt.show()
# # plt.savefig('psnr200.png')
#
# plt.title('Result SSIM')
# # plt.plot(list(range(count)), ssim_list, color='green', label='test ssim')
# plt.plot(count_mean_list, mean_ssim_list, color='red', label='mean ssim')
# plt.legend()  # 显示图例
#
# plt.xlabel('iteration')
# plt.ylabel('ssim')
# plt.show()
# # plt.savefig('ssim200.png')

print("总数：",count)
mean_ssim=sum_ssim/count
mean_psnr=sum_psnr/count
print("ssim:",mean_ssim)
print("psnr:",mean_psnr)
bh_mean_ssim=bh_sum_ssim/count
bh_mean_psnr=bh_sum_psnr/count
print("bh_ssim:",bh_mean_ssim)
print("bh_psnr:",bh_mean_psnr)

print(ssim_count)
print(psnr_count)
print(count)
print("ssim over rate:",float(ssim_count)/count)
print("psnr over rate:",float(psnr_count)/count)
# fig = plt.figure()
# a = fig.add_subplot(1,3,1)
# sr_image=outputs[:,:,0]
# plt.imshow(sr_image, cmap="gray")
# a.set_title('SR')
# a = fig.add_subplot(1,3,2)
# H_image=a_1[:,:,0]
# plt.imshow(H_image,cmap="gray")
# a.set_title('H_image')
# a = fig.add_subplot(1,3,3)
# input_img=x2[:,:,0]
# plt.imshow(input_img,cmap="gray")
# a.set_title('Bicubic')
# plt.show()
