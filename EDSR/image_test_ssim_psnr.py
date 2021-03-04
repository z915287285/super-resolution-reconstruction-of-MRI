# standerlize the image
import scipy.misc
from skimage.measure._structural_similarity import compare_ssim
from skimage.measure import compare_psnr
import scipy.ndimage.interpolation
import cv2
import numpy as np
import matplotlib.pyplot as plt


#get the image

path='./out'
path1='./Test1'

input_image=cv2.imread(path+'/bc_image.png')

input_image1=np.array(input_image,dtype=float)
sr_image=cv2.imread(path+'/input.png')
H_image=cv2.imread(path1+'/7_180_0.305735833183102.png')
'''input_image=np.double(input_image)
sr_image=np.double(sr_image)
H_image=np.double(H_image)
'''
#bicubic
'''input_image1=scipy.ndimage.zoom(input_image, 3, order=3)'''


H_image=np.array(H_image,dtype=float)
sr_image=np.array(sr_image,dtype=float)
#BN'''
max_input_image1=np.ones_like(input_image1)*np.max(input_image1)
min_input_image1=np.min(input_image1)
input_image2=(input_image1-min_input_image1)/(max_input_image1-min_input_image1)


max_H_image=np.ones_like(H_image)*np.max(H_image)
min_H_image=np.min(H_image)
H_image=(H_image-min_H_image)/(max_H_image-min_H_image)

max_sr_image=np.ones_like(sr_image)*np.max(sr_image)
min_sr_image=np.min(sr_image)
sr_image=(sr_image-min_sr_image)/(max_sr_image-min_sr_image)

input_image2=input_image2[:,:,1]
sr_image=sr_image[:,:,1]
H_image=H_image[:,:,1]



#SSIM
bh_ssim=compare_ssim(input_image2, H_image, multichannel=False)
print('bh_ssim:',bh_ssim)
sh_ssim=compare_ssim(sr_image, H_image, multichannel=False)
print('sh_ssim:',sh_ssim)
#psnr
bh_psnr=compare_psnr(H_image, input_image2, data_range=1)
print('bh_psnr:',bh_psnr)
sh_psnr=compare_psnr(H_image, sr_image, data_range=1)
print('sh_psnr:', sh_psnr)



fig = plt.figure()
a = fig.add_subplot(1,3,1)
sr_image=sr_image[:,:,1]
plt.imshow(sr_image, cmap="gray")
a.set_title('SR')
a = fig.add_subplot(1,3,2)
H_image=H_image[:,:,1]
plt.imshow(H_image,cmap="gray")
a.set_title('H_image')
a = fig.add_subplot(1,3,3)
input_image2=input_image1[:,:,1]
plt.imshow(input_image2,cmap="gray")
a.set_title('Bicubic')
plt.show()