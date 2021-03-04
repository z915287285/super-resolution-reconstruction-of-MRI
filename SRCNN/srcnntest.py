import SimpleITK as sitk
import matplotlib.pyplot as plt
import os
import glob
from scipy.misc import imread, imsave
from skimage.color import ycbcr2rgb
import numpy as np
import h5py
import scipy.ndimage
from skimage.measure._structural_similarity import compare_ssim
from skimage.measure import compare_psnr
import random
#制作RGB图像
def makeRGB():
    dir = os.getcwd()+"\Test\Set5"
    data = glob.glob(os.path.join(dir, "*.bmp"))
    result_dir = os.getcwd()+"/RGBtest/"
    # 以YCbCr色彩空间打开,分别存储，通道
    Image = imread(data[2],flatten=False,mode="YCbCr").astype(np.float)
    Yimage = Image[:,:,0]
    CbANDCrimage = Image[:,:,1:]

    imsave(result_dir+"Yimage.bmp", Yimage)#存储Y通道
    #存储其他两个通道
    with h5py.File(os.getcwd()+"/RGBtest/CbANDCrimage.h5", 'w') as hf:
        hf.create_dataset("CbANDCrimage", data=CbANDCrimage)
    #打开两个其他通道
    with h5py.File(os.getcwd()+"/RGBtest/CbANDCrimage.h5", 'r') as hf:
        dataCbANDCrimage = np.array(hf.get("CbANDCrimage"))
    #组合在一起
    coimage = np.zeros([Yimage.shape[0],Yimage.shape[1],3])
    Yimage=Yimage.reshape([Yimage.shape[0],Yimage.shape[0],1])
    coimage[:, :,0:1] = Yimage
    coimage[:,:,1:] = dataCbANDCrimage

    imsave(result_dir+"YCbCr组合结果.bmp",coimage)
    imsave(result_dir+"RGB组合结果.bmp",ycbcr2rgb(coimage))


#把一个人的脑图切成128*128的，把他存到testimage图片中了，同时存到jist_test
def test3():
    data_dir="./rawdata/"
    img_files = os.listdir(data_dir)
    path="./checkpoint"
    a=img_files[-1]#-1
    b=img_files[:-1]
    isTrain = True
    image_size = 33
    scale = 3
    img_input = []
    img_label = []
    if isTrain:
        if not os.path.exists(path):
            os.makedirs(path)
        cont=0
        for img in b:
            d_imgshape = (3540, image_size // scale, image_size // scale, 1)
            d_labelshape = (3540, image_size, image_size, 1)
            tmp = sitk.ReadImage(data_dir + "/" + img + "/brain.nii.gz")
            tmp1 = sitk.ReadImage(data_dir + "/" + img + "/brain_mask.nii.gz")
            data = sitk.GetArrayFromImage(tmp)
            m = sitk.GetArrayFromImage(tmp1)
            data = np.transpose(data, (1, 2, 0))
            data = data[0:300, 0:258, 0:258]
            m = np.transpose(m, (1, 2, 0))
            mask = m[0:300, 0:258, 0:258]

            max_inp = np.ones_like(data) * np.max(data)
            min_inp = np.min(data)
            inputs_1 = (data - min_inp) / (max_inp - min_inp)
            count = 0
            maskVaildValue = []
            for i in range(data.shape[0]):
                mask_1 = mask[i:i + 1, :, :]
                m1 = mask_1
                m2 = mask_1
                m_zero = m1[m1 == 0]
                m_one = m2[m2 == 1]
                print ("num_zero:", m_zero.size)
                print ("num_one:", m_one.size)
                rate = m_one.size / (float(m_one.size) + float(m_zero.size))
                print("img:", img, "rate:", rate)
                if rate > 0.3:
                    H_image = inputs_1[i:i + 1, :, :]
                    H_image1 = np.transpose(H_image, (1, 2, 0))
                    maskOne = mask_1.reshape([258, 258])
                    H_image1 = H_image1.reshape([258, 258])
                    for i in range(65, maskOne.shape[0]-65,64):  # 有步长
                        for j in range(65, maskOne.shape[1]-65,64):
                            if (maskOne[i][j] == 1):
                                valueArea = maskOne[i - 64:i + 63, j - 64:j + 63]
                                m_zero = valueArea[valueArea == 0]
                                m_one = valueArea[valueArea == 1]
                                rate = m_one.size / (float(m_one.size) + float(m_zero.size))
                                if rate > 0.5 :  # 有效区域大于0.7的才记录下来
                                    maskVaildValue.append([i, j])
                                    valueArea = H_image1[i - 64:i + 64, j - 64:j + 64]
                                    valueArea = valueArea.reshape([128, 128])
                                    imsave(os.getcwd()+"/testimage/"+str(count)+".png",valueArea)
                                    img_input.append(valueArea)
                                    count = count+1
            break
        img_input = np.asarray(img_input)
        with h5py.File(os.path.join(path, "jist_test.h5"), 'w') as hf:
            hf.create_dataset("test", data=img_input)
        with h5py.File(os.path.join(path, "jist_test.h5"), 'r') as hf:
            images = np.array(hf.get("test"))
        count=0
        for img in images:
            if count>10: break
            plt.figure()
            plt.imshow(img,cmap="gray")
            plt.show()
            count = count+1

#制作遥感图像的h5
def test1():
    scale=3
    dir=os.getcwd()+"/Test/"
    image = imread(dir+"470.tif", mode="L")
    image = image[0:258, 0:258]
    image_test = image[6:258-6,6:258-6]#lable
    image_data = scipy.ndimage.interpolation.zoom(image, (1. / scale), prefilter=False)
    image_data = scipy.ndimage.interpolation.zoom(image_data, (scale / 1.), prefilter=False)
    image_data = image_data.reshape([1, 258, 258, 1])#train

    #这里直接测试SI的值
    image_bicubic = scipy.ndimage.interpolation.zoom(image_test, (1. / scale), prefilter=False)
    image_bicubic = scipy.ndimage.interpolation.zoom(image_bicubic, (scale / 1.), prefilter=False)
    psnr = compare_psnr(image_test, image_bicubic)
    ssim = compare_ssim(image_test, image_bicubic)


    #SI测试
    with h5py.File(dir+"SItest.h5",'w') as hf:
        hf.create_dataset("SItest", data=image_data)
        hf.create_dataset("SIlabel",data=image_test.reshape([1,246,246,1]))
        hf.create_dataset("origin", data= (image[6:258-6,6:258-6]))
        hf.create_dataset("bicubic_psnr", data=psnr)
        hf.create_dataset("bicubic_ssim", data=ssim)


def test2():
    a = np.array([[0, 1, 2],
         [3, 1, 1]])
    b = np.argwhere(a==1)#获取所有有效值
    print(b)
    b = b.tolist()#做成list
    print(b)
    random.shuffle(b)#打乱顺序
    for item in b:
        print(item)

#座一张特定的图片，有效数据是70%
def test4():
    data_dir = "./rawdata/"
    img_files = os.listdir(data_dir)
    path = "./checkpoint"
    a = img_files[-1]  # -1
    b = img_files[:-1]
    isTrain = True
    image_size = 33
    scale = 3
    img_input = []
    img_label = []
    if isTrain:
        if not os.path.exists(path):
            os.makedirs(path)
        cont = 0
        for img in a:
            d_imgshape = (3540, image_size // scale, image_size // scale, 1)
            d_labelshape = (3540, image_size, image_size, 1)
            tmp = sitk.ReadImage(data_dir + "/" + img + "/brain.nii.gz")
            tmp1 = sitk.ReadImage(data_dir + "/" + img + "/brain_mask.nii.gz")
            data = sitk.GetArrayFromImage(tmp)
            m = sitk.GetArrayFromImage(tmp1)
            data = np.transpose(data, (1, 2, 0))
            data = data[0:300, 0:258, 0:258]
            m = np.transpose(m, (1, 2, 0))
            mask = m[0:300, 0:258, 0:258]

            max_inp = np.ones_like(data) * np.max(data)
            min_inp = np.min(data)
            inputs_1 = (data - min_inp) / (max_inp - min_inp)
            #现有的操过数据140 139 141
            chooise = 139
            needimage = mask[chooise]
            # plt.figure()
            # plt.imshow(data[chooise])
            # plt.show()
            for stride in range(0,100):
                outbox = needimage[0+stride:258-stride, 0+stride:258-stride]
                outvalid = outbox[outbox==1]
                outinvalid = outbox[outbox==0]
                sore = outvalid.size/(outvalid.size+outinvalid.size)
                if sore >0.9:
                    train_image = inputs_1[chooise,0+stride:258-stride, 0+stride:258-stride]# id:140 152*152 ||id 50 90*90
                    print("stride:",stride)
                    break
            # plt.figure()
            # plt.imshow(train_image,cmap="gray")
            # plt.show()
            h = train_image.shape[0]
            w = train_image.shape[1]

            test_train = scipy.ndimage.interpolation.zoom(train_image, (1. / scale), prefilter=False)
            test_train = scipy.ndimage.interpolation.zoom(test_train, (scale / 1.), prefilter=False)# ID140  153*153 ID 130 141*141  ID 150 153*1535
            #test_train = test_train[1:153, 1:153]   #152*152
            bicubicpsnr = compare_psnr(train_image, test_train)

            test_label = train_image[ 0+6:h-6, 0+6:w-6]#140 140

            test_label = test_label.reshape([1,test_label.shape[0],test_label.shape[1],1])
            test_train = test_train.reshape(([1,h,w,1]))
            with h5py.File(os.path.join(path, "singleimage.h5"), 'w') as hf:
                hf.create_dataset("trian_data", data= test_train)
                hf.create_dataset("train_label", data=test_label)
                hf.create_dataset("bicubicpsinr", data=bicubicpsnr)
test4()







































# #读取他们的三个通道 组合在一起
# openYimage = imread(result_dir+"Yimage.bmp",flatten=True, mode="YCbCr").astype(np.float)
# openCbimage = imread(result_dir+"Cbimage.bmp",flatten=False, mode="YCbCr").astype(np.float)
# openCrimage = imread(result_dir+"Crimage.bmp",flatten=False, mode="YCbCr").astype(np.float)
#
# # imsave(result_dir+"openYimage.bmp", openYimage)
# # imsave(result_dir+"openCbimage.bmp", openCbimage)
# # imsave(result_dir+"openCrimage.bmp", openCrimage)
#
#
#
# coimage = np.zeros([openCrimage.shape[0],openCrimage.shape[1],3])
# coimage = coimage[:,:,0] = openYimage
# coimage = coimage[:,:,1] = openCbimage
# coimage = coimage[:,:,2] = openCbimage
#
#
# imsave(result_dir+"组合结果.bmp",coimage)



