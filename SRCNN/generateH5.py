import numpy as np
import SimpleITK as sitk
import os
import h5py
import cv2
import random
import matplotlib.pyplot as plt
import scipy.misc
import scipy.ndimage
from skimage.measure import compare_psnr
data_dir="./rawdata/"
img_files = os.listdir(data_dir)
path="./checkpoint"
a=img_files[-1]#-1
b=img_files[:-1]

isTrain = False
image_size = 33
scale = 3

count = 0
sub_input_sequence = []
sub_label_sequence = []
#把一张图片overloping分成33*33的小块
def makeimage33(inputs_1,mask):
    scale = 3
    image_size = 33
    label_size = 21
    stride=14
    padding=6
    sub_input_sequence = []
    sub_label_sequence = []

    input_ = scipy.ndimage.interpolation.zoom(inputs_1, (1. / scale), prefilter=False)
    input_ = scipy.ndimage.interpolation.zoom(input_, ( scale/ 1.), prefilter=False)
    label_= inputs_1
    if len(input_.shape) == 3:
        h, w, _ = input_.shape
    else:
        h, w = input_.shape
    loc = []
    # Numbers of sub-images in height and width of image are needed to compute merge operation.
    nx = ny = 0
    for x in range(0, h - image_size + 1, stride):
        nx += 1;
        ny = 0
        for y in range(0, w - image_size + 1, stride):
            ny += 1
            sub_input = input_[x:x + image_size, y:y + image_size]  # [33 x 33]
            sub_label = label_[x + int(padding):x + int(padding) + label_size,y + int(padding):y + int(padding) + label_size]  # [21 x 21]

            sub_input = sub_input.reshape([image_size, image_size, 1])
            sub_label = sub_label.reshape([label_size, label_size, 1])

            sub_input_sequence.append(sub_input)
            sub_label_sequence.append(sub_label)

            loc.append([x + int(padding), y + int(padding)])

    # Make list to numpy array. With this transform
    arrdata = np.asarray(sub_input_sequence)  # [?, 33, 33, 1]
    arrlabel = np.asarray(sub_label_sequence)  # [?, 21, 21, 1]
    nx= np.asarray(nx)
    ny = np.asarray(ny)
    mask = mask[0 + 6:258 - 7, 0 + 6:258 - 7]
    with h5py.File(os.path.join(path, "data_test.h5"), 'w') as hf:
        hf.create_dataset("test_train", data=arrdata)
        hf.create_dataset("test_label", data=arrlabel)
        hf.create_dataset("mask",data=mask)
        hf.create_dataset("loc", data=loc)
        hf.create_dataset("nx",data=nx)
        hf.create_dataset('ny',data=ny)

#忽略图形的形状，直接计算值
def makeimagebychoose(inputs_1,mask):
    scale = 3
    image_size = 33
    label_size = 21
    stride = 14
    padding = 6
    sub_input_sequence = []
    sub_label_sequence = []
    sub_label_psnr = []

    input_ = scipy.ndimage.interpolation.zoom(inputs_1, (1. / scale), prefilter=False)
    input_ = scipy.ndimage.interpolation.zoom(input_, (scale / 1.), prefilter=False)
    label_ = inputs_1
    if len(input_.shape) == 3:
        h, w, _ = input_.shape
    else:
        h, w = input_.shape
    # Numbers of sub-images in height and width of image are needed to compute merge operation.
    nx = ny = 0
    for x in range(0, h - image_size + 1, stride):
        nx += 1;
        ny = 0
        for y in range(0, w - image_size + 1, stride):
            ny += 1
            sub_true =  inputs_1[x:x + image_size, y:y + image_size]
            sub_input = input_[x:x + image_size, y:y + image_size]  # [33 x 33]
            sub_label = label_[x + int(padding):x + int(padding) + label_size,y + int(padding):y + int(padding) + label_size]  # [21 x 21]

            m_zero = sub_label[sub_label == 0]
            if m_zero.size == sub_label.size:
                continue
            else:
                m_one = sub_label[sub_label != 0]
            rate = m_one.size / (float(m_one.size) + float(m_zero.size))
            if rate > 0.7 :  # 有效区域大于0.7的才记录下来
                #计算bicubic的psnr
                sub_tru = label_[x:x + image_size, y:y + image_size]  # [33 x 33]
                sub_label_psnr.append(compare_psnr(sub_tru,sub_input))

                sub_input = sub_input.reshape([image_size, image_size, 1])
                sub_label = sub_label.reshape([label_size, label_size, 1])

                sub_input_sequence.append(sub_input)
                sub_label_sequence.append(sub_label)
    # Make list to numpy array. With this transform
    arrdata = np.asarray(sub_input_sequence)  # [?, 33, 33, 1]
    arrlabel = np.asarray(sub_label_sequence)  # [?, 21, 21, 1]


    with h5py.File(os.path.join(path, "data_test.h5"), 'w') as hf:
        hf.create_dataset("test_train", data=arrdata)
        hf.create_dataset("test_label", data=arrlabel)
        hf.create_dataset("test_psnr", data=sub_label_psnr)

#把一个人的图片全部做好，并且按规格剪裁
def makeimageCutByHand(inputs_1,mask):
    test_image_data = []#测试图片原图的bicubic操作结果图
    test_image_lable = []#测试图片的标签
    origin_data_258 = []#测试图片原图
    origin_data_246 = []#测试图片的裁剪图stride-12
    maskimages = []

    # 对所有切片进行剪切，stride=54 30
    stride = 54
    for i in range(mask.shape[0]):
        mask_1 = mask[i, :, :]
        m1 = mask_1
        m2 = mask_1
        m_zero = m1[m1 == 0]
        m_one = m2[m2 == 1]
        # print ("num_zero:", m_zero.size)
        # print ("num_one:", m_one.size)
        rate = m_one.size / (float(m_one.size) + float(m_zero.size))
        if rate > 0.3:
            print("img:", i, "rate:", rate)
            # 抓取重点信息
            maskimage = mask[i, 0 + stride + 6:258 - stride - 6, 0 + stride + 6:258 - stride - 6]
            maskimages.append(maskimage)
            test_image = inputs_1[i, 0 + stride: 258 - stride, 0 + stride:258 - stride]
            origin_data_258.append(test_image)  # bicubic

            # test_label的制作
            test_lable = test_image[0 + 6:test_image.shape[0] - 6, 0 + 6:test_image.shape[0] - 6]
            origin_data_246.append(test_lable)
            test_lable = test_lable.reshape([test_lable.shape[0], test_lable.shape[1], 1])
            test_image_lable.append(test_lable)

            # test_data的制作
            test_image = scipy.ndimage.interpolation.zoom(test_image, (1. / scale), prefilter=False)
            test_image = scipy.ndimage.interpolation.zoom(test_image, (scale / 1.), prefilter=False)
            test_image = test_image.reshape([test_image.shape[0], test_image.shape[1], 1])
            test_image_data.append(test_image)

            # plt.figure()
            # plt.subplot(1,2,1)
            # temp = test_lable.reshape([test_lable.shape[0],test_lable.shape[1]])
            # plt.imshow(temp, cmap='gray')
            #
            # plt.subplot(1,2,2)
            # temp = test_image.reshape([test_image.shape[1], test_image.shape[1]])
            # plt.imshow(temp, cmap="gray")
            # plt.show()

    origin_data_258 = np.asarray(origin_data_258)
    origin_data_246 = np.asarray(origin_data_246)
    test_image_data = np.asarray(test_image_data)
    test_image_lable = np.asarray(test_image_lable)
    maskimages = np.asarray(maskimages)

    with h5py.File(os.path.join(path, "data_origin_258.h5"), 'w') as hf:
        hf.create_dataset("origin_data_258", data=origin_data_258)

    with h5py.File(os.path.join(path, "data_origin_246.h5"), 'w') as hf:
        hf.create_dataset("origin_data_246", data=origin_data_246)

    with h5py.File(os.path.join(path, "data_test.h5"), 'w') as hf:
        hf.create_dataset("test_train", data=test_image_data)
        hf.create_dataset("test_label", data=test_image_lable)
        hf.create_dataset("test_mask", data=maskimages)

#把测试集的所有有效果数据大于0.3的图片保存起来，测试它们的效果
def makeimageCut(inputs_1,mask):
    test_image_data = []
    test_image_lable = []

    origin_data_258 = []
    origin_data_246 = []

    maskimages = []

    for i in range(mask.shape[0]):
        mask_1 = mask[i, :, :]
        m1 = mask_1
        m2 = mask_1
        m_zero = m1[m1 == 0]
        m_one = m2[m2 == 1]
        # print ("num_zero:", m_zero.size)
        # print ("num_one:", m_one.size)
        rate = m_one.size / (float(m_one.size) + float(m_zero.size))
        if rate > 0.3:
            print("img:", i, "rate:", rate)
            maskimage = mask[i, 6:258 - 6, 6:258 - 6]
            maskimages.append(maskimage)

            test_image = inputs_1[i, 0:258, 0:258]
            origin_data_258.append(test_image)  # bicubic

            # label的制作
            test_lable = test_image[6:258 - 6, 6:258 - 6]
            origin_data_246.append(test_lable)
            test_lable = test_lable.reshape([246, 246, 1])
            test_image_lable.append(test_lable)

            # test_data的制作
            test_image = scipy.ndimage.interpolation.zoom(test_image, (1. / scale), prefilter=False)
            test_image = scipy.ndimage.interpolation.zoom(test_image, (scale / 1.), prefilter=False)
            test_image = test_image.reshape([258, 258, 1])
            test_image_data.append(test_image)

    origin_data_258 = np.asarray(origin_data_258)
    origin_data_246 = np.asarray(origin_data_246)
    test_image_data = np.asarray(test_image_data)
    test_image_lable = np.asarray(test_image_lable)
    maskimages = np.asarray(maskimages)

    with h5py.File(os.path.join(path, "data_origin_258.h5"), 'w') as hf:
        hf.create_dataset("origin_data_258", data=origin_data_258)

    with h5py.File(os.path.join(path, "data_origin_246.h5"), 'w') as hf:
        hf.create_dataset("origin_data_246", data=origin_data_246)

    with h5py.File(os.path.join(path, "data_test.h5"), 'w') as hf:
        hf.create_dataset("test_train", data=test_image_data)
        hf.create_dataset("test_label", data=test_image_lable)
        hf.create_dataset("test_mask", data=maskimages)

#选择有效值大于0.7的，并且中心像素点距离大于10
def triandata1(inputs_1,mask):
    global sub_input_sequence
    global sub_label_sequence
    # 处理一个切片
    for i in range(inputs_1.shape[0]):
        mask_1 = mask[i:i + 1, :, :]
        m1 = mask_1
        m2 = mask_1
        m_zero = m1[m1 == 0]
        m_one = m2[m2 == 1]
        # print ("num_zero:", m_zero.size)
        # print ("num_one:", m_one.size)
        rate = m_one.size / (float(m_one.size) + float(m_zero.size))
        print("img:", img, "rate:", rate)
        if rate > 0.3:  # 图像有效区域
            H_image = inputs_1[i:i + 1, :, :]
            H_image1 = np.transpose(H_image, (1, 2, 0))
            # 获取所有mask的有效坐标,
            maskOne = mask_1.reshape([258, 258])
            H_image1 = H_image1.reshape([258, 258])

            # 在mask_1中随机去10个点，且有一定的距离要求
            samplenNumber = 3
            limitdistence = 30
            vaildlist = np.argwhere(maskOne == 1)  # 获取所有有效值
            vaildlist = vaildlist.tolist()  # 做成list
            random.shuffle(vaildlist)  # 打乱顺序
            makeals = []  # 在有效数据中，拉开距离
            for item in vaildlist:
                if len(makeals) == 0:
                    makeals.append(item)  # 只有一条数据
                else:
                    flg = 1
                    vector1 = np.array(item)
                    for idx in makeals:  # 距离小于已有点中的任意一个就flg=0,不选择
                        vector2 = np.array(idx)
                        op2 = np.linalg.norm(vector1 - vector2)
                        if op2 < limitdistence:
                            flg = 0
                            break
                    if flg == 1:
                        makeals.append(item)
                    if len(makeals) >= samplenNumber:
                        break

            for loc in makeals:
                i = loc[0]
                j = loc[1]
                if (j >= 17 and i >= 17 and j + 16 <= 258 and i + 16 <= 258):
                    valueArea = maskOne[i - 17:i + 16, j - 17:j + 16]
                    m_zero = valueArea[valueArea == 0]
                    m_one = valueArea[valueArea == 1]
                    rate = m_one.size / (float(m_one.size) + float(m_zero.size))
                    if rate > 0.7:  # 有效区域大于0.7的才记录下来
                        valueArea = H_image1[i - 17:i + 16, j - 17:j + 16]
                        valueArea = valueArea.reshape([33, 33])
                        image = valueArea[5:26, 5:26]
                        image = image.reshape([21, 21, 1])
                        sub_label_sequence.append(image)
                        valueArea = scipy.ndimage.interpolation.zoom(valueArea, (1. / scale), prefilter=False)
                        valueArea = scipy.ndimage.interpolation.zoom(valueArea, (scale / 1.), prefilter=False)
                        valueArea = valueArea.reshape([33, 33, 1])
                        sub_input_sequence.append(valueArea)
    print("sub")

#在每个slice有效值尽量大的时候：0.4,使用srcnn原始模型制作数据集
def triandata2(inputs_1,mask):
    global sub_input_sequence
    global sub_label_sequence
    scale = 3
    image_size = 33
    label_size = 21
    stride = 14
    padding = 6
    global count
    # 处理一个切片
    for i in range(data.shape[0]):
        mask_1 = mask[i:i + 1, :, :]
        m1 = mask_1
        m2 = mask_1
        m_zero = m1[m1 == 0]
        m_one = m2[m2 == 1]
        rate = m_one.size / (float(m_one.size) + float(m_zero.size))
        if rate > 0.4:  # 图像有效区域
            print(rate)
            count=count+1
            input = inputs_1[i]*mask_1
            input = input.reshape([258,258])
            #scipy.misc.imsave("testimage/"+str(i)+".png", inputs_1[i])
            input_ = scipy.ndimage.interpolation.zoom(input, (1. / scale), prefilter=False)
            input_ = scipy.ndimage.interpolation.zoom(input_, (scale / 1.), prefilter=False)
            label_ = input
            if len(input_.shape) == 3:
                h, w, _ = input_.shape
            else:
                h, w = input_.shape

            for x in range(0, h - image_size + 1, stride):
                for y in range(0, w - image_size + 1, stride):
                    sub_input = input_[x:x + image_size, y:y + image_size]  # [33 x 33]
                    sub_label = label_[x + int(padding):x + int(padding) + label_size,
                                y + int(padding):y + int(padding) + label_size]  # [21 x 21]

                    # Make channel value
                    sub_input = sub_input.reshape([image_size, image_size, 1])
                    sub_label = sub_label.reshape([label_size, label_size, 1])

                    sub_input_sequence.append(sub_input)
                    sub_label_sequence.append(sub_label)

#在maskOne中找到有效数据，以它为中心取33*33大小的块，如果这个块的效数据大于70%才把它加入到数据集中
def traindata3(inputs_1,mask,conut):
    global sub_input_sequence
    global sub_label_sequence
    # 处理一个切片
    num = 0
    flag = False
    for i in range(inputs_1.shape[0]):
        mask_1 = mask[i:i + 1, :, :]
        m1 = mask_1
        m2 = mask_1
        m_zero = m1[m1 == 0]
        m_one = m2[m2 == 1]
        # print ("num_zero:", m_zero.size)
        # print ("num_one:", m_one.size)
        rate = m_one.size / (float(m_one.size) + float(m_zero.size))
        print("img:", img, "rate:", rate)
        if rate > 0.3:  # 图像有效区域
            H_image = inputs_1[i:i + 1, :, :]
            H_image1 = np.transpose(H_image, (1, 2, 0))
            # 获取所有mask的有效坐标,
            maskOne = mask_1.reshape([258, 258])
            H_image1 = H_image1.reshape([258, 258])
            vailraite=762#33*33=1089 , 1087*0.7=762.3
            # 在maskOne中找到有效数据，以它为中心取33*33大小的块，如果这个块的效数据大于70%才把它加入到数据集中
            for i in range(30,maskOne.shape[0]-30):
                for j in range(30,maskOne.shape[1]-30):
                    if maskOne[i][j] == 1 :
                        valueArea = H_image1[i-16:i+17,j-16:j+17]
                        zores = (valueArea[valueArea==1]).size
                        if zores<=vailraite:
                            num = num +1
                            input = scipy.ndimage.interpolation.zoom(H_image1, (1. / scale), prefilter=False)
                            input = scipy.ndimage.interpolation.zoom(input, (scale / 1.), prefilter=False)
                            valueArea = valueArea[6:27,6:27]
                            sub_input_sequence.append(input.reshape([33,33,1]))
                            sub_label_sequence.append(valueArea.reshape([21,21,1]))
                            if num > count:
                                flag = True
                                break
        if not flag:
            break

if isTrain:
    if not os.path.exists(path):
        os.makedirs(path)
    d_imgshape = (3540, image_size // scale, image_size // scale,1)
    d_labelshape = (3540, image_size, image_size,1)

    for img in b:
        tmp = sitk.ReadImage(data_dir + "/" + img + "/brain.nii.gz")
        tmp1 = sitk.ReadImage(data_dir + "/" + img + "/brain_mask.nii.gz")
        data = sitk.GetArrayFromImage(tmp)
        m = sitk.GetArrayFromImage(tmp1)
        data = np.transpose(data, (1, 2, 0))
        data = data[0:300, 0:258, 0:258]
        m = np.transpose(m, (1, 2, 0))
        mask = m[0:300, 0:258, 0:258]
        #归一化
        max_inp = np.ones_like(data) * np.max(data)
        min_inp = np.min(data)
        inputs_1 = (data - min_inp) / (max_inp - min_inp)

        #triandata1(inputs_1,mask)#图像制作方式：随机取图，且保证了一定的有效数据
        triandata2(inputs_1, mask)
        print("sub")

    arrdata = np.asarray(sub_input_sequence)  # [?, 33, 33, 1]
    arrlabel = np.asarray(sub_label_sequence)  # [?, 21, 21, 1]
    #数据打乱
    permutation = np.random.permutation(arrdata.shape[0])
    shuffled_input = arrdata[permutation, :, :, :]
    shuffled_label = arrlabel[permutation, :, :, :]

    with h5py.File(os.path.join(path, "data_train.h5"), 'w') as hf:
        hf.create_dataset("train_inpt", data=shuffled_input)
        hf.create_dataset("train_labl", data=shuffled_label)
else:
    for img in a:
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

        #makeimage33(inputs_1[140], mask[140])
        #makeimagebychoose(inputs_1[140], mask[140])
        #stride = 54
        #makeimagebychoose(inputs_1[140, 0 + stride: 258 - stride, 0 + stride:258 - stride],mask[140, 0 + stride: 258 - stride, 0 + stride:258 - stride])
        #makeimageCutByHand(inputs_1, mask)
        makeimageCut(inputs_1, mask)




