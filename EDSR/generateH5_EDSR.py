import nibabel as nib
import numpy as np
import os
# import matplotlib.pyplot as plt
from PIL import Image
import h5py
import configparser
import SimpleITK as sitk
import time
from random import shuffle


config_file_name = "ParameterConf.conf"
'''    
h5_data_[S][A][M][N]
    S: mean substraction
    A: data augmentation
    M: keep margins
    N: normalize
'''

def flip(inputs, labels, axis):
    '''
    axis : integer. Axis in array, which entries are reversed.
    '''
    return np.fliplr(inputs), np.fliplr(labels)

def rotate(inputs, labels, num_of_rots, axes):
    '''
    num_of_rots : integer. Number of times the array is rotated by 90 degrees.
    axes : (2,) array_like. The array is rotated in the plane defined by the axes. Axes must be different.
    '''
    return np.rot90(inputs, num_of_rots), np.rot90(labels, num_of_rots)


def convert_labels(labels):
    '''
    function that converts 0:background / 10:CSF / 150:GM / 250:WM to 0/1/2/3
    '''
    D, H, W, C = labels.shape
    for d in range(D):
        for h in range(H):
            for w in range(W):
                for c in range(C):
                    if labels[d,h,w,c] == 10:
                        labels[d,h,w,c] = 1
                    elif labels[d,h,w,c] == 150:
                        labels[d,h,w,c] = 2
                    elif labels[d,h,w,c] == 250:
                        labels[d,h,w,c] = 3


def build_h5_dataset():
    # n_samples = len(training_list)
    training_files = list()
    cf=configparser.ConfigParser()
    cf.read("ParameterConf.conf")
    data_path = cf.get("DataPreProcess", "data_path")
    file = open(data_path, "r+")
    training_list = file.readlines()

    n_channels = cf.getint("DataPreProcess","n_channels")

    # get the file list, the row of training_files is the num of samples, the column is the num of files for each samples
    for patientname in training_list:
        if n_channels == 1:
            training_files.append((patientname.strip() + "/RMRI_roi.nii", patientname.strip() + "/RMRI_roi_Label.nii"))
        elif n_channels == 2:
            training_files.append((patientname.strip() + "/RMRI_roi.nii",patientname.strip() + "/RCT_roi.nii", patientname.strip() + "/RMRI_roi_Label.nii"))
        else:
            print("The number of channel is not valid!")

    MRI = training_files[0][0]
    tmp = sitk.ReadImage(MRI)
    mri_shape = tmp.GetSize()

    xmin = 0
    xmax = mri_shape[0]-1
    ymin = 0
    ymax = mri_shape[1]-1
    zmin = 0
    zmax = mri_shape[2]-1

    SIZE = [xmin,xmax,ymin,ymax,zmin,zmax]
    print(SIZE)

    # depth , height , weight
    new_shape = (SIZE[1] - SIZE[0] + 1, SIZE[3] - SIZE[2] + 1, SIZE[5] - SIZE[4] + 1)
    print(new_shape)

    cf.set("DataPreProcess","origsize_xmin",str(xmin))
    cf.set("DataPreProcess", "origsize_xmax",str(xmax))
    cf.set("DataPreProcess","origsize_ymin",str(ymin))
    cf.set("DataPreProcess", "origsize_ymax",str(ymax))
    cf.set("DataPreProcess","origsize_zmin",str(zmin))
    cf.set("DataPreProcess", "origsize_zmax",str(zmax))
    cf.write(open("ParameterConf.conf", "w"))

    d_imgshape = (len(training_list), new_shape[0], new_shape[1], new_shape[2], n_channels)
    d_labelshape = (len(training_list), new_shape[0], new_shape[1], new_shape[2])

    d_imgshape_r1 = (len(training_list), new_shape[1], new_shape[0], new_shape[2], n_channels)
    d_labelshape_r1 = (len(training_list), new_shape[1], new_shape[0], new_shape[2])

    d_imgshape_r2 = (len(training_list), new_shape[2], new_shape[1], new_shape[0], n_channels)
    d_labelshape_r2 = (len(training_list), new_shape[2], new_shape[1], new_shape[0])

    d_imgshape_r3 = (len(training_list), new_shape[0], new_shape[2], new_shape[1], n_channels)
    d_labelshape_r3 = (len(training_list), new_shape[0], new_shape[2], new_shape[1])

    # target_path = config["target_path"]
    target_path = cf.get("DataPreProcess","target_path")
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    dataset = h5py.File(os.path.join(target_path, "data.h5"), 'w')

    dataset.create_dataset('X', d_imgshape, dtype='f')
    dataset.create_dataset('Y', d_labelshape, dtype='i')

    # AUGMENT = config["AUGMENT"]
    AUGMENT = cf.getboolean("DataPreProcess", "AUGMENT")

    if AUGMENT:
        # data after cutting, with flipping in first dim
        dataset_f1 = h5py.File(os.path.join(target_path, "data_flip1.h5"), 'w')
        dataset_f1.create_dataset('X', d_imgshape, dtype='f')
        dataset_f1.create_dataset('Y', d_labelshape, dtype='i')

        # data after cutting, with flipping in second dim
        dataset_f2 = h5py.File(os.path.join(target_path, "data_flip2.h5"), 'w')
        dataset_f2.create_dataset('X', d_imgshape, dtype='f')
        dataset_f2.create_dataset('Y', d_labelshape, dtype='i')

        # data after cutting, with flipping in third dim
        dataset_f3 = h5py.File(os.path.join(target_path, "data_flip3.h5"), 'w')
        dataset_f3.create_dataset('X', d_imgshape, dtype='f')
        dataset_f3.create_dataset('Y', d_labelshape, dtype='i')

        # data after cutting, with rotating k=1 axes=(0,1)
        dataset_r1_1 = h5py.File(os.path.join(target_path, "data_rotate1_1.h5"), 'w')
        dataset_r1_1.create_dataset('X', d_imgshape_r1, dtype='f')
        dataset_r1_1.create_dataset('Y', d_labelshape_r1, dtype='i')

        # data after cutting, with rotating k=2 axes=(0,1)
        dataset_r1_2 = h5py.File(os.path.join(target_path, "data_rotate1_2.h5"), 'w')
        dataset_r1_2.create_dataset('X', d_imgshape, dtype='f')
        dataset_r1_2.create_dataset('Y', d_labelshape, dtype='i')

        # data after cutting, with rotating k=3 axes=(0,1)
        dataset_r1_3 = h5py.File(os.path.join(target_path, "data_rotate1_3.h5"), 'w')
        dataset_r1_3.create_dataset('X', d_imgshape_r1, dtype='f')
        dataset_r1_3.create_dataset('Y', d_labelshape_r1, dtype='i')

        # data after cutting, with rotating k=1 axes=(0,2)
        dataset_r2_1 = h5py.File(os.path.join(target_path, "data_rotate2_1.h5"), 'w')
        dataset_r2_1.create_dataset('X', d_imgshape_r2, dtype='f')
        dataset_r2_1.create_dataset('Y', d_labelshape_r2, dtype='i')

        # data after cutting, with rotating k=2 axes=(0,2)
        dataset_r2_2 = h5py.File(os.path.join(target_path, "data_rotate2_2.h5"), 'w')
        dataset_r2_2.create_dataset('X', d_imgshape, dtype='f')
        dataset_r2_2.create_dataset('Y', d_labelshape, dtype='i')

        # data after cutting, with rotating k=3 axes=(0,2)
        dataset_r2_3 = h5py.File(os.path.join(target_path, "data_rotate2_3.h5"), 'w')
        dataset_r2_3.create_dataset('X', d_imgshape_r2, dtype='f')
        dataset_r2_3.create_dataset('Y', d_labelshape_r2, dtype='i')

        # data after cutting, with rotating k=1 axes=(1,2)
        dataset_r3_1 = h5py.File(os.path.join(target_path, "data_rotate3_1.h5"), 'w')
        dataset_r3_1.create_dataset('X', d_imgshape_r3, dtype='f')
        dataset_r3_1.create_dataset('Y', d_labelshape_r3, dtype='i')

        # data after cutting, with rotating k=2 axes=(1,2)
        dataset_r3_2 = h5py.File(os.path.join(target_path, "data_rotate3_2.h5"), 'w')
        dataset_r3_2.create_dataset('X', d_imgshape, dtype='f')
        dataset_r3_2.create_dataset('Y', d_labelshape, dtype='i')

        # data after cutting, with rotating k=3 axes=(1,2)
        dataset_r3_3 = h5py.File(os.path.join(target_path, "data_rotate3_3.h5"), 'w')
        dataset_r3_3.create_dataset('X', d_imgshape_r3, dtype='f')
        dataset_r3_3.create_dataset('Y', d_labelshape_r3, dtype='i')

    for i in range(len(training_list)):
        if n_channels == 1:
            MRI = training_files[i][0]
            img_MRI = sitk.ReadImage(MRI)
            print(img_MRI.GetSize())
            inputs_tmp_MRI = sitk.GetArrayFromImage(img_MRI)
            print(inputs_tmp_MRI.shape)
            print(sitk.GetImageFromArray(inputs_tmp_MRI).GetSize())
            inputs_tmp_MRI = np.transpose(inputs_tmp_MRI,(1,2,0))
            print(inputs_tmp_MRI.shape)
            f_l = training_files[i][1]
            img_l = sitk.ReadImage(f_l)
            labels = sitk.GetArrayFromImage(img_l)
            labels = np.transpose(labels,(1,2,0))
            inputs = inputs_tmp_MRI
            inputs = inputs[:,:,:,np.newaxis]
            mean = cf.getfloat("DataPreProcess", "new_mean_MRI")
            std = cf.getfloat("DataPreProcess", "new_std_MRI")
        elif n_channels == 2:
                MRI = training_files[i][0]
                img_MRI = sitk.ReadImage(MRI)
                inputs_tmp_MRI = sitk.GetArrayFromImage(img_MRI)
                inputs_tmp_MRI = np.transpose(inputs_tmp_MRI, (1, 2, 0))
                CT = training_files[i][1]
                img_CT = sitk.ReadImage(CT)
                inputs_tmp_CT = sitk.GetArrayFromImage(img_CT)
                inputs_tmp_CT = np.transpose(inputs_tmp_CT, (1, 2, 0))
                f_l = training_files[i][2]
                img_l = sitk.ReadImage(f_l)
                labels = sitk.GetArrayFromImage(img_l)
                labels = np.transpose(labels, (1, 2, 0))
                inputs = np.stack((inputs_tmp_MRI, inputs_tmp_CT),axis=3)
                mean = list()
                std = list()
                mean.append(cf.getfloat("DataPreProcess", "new_mean_MRI"))
                mean.append(cf.getfloat("DataPreProcess", "new_mean_CT"))
                std.append(cf.getfloat("DataPreProcess", "new_std_MRI"))
                std.append(cf.getfloat("DataPreProcess", "new_std_CT"))


        print(inputs[SIZE[0]:SIZE[1] + 1, SIZE[2]:SIZE[3] + 1, SIZE[4]:SIZE[5] + 1,:].shape)

        inputs = inputs[SIZE[0]:SIZE[1] + 1, SIZE[2]:SIZE[3] + 1, SIZE[4]:SIZE[5] + 1,:].astype('float32')
        labels = labels[SIZE[0]:SIZE[1] + 1, SIZE[2]:SIZE[3] + 1, SIZE[4]:SIZE[5] + 1].reshape(new_shape)

        # NORMALIZE = config["NORMALIZE"]
        # mean = config["new_mean"]
        # std = config["new_std"]

        NORMALIZE = cf.getint("DataPreProcess", "NORMALIZE")
        # mean = np.array(cf.get("DataPreProcess", "new_mean"))
        # std = cf.getfloat("DataPreProcess", "new_std")

        if NORMALIZE == 2:
            inputs -= mean
            inputs /= std
        elif NORMALIZE == 1:
              max_ary = np.ones_like(inputs)*np.max(inputs)
              inputs = inputs/max_ary

        dataset['X'][i] = list(inputs)
        dataset['Y'][i] = list(labels)

# if __name__ == '__main__':
#     build_h5_dataset(data_path, target_path)
def gen_train_valid_samples():
    cf = configparser.ConfigParser()
    cf.read("ParameterConf.conf")
    data_dir = cf.get("DataPreProcess", "target_path")
    patch_size = cf.getint("Train","patch_size")
    num_patch = cf.getint("Train", "num_patch")
    pos_num_percent = cf.getfloat("Train", "pos_num_percent")

    # patch size
    depth = patch_size
    height = patch_size
    width = patch_size

    n_channels = cf.getint("DataPreProcess","n_channels")
    # load labels
    data_file = h5py.File(os.path.join(data_dir, 'data.h5'), 'r')
    train_inputs, train_labels = np.array(data_file['X']), np.array(data_file['Y'])

    num_file = train_inputs.shape[0]

    npatches_npc = int(num_patch * pos_num_percent)  # len(bgidx[0]) # bg patches number
    npatches_bg = int(num_patch * (1 - pos_num_percent))  # len(npcidx[0])  # npc patches number

    dataset = h5py.File(os.path.join(data_dir, "train_idx.h5"), 'w')
    # save idx
    d_pos = (1,npatches_npc*num_file,4)
    d_neg = (1,npatches_bg*num_file,4)

    dataset.create_dataset('pX', d_pos, dtype='i')#positive data
    dataset.create_dataset('nX', d_neg, dtype='i')#negative data
    negative_idxs = list()
    positive_idxs = list()

    for n in range(num_file):

        data = train_inputs[n, :, :, :, :]  # num,d,h,w,ch
        label = train_labels[n, :, :, :]  # num,d,h,w
        segitkDistanceMap = sitk.SignedMaurerDistanceMap(sitk.GetImageFromArray(label), \
                                                         insideIsPositive=False, squaredDistance=False,
                                                         useImageSpacing=False)
        segitkDistanceMap = sitk.GetArrayFromImage(segitkDistanceMap)
        print(segitkDistanceMap.shape)
        [d, h, w, _] = data.shape

        patch_sz = [depth, height, width]
        all_organs_mask = label > 0

        maskborders = np.ones_like(all_organs_mask)
        maskborders[d - int(patch_sz[0] / 2):d , :, :] = 0
        maskborders[:, h - int(patch_sz[1] / 2):h , :] = 0
        maskborders[:, :, w - int(patch_sz[2] / 2):w ] = 0

        maskborders[0:0 + int(patch_sz[0] / 2), :, :] = 0
        maskborders[:, 0:0 + int(patch_sz[1] / 2), :] = 0
        maskborders[:, :, 0:0 + int(patch_sz[2] / 2)] = 0

        maskbody = np.logical_and(data[:,:,:,0], maskborders)
        mask_npc = (label == 1) * maskborders

        segitkDistanceMask1 = segitkDistanceMap > 0
        segitkDistanceMask2 = segitkDistanceMap < 10
        finalsegitkDistanceMask = np.logical_and(segitkDistanceMask1, segitkDistanceMask2)
        bgidx = np.where(
            np.logical_and(np.logical_and(maskbody, np.logical_not(all_organs_mask) > 0), finalsegitkDistanceMask))
        npcidx = np.where(mask_npc == 1)

        num_rndbg = len(bgidx[0])
        num_rndnpc = len(npcidx[0])
        print("num_rndbg:", num_rndbg)
        print("num_rndnpc:", num_rndnpc)

        # ==============================================================================
        list_rndbg = np.random.choice(len(bgidx[0]), num_rndbg, replace=False)
        list_rndnpc = np.random.choice(len(npcidx[0]), num_rndnpc, replace=False)
        # ==============================================================================

        half_patch_sz_x = int(patch_sz[0] / 2)
        half_patch_sz_y = int(patch_sz[1] / 2)
        half_patch_sz_z = int(patch_sz[2] / 2)

        print("----------------select negative sample-------------------")

        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

        patchid = 0

        for i in list_rndbg:
            index = np.array((bgidx[0][i], bgidx[1][i], bgidx[2][i]))
            # select negative 3D patch
            tempSeg = label[index[0] - half_patch_sz_x:index[0] + half_patch_sz_x, \
                      index[1] - half_patch_sz_y:index[1] + half_patch_sz_y, \
                      index[2] - half_patch_sz_z:index[2] + half_patch_sz_z]
            iszero = (tempSeg == 0)
            numOfzero = np.count_nonzero(iszero)
            percent = 1.0 * numOfzero / (patch_sz[0] * patch_sz[1] * patch_sz[2])

            if percent >= 0.9:
                tempPatch = data[index[0] - half_patch_sz_x:index[0] + half_patch_sz_x, \
                            index[1] - half_patch_sz_y:index[1] + half_patch_sz_y, \
                            index[2] - half_patch_sz_z:index[2] + half_patch_sz_z,:]

                negative_idxs.append([n, index[0], index[1], index[2]])
                patchid = patchid + 1
                if patchid > (npatches_bg - 1):
                    break

        print("----------------select positive sample-------------------")

        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        patchid = 0

        for i in list_rndnpc:
            index = np.array((npcidx[0][i], npcidx[1][i], npcidx[2][i]))
            # select positive 3D patch
            print(index[0] - half_patch_sz_x,index[0] + half_patch_sz_x)
            print(index[1] - half_patch_sz_y,index[1] + half_patch_sz_y)
            print(index[2] - half_patch_sz_y,index[2] + half_patch_sz_y)
            tempSeg = label[index[0] - half_patch_sz_x:index[0] + half_patch_sz_x, \
                      index[1] - half_patch_sz_y:index[1] + half_patch_sz_y, \
                      index[2] - half_patch_sz_y:index[2] + half_patch_sz_y]
            print(label[index[0],index[1],index[2]])
            isone = (tempSeg == 1)
            numOfzero = np.count_nonzero(isone)
            percent = 1.0 * numOfzero / (patch_sz[0] * patch_sz[1] * patch_sz[2])

            if percent >= 0.015:
                positive_idxs.append([n, index[0], index[1], index[2]])
                patchid = patchid + 1
                if patchid > (npatches_npc - 1):
                    break

    shuffle(negative_idxs)
    dataset['nX'][0] = negative_idxs

    shuffle(positive_idxs)
    dataset['pX'][0] = positive_idxs




