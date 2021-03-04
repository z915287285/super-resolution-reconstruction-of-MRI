from skimage import io
import SimpleITK as sitk
import os
import numpy as np
import scipy
from scipy.ndimage import zoom


config=dict()

data_path="F:\AI object\zhu\\rawdata\\"
data_file="datalist.txt"

def main():
    #Get data_path
    dataList=open(data_file,"w+")
    datalist=list()
    for root,patients,filename in os.walk(data_path):
        for patientname in patients:
            path=data_path+patientname
            datalist.append(path+"\n")
            print(path)
    dataList.writelines(datalist)
    dataList.close()


    data_files=list()
    mask_files=list()

    file=open(data_file,"r+")
    data_list=file.readlines()
    #Get data and mask_data
    for i in data_list:
        data_files.append(i.strip()+"/brain.nii")

    for j in datalist:
        mask_files.append(j.strip()+"/brain_mask.nii")

    #Begin get image
    for z in range(len(data_list)):
        MRI=data_files[z]
        m=mask_files[z]
        tmp=sitk.ReadImage(MRI)
        data=sitk.GetArrayFromImage(tmp)
        data=np.transpose(data,(1,2,0))
        data=data[0:300,0:258,0:258]
        # d=data
        # label=scipy.ndimage.zoom(d,1/2,order=2)
        # label_s=scipy.ndimage.zoom(label,2,order=2)
        #BN
        max_inp=np.ones_like(data)*np.max(data)
        min_inp=np.min(data)
        inputs_1=(data-min_inp)/(max_inp-min_inp)
        # inputs_1=inputs_1*255

        # max_int=np.ones_like(label_s)*np.max(label_s)
        # min_int=np.min(label_s)
        # labels_1=(label_s-min_int)/(max_int-min_int)
        # labels_1=labels*255

        tmp_1=sitk.ReadImage(m)
        mask=sitk.GetArrayFromImage(tmp_1)
        mask=np.transpose(mask,(1,2,0))
        mask=mask[0:300,0:258,0:258]
        #get image from slices
        for y in range(data.shape[0]):
            mask_1=mask[y:y+1,:,:]
            m1=mask_1
            m2=mask_1
            m_zero=m1[m1==0]
            m_one=m2[m2==1]
            rate=m_one.size/(m_one.size+m_zero.size)
            if rate>0.3:
               H_image=inputs_1[y:y+1,:,:]
               # L_image=labels_1[y:y+1,:,:]
               H_image=np.squeeze(H_image)
               # L_image=np.squeeze(L_image)
               io.imsave("F:\AI object\zhu\images\\"+str(z)+'_'+str(y)+'_'+str(rate)+'.png',H_image)
               # io.imsave("/home/kevin/zhc/data_image/Low/"+str(z)+'_'+str(y)
               #           +'_'+str(rate)+'_LowImage.png',L_image)
            else:
                continue




if __name__ == '__main__':
    main()