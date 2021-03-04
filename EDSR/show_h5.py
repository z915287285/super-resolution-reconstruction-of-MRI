import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
data_path = './'
data_path_files = h5py.File(os.path.join(data_path, 'data_train_re.h5'), 'r')
x = np.array(data_path_files['val_inpt'])
y = np.array(data_path_files['val_labl'])
input=x
label=y
count=1
fig=plt.figure()
m=len(input)
n=2
for j in range(len(x)):
    in1=x[j]
    lab1=y[j]
    fig.add_subplot(m,n,count)
    plt.plot(in1[:, :, 0])
    print count
    count=count+1
    fig.add_subplot(m, n, count)
    plt.plot(lab1[:, :, 0])
    print count
    count=count+1
    if count ==5:
        break
plt.show()
