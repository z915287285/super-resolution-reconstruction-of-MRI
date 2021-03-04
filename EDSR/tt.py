import h5py
import numpy as np
import glob
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
for j in range(len(x_list)):
	if j == 0:
		input_val = x_list[j]
		label_val = y_list[j]
		continue
	input_val = np.concatenate((input_val, x_list[j]))
	label_val = np.concatenate((label_val, y_list[j]))