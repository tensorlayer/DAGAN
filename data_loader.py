import pickle
import tensorlayer as tl
import numpy as np
import os
import nibabel as nib

training_data_path = "data/MICCAI13_SegChallenge/Training_100"
testing_data_path = "data/MICCAI13_SegChallenge/Testing_50"
val_ratio = 0.3
seed = 100
preserving_ratio = 0.1 # filter out 2d images containing < 10% non-zeros


f_train_all = tl.files.load_file_list(path=training_data_path,
                                      regx='.*.gz',
                                      printable=False)
train_all_num = len(f_train_all)
val_num = int(train_all_num * val_ratio)

f_train = []
f_val = []

val_idex = tl.utils.get_random_int(min=0,
                                   max=train_all_num - 1,
                                   number=val_num,
                                   seed=seed)
for i in range(train_all_num):
    if i in val_idex:
        f_val.append(f_train_all[i])
    else:
        f_train.append(f_train_all[i])

f_test = tl.files.load_file_list(path=testing_data_path,
                                 regx='.*.gz',
                                 printable=False)

train_3d_num, val_3d_num, test_3d_num = len(f_train), len(f_val), len(f_test)


X_train = []
for fi, f in enumerate(f_train):
    print("processing [{}/{}] 3d image ({}) for training set ...".format(fi + 1, train_3d_num, f))
    img_path = os.path.join(training_data_path, f)
    img = nib.load(img_path).get_data()
    img_3d_max = np.max(img)
    img = img / img_3d_max * 255
    for i in range(img.shape[2]):
        img_2d = img[:, :, i]
        # filter out 2d images containing < 10% non-zeros
        if float(np.count_nonzero(img_2d)) / img_2d.size >= preserving_ratio:
            img_2d = img_2d / 127.5 - 1
            img_2d = np.transpose(img_2d, (1, 0))
            X_train.append(img_2d)

X_val = []
for fi, f in enumerate(f_val):
    print("processing [{}/{}] 3d image ({}) for validation set ...".format(fi + 1, val_3d_num, f))
    img_path = os.path.join(training_data_path, f)
    img = nib.load(img_path).get_data()
    img_3d_max = np.max(img)
    img = img / img_3d_max * 255
    for i in range(img.shape[2]):
        img_2d = img[:, :, i]
        # filter out 2d images containing < 10% non-zeros
        if float(np.count_nonzero(img_2d)) / img_2d.size >= preserving_ratio:
            img_2d = img_2d / 127.5 - 1
            img_2d = np.transpose(img_2d, (1, 0))
            X_val.append(img_2d)

X_test = []
for fi, f in enumerate(f_test):
    print("processing [{}/{}] 3d image ({}) for test set ...".format(fi + 1, test_3d_num, f))
    img_path = os.path.join(testing_data_path, f)
    img = nib.load(img_path).get_data()
    img_3d_max = np.max(img)
    img = img / img_3d_max * 255
    for i in range(img.shape[2]):
        img_2d = img[:, :, i]
        # filter out 2d images containing < 10% non-zeros
        if float(np.count_nonzero(img_2d)) / img_2d.size >= preserving_ratio:
            img_2d = img_2d / 127.5 - 1
            img_2d = np.transpose(img_2d, (1, 0))
            X_test.append(img_2d)

X_train = np.asarray(X_train)
X_train = X_train[:, :, :, np.newaxis]
X_val = np.asarray(X_val)
X_val = X_val[:, :, :, np.newaxis]
X_test = np.asarray(X_test)
X_test = X_test[:, :, :, np.newaxis]

# save data into pickle format
data_saving_path = 'data/MICCAI13_SegChallenge/'
tl.files.exists_or_mkdir(data_saving_path)

print("save training set into pickle format")
with open(os.path.join(data_saving_path, 'training.pickle'), 'wb') as f:
    pickle.dump(X_train, f, protocol=4)

print("save validation set into pickle format")
with open(os.path.join(data_saving_path, 'validation.pickle'), 'wb') as f:
    pickle.dump(X_val, f, protocol=4)

print("save test set into pickle format")
with open(os.path.join(data_saving_path, 'testing.pickle'), 'wb') as f:
    pickle.dump(X_test, f, protocol=4)

print("processing data finished!")
