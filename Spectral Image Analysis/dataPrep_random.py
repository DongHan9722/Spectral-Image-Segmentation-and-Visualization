import os, sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import keras
from skimage.transform import resize

from keras_unet_collection import models
from spectral_tiffs import read_stiff, read_mtiff

IMG_SIZE = (128, 128)

def make_segmentation_test_data(dir_name, save_state):
    files = []
    spim_list = []
    rgb_list = []
    for i, filename in enumerate(os.listdir(dir_name)):
        extension = os.path.splitext(filename)[1][1:]

        if extension == 'tif':
            filener = filename.replace('.tif', '')
            files.append(filener)
            num_img = read_stiff(dir_name + os.sep + filename)
            spim_list.append(resize(num_img[0], IMG_SIZE))
            rgb_list.append(num_img[2])
            
    data_array = np.array(spim_list)
    rgb_array = np.array(rgb_list)
    if save_state:
        if not os.path.exists('data_arrays'):
            os.makedirs('data_arrays')
        np.save('data_arrays/all_data', data_array)
    return data_array, files, rgb_array

def make_training_data(img_dir_name, mask_dir_name, save_state):
    labels_dict = []
    files = []
    for filename in os.listdir(mask_dir_name):
        img_mask = read_mtiff(mask_dir_name + os.sep + filename)
        labels_dict = labels_dict + list(img_mask.keys())
    img_shape = img_mask[list(img_mask.keys())[0]].shape

    labels_dict = np.unique(np.array(labels_dict))

    labels_order = [0, 10, 1, 6, 7, 9, 8, 5, 2, 3, 4] # set priority order for labels

    labels_dict = labels_dict[labels_order]

    spim_list = []
    label_list = []
    files = []

    for i, filename in enumerate(os.listdir(img_dir_name)):
        files.append(filename)
        num_img = read_stiff(img_dir_name + os.sep + filename)
        mask_img = read_mtiff(mask_dir_name + os.sep + filename.replace('.tif', '_masks.tif'))

        Y = np.zeros(img_shape)
        for lbl in list(mask_img.keys()):
            Y = Y + mask_img[lbl] * (np.where(labels_dict == lbl)[0][0] + 1) * (Y < 1)

        spim_list.append(resize(num_img[0], IMG_SIZE))
        label_list.append(resize(Y, IMG_SIZE))

    data_array = np.array(spim_list)
    label_array = np.array(label_list)

    # randomly shuffle dataset
    shuffled_indices = np.arange(data_array.shape[0])
    np.random.shuffle(shuffled_indices)

    data_array = data_array[shuffled_indices]
    label_array = label_array[shuffled_indices]

    label_onehot = np.zeros((label_array.shape + (len(labels_dict) + 1,)))

    for i in range(label_array.shape[0]):
        for j in range(label_array.shape[1]):
            label_onehot[i][j] = np.eye(12)[label_array[i][j].astype(int)]

    (train_perc, val_perc) = (0.70, 0.14) #train, val split
    (train_size, val_size) = (int(label_array.shape[0] * train_perc), int(label_array.shape[0] * val_perc))
    test_size = label_array.shape[0] - train_size - val_size

    #validation set
    val_items = data_array[:val_size]
    val_labels = label_onehot[:val_size]

    #test set
    test_items = data_array[val_size:val_size + test_size]
    test_labels = label_onehot[val_size:val_size + test_size]

    #train set
    train_items = data_array[val_size + test_size:]
    train_labels = label_onehot[val_size + test_size:]

    # save to .npy files
    if save_state:
        if not os.path.exists('data_arrays'):
            os.makedirs('data_arrays')
        np.save('data_arrays/train_data', train_items)
        np.save('data_arrays/train_labels', train_labels)
        np.save('data_arrays/val_data', val_items)
        np.save('data_arrays/val_labels', val_labels)
        np.save('data_arrays/test_data', test_items)
        np.save('data_arrays/test_labels', test_labels)

    return(train_items, train_labels, val_items, val_labels, test_items, test_labels)
