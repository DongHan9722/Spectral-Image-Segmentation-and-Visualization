import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import matplotlib.pyplot as plt
import matplotlib

import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, Dropout, Activation, UpSampling2D, GlobalMaxPooling2D, multiply
from tensorflow.keras.backend import max
from skimage.transform import resize

from keras_unet_collection import models, base, utils

from dataPrep_random import make_segmentation_test_data

import argparse

parser = argparse.ArgumentParser(description='Image segmentation tool')
parser.add_argument('--model', metavar='Name', type=str, default='swinUnet',
                    help='model to use for segmentation')
parser.add_argument('--img_dir', metavar='PATH', type=str, default='./spectral',
                    help='path to images to be segmented')

args = parser.parse_args()

# GPU memory settings
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
session = tf.compat.v1.InteractiveSession(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

model_name = args.model
dir_name = args.img_dir

# read data to segment
data_array, filenames, rgb_array = make_segmentation_test_data(dir_name, True)

# Declare prediction model

if model_name == 'unet3plus':
    name = 'prediction_model'
    activation = 'ReLU'
    filter_num_down = [32, 64, 128, 256, 512]
    filter_num_skip = [32, 32, 32, 32]
    filter_num_aggregate = 160

    stack_num_down = 2
    stack_num_up = 1
    n_labels = 12

    # ---------------------------------------- #
    input_tensor = keras.layers.Input((128, 128, 38))
    # base architecture
    X_decoder = base.unet_3plus_2d_base(
        input_tensor, filter_num_down, filter_num_skip, filter_num_aggregate,
        stack_num_down=stack_num_down, stack_num_up=stack_num_up, activation=activation,
        batch_norm=True, pool=True, unpool=True, backbone=None, name=name)

    # allocating deep supervision tensors
    OUT_stack = []
    # reverse indexing `X_decoder`, so smaller tensors have larger list indices
    X_decoder = X_decoder[::-1]

    # deep supervision outputs
    for i in range(1, len(X_decoder)):
        # 3-by-3 conv2d --> upsampling --> sigmoid output activation
        pool_size = 2**(i)
        X = Conv2D(n_labels, 3, padding='same', name='{}_output_conv1_{}'.format(name, i-1))(X_decoder[i])

        X = UpSampling2D((pool_size, pool_size), interpolation='bilinear',
                         name='{}_output_sup{}'.format(name, i-1))(X)

        X = Activation('sigmoid', name='{}_output_sup{}_activation'.format(name, i-1))(X)
        # collecting deep supervision tensors
        OUT_stack.append(X)

    # the final output (without extra upsampling)
    # 3-by-3 conv2d --> sigmoid output activation
    X = Conv2D(n_labels, 3, padding='same', name='{}_output_final'.format(name))(X_decoder[0])
    X = Activation('sigmoid', name='{}_output_final_activation'.format(name))(X)
    # collecting final output tensors
    OUT_stack.append(X)

    X_CGM = X_decoder[-1]
    X_CGM = Dropout(rate=0.1)(X_CGM)
    X_CGM = Conv2D(filter_num_skip[-1], 1, padding='same')(X_CGM)
    X_CGM = GlobalMaxPooling2D()(X_CGM)
    X_CGM = Activation('sigmoid')(X_CGM)

    CGM_mask = max(X_CGM, axis=-1)

    for i in range(len(OUT_stack)):
        if i < len(OUT_stack)-1:
            # deep-supervision
            OUT_stack[i] = multiply([OUT_stack[i], CGM_mask], name='{}_output_sup{}_CGM'.format(name, i))
        else:
            # final output
            OUT_stack[i] = multiply([OUT_stack[i], CGM_mask], name='{}_output_final_CGM'.format(name))

    prediction_model = keras.models.Model([input_tensor,], OUT_stack)
    prediction_model.load_weights('saved_models/unet3plus_weights.h5')
    # predicted output
    y_pred = prediction_model.predict([data_array,])
    y_pred = y_pred[-1]

elif model_name == 'swinUnet':
    prediction_model = models.swin_unet_2d((128, 128, 38), filter_num_begin=64, n_labels=12, depth=4, stack_num_down=2, stack_num_up=2,
                            patch_size=(2, 2), num_heads=[4, 8, 8, 8], window_size=[4, 2, 2, 2], num_mlp=512,
                            output_activation='Softmax', shift_window=True, name='prediction_model')
    prediction_model.load_weights('saved_models/swin_weights.h5')
    foo_label = np.zeros((data_array.shape[0], 128, 128, 12))
    data_array = np.expand_dims(data_array, axis = 1)
    foo_label = np.expand_dims(foo_label, axis = 1)
    input_data = tf.data.Dataset.from_tensor_slices((data_array, foo_label))
    # redicted output
    y_pred = prediction_model.predict(input_data)

num_labels = 12
y_pred_label = np.zeros((y_pred.shape[0], y_pred.shape[1], y_pred.shape[2]))

for i in range(y_pred.shape[0]):
    y_pred_label[i] = np.argmax(y_pred[i], axis = 2)

y_pred_label_plot = np.zeros((y_pred_label.shape[0], 1024, 1024))
y_pred_class_plot = np.zeros((y_pred_label.shape[0], num_labels, 1024, 1024))

for i in range(y_pred.shape[0]):
    y_pred_label_plot[i] = resize(y_pred_label[i], (1024, 1024)).astype(int)
    #y_pred_class_plot[i][j] = resize(y_pred_class[i][j], (1024, 1024))
    for j in range(num_labels):
        y_pred_class_plot[i, j, :, :] = (y_pred_label_plot[i] == j) * j

colors = np.array([[0, 0, 0],
                [237, 28, 36],
                [63, 72, 204],
                [34, 177, 76],
                [128, 128, 128],
                [255, 255, 255],
                [255, 242, 0],
                [128, 128, 0],
                [191, 91, 22],
                [0, 255, 255],
                [0, 255, 0],
                [128, 0, 0]
                ])

colors_individual = np.zeros((12, 2, 3))
for i in range(colors_individual.shape[0]):
    colors_individual[i] = np.array([colors[0], colors[i]])

colors = np.array([colors[0], colors[1], colors[2], colors[3], colors[4], colors[5], colors[6], colors[7], colors[8], colors[9], colors[10], colors[11]])
colors = colors / 255.0
colors_individual = colors_individual / 255.0

label_names = ['Background', 'Artery', 'Vein', 'Artery, ICG', 'Stroma', 'Stroma, ICG',
       'Umbilical cord', 'Suture', 'Specular reflection', 'Blue dye',
       'ICG', 'Red dye']

label_names1 = ['Background', 'Blue dye', 'ICG', 'Specular reflection', 'Artery', 'Vein', 'Stroma', 'Artery, ICG', 'Stroma, ICG', 'Suture', 'Umbilical cord', 'Red dye']

if not os.path.exists('output_images'):
    os.makedirs('output_images')

for i in range(y_pred.shape[0]):
    if not os.path.exists(f'output_images/{filenames[i]}'):
        os.makedirs(f'output_images/{filenames[i]}')
    colors_foo = np.array([[0, 0, 0]])
    for labeli in np.unique(y_pred_label_plot[i]):
        if labeli.astype(int) != 0:
            colors_foo = np.append(colors_foo, [colors[labeli.astype(int)]], axis = 0)
    cmap =  matplotlib.colors.ListedColormap(colors_foo)
    #print(rgb_array[i].shape)
    plt.imshow(rgb_array[i])
    plt.draw()
    plt.pause(0.001)
    plt.imsave(f'output_images/{filenames[i]}/{filenames[i]}.png', y_pred_label_plot[i], cmap = cmap)
    for j in range(num_labels):
        cmap_sub = matplotlib.colors.ListedColormap(colors_individual[j])
        foo = label_names1.index(label_names[j])
        plt.imsave(f'output_images/{filenames[i]}/{foo}.png', y_pred_class_plot[i][j], cmap = cmap_sub)
    #plt.clf()
