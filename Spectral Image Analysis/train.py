import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import matplotlib.pyplot as plt

import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, Dropout, Activation, UpSampling2D, GlobalMaxPooling2D, multiply
from tensorflow.keras.backend import max

from keras_unet_collection import models, base, utils, losses

from dataPrep_random import make_training_data

import argparse

parser = argparse.ArgumentParser(description='Image segmentation tool')
parser.add_argument('--model', metavar='Name', type=str, default='swinUnet',
                    help='model to use for segmentation')
parser.add_argument('--img_dir', metavar='PATH', type=str, default='./spectral',
                    help='path to images to be segmented')
parser.add_argument('--mask_dir', metavar='PATH', type=str, default='./mask',
                    help='path to mask images')


args = parser.parse_args()

# GPU memory settings
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
session = tf.compat.v1.InteractiveSession(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

model_name = args.model
dir_name = args.img_dir
mask_dir_name = args.mask_dir

# read dataset
make_training_data(dir_name, mask_dir_name, True)

valid_input = np.load('data_arrays/val_data.npy')
valid_target = np.load('data_arrays/val_labels.npy')
test_input = np.load('data_arrays/test_data.npy')
test_target = np.load('data_arrays/test_labels.npy')
train_input1 = np.load('data_arrays/train_data.npy')
train_target1 = np.load('data_arrays/train_labels.npy')
train_input = np.load('data_arrays/train_data.npy')
train_target = np.load('data_arrays/train_labels.npy')

# Declare model

def hybrid_loss(y_true, y_pred):

    loss_focal = losses.focal_tversky(y_true, y_pred, alpha=0.5, gamma=4/3)
    loss_iou = losses.iou_seg(y_true, y_pred)

    return loss_focal+loss_iou #+loss_ssim

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
    prediction_model.compile(loss=[hybrid_loss, hybrid_loss, hybrid_loss, hybrid_loss, hybrid_loss],
                      loss_weights=[0.25, 0.25, 0.25, 0.25, 1.0],
                      optimizer=keras.optimizers.Adam(learning_rate=1e-4))

    N_epoch = 1000 # number of epoches
    N_batch = 30 # number of batches per epoch
    N_sample = 10 # number of samples per batch

    tol = 0 # current early stopping patience
    max_tol = 50 # the max-allowed early stopping patience
    min_del = 0 # the lowest acceptable loss value reduction
    L_train = 30
    loss_buffer = []
    accuracies = []

    # loop over epoches
    for epoch in range(N_epoch):

        # initial loss record
        if epoch == 0:
            temp_out = prediction_model.predict([valid_input])
            y_pred = temp_out[-1]
            record = np.mean(hybrid_loss(valid_target, y_pred))
            print('\tInitial loss = {}'.format(record))

        # loop over batches
        for step in range(N_batch):
            # selecting smaples for the current batch
            ind_train_shuffle = utils.shuffle_ind(L_train)[:N_sample]

            train_input = np.load('data_arrays/train_data.npy')[ind_train_shuffle]
            train_target = np.load('data_arrays/train_labels.npy')[ind_train_shuffle]

            # train on batch
            loss_ = prediction_model.train_on_batch([train_input,],
                                             [train_target, train_target, train_target, train_target, train_target,])

        # epoch-end validation
        temp_out = prediction_model.predict([valid_input])
        y_pred = temp_out[-1]
        record_temp = np.mean(hybrid_loss(valid_target, y_pred))
        loss_buffer.append(record_temp)
        # ** validation loss is not stored ** #

        # training accuracy log
        temp_out = prediction_model.predict([train_input1])
        y_pred = temp_out[-1]
        y_pred_lbl = np.zeros((y_pred.shape[0], y_pred.shape[1], y_pred.shape[2]))
        for insta in range(y_pred.shape[0]):
            y_pred_lbl[insta] = np.argmax(y_pred[insta], axis = 2)
        train_target_lbl = np.zeros((y_pred.shape[0], 128, 128))
        for insta in range(y_pred.shape[0]):
            train_target_lbl[insta] = np.argmax(train_target1[insta], axis = 2)
        temp_acc = 0
        for insta in range(y_pred.shape[0]):
            temp_acc = temp_acc + (np.where(y_pred_lbl[insta] == train_target_lbl[insta])[0].size / (128 * 128))
        temp_acc = temp_acc / y_pred.shape[0]
        accuracies.append(temp_acc)

        # if loss is reduced
        if record - record_temp > min_del:
            print('Validation performance is improved from {} to {}'.format(record, record_temp))
            record = record_temp; # update the loss record
            tol = 0;

        else:
            print('Validation performance {} is NOT improved'.format(record_temp))
            tol += 1
            if tol >= max_tol:
                print('Early stopping')
                break;
            else:
                # Pass to the next epoch
                continue;

    if not os.path.exists('training_results'):
        os.makedirs('training_results')

    # Plot training loss
    plt.plot(loss_buffer)
    plt.rcParams['figure.figsize'] = (12, 5)
    plt.title('UNet 3+ Training Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch #')
    plt.legend(['Training Loss'])
    plt.savefig('training_results/unet3plus_loss_plot.png')
    plt.clf()

    # Plot training accuracy
    plt.plot(accuracies)
    plt.rcParams['figure.figsize'] = (12, 5)
    plt.title('UNet 3+ Training Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch #')
    plt.legend(['Training Accuracy'])
    plt.savefig('training_results/unet3plus_accuracy_plot.png')
    plt.clf()

    # Evaluate final
    temp_out = prediction_model.predict([test_input,])
    y_pred = temp_out[-1]
    print('Testing set IoU loss = {}'.format(np.mean(losses.iou_seg(test_target, y_pred))))

    # Save trained weights
    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')
    prediction_model.save_weights('saved_models/unet3plus_weights.h5')


elif model_name == 'swinUnet':
    prediction_model = models.swin_unet_2d((128, 128, 38), filter_num_begin=64, n_labels=12, depth=4, stack_num_down=2, stack_num_up=2,
                            patch_size=(2, 2), num_heads=[4, 8, 8, 8], window_size=[4, 2, 2, 2], num_mlp=512,
                            output_activation='Softmax', shift_window=True, name='prediction_model')
    prediction_model.compile(loss=losses.iou_seg, optimizer = keras.optimizers.Adam(learning_rate=1e-4), metrics = [keras.metrics.CategoricalAccuracy()])


    valid_input = np.expand_dims(valid_input, axis = 1)
    valid_target = np.expand_dims(valid_target, axis = 1)
    test_input = np.expand_dims(test_input, axis = 1)
    test_target = np.expand_dims(test_target, axis = 1)
    train_input = np.expand_dims(train_input, axis = 1)
    train_target = np.expand_dims(train_target, axis = 1)

    train_dataset = tf.data.Dataset.from_tensor_slices((train_input, train_target))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_input, test_target))
    val_dataset = tf.data.Dataset.from_tensor_slices((valid_input, valid_target))

    history = prediction_model.fit(train_dataset, epochs = 1000, validation_data = val_dataset, verbose = 1)

    if not os.path.exists('training_results'):
        os.makedirs('training_results')

    # Plot training loss
    plt.plot(history.history['loss'])
    plt.rcParams['figure.figsize'] = (12, 5)
    plt.title('Swin-Unet Training Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch #')
    plt.legend(['Training Loss'])
    plt.savefig('training_results/swinUnet_loss_plot.png')
    plt.clf()

    # Plot training accuracy
    plt.plot(history.history['categorical_accuracy'])
    plt.rcParams['figure.figsize'] = (12, 5)
    plt.title('Swin-Unet Training Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch #')
    plt.legend(['Training Accuracy'])
    plt.savefig('training_results/swinUnet_accuracy_plot.png')
    plt.clf()

    # Evaluate final
    y_pred = prediction_model.predict(test_dataset)
    test_target = np.load('data_arrays/test_labels.npy')
    print('Testing set IoU loss = {}'.format(np.mean(losses.iou_seg(test_target, y_pred))))

    # Save trained weights
    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')
    prediction_model.save_weights('saved_models/swin_weights.h5')
