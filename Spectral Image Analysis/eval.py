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
#make_training_data(dir_name, mask_dir_name, True)
if not os.path.exists('training_results'):
    os.makedirs('training_results')

valid_input = np.load('data_arrays/val_data.npy')
valid_target = np.load('data_arrays/val_labels.npy')
test_input = np.load('data_arrays/test_data.npy')
test_target = np.load('data_arrays/test_labels.npy')
train_input = np.load('data_arrays/train_data.npy')
train_target = np.load('data_arrays/train_labels.npy')
train_input1 = np.load('data_arrays/train_data.npy')
train_target1 = np.load('data_arrays/train_labels.npy')

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
    prediction_model.load_weights('saved_models/unet3plus_weights.h5')

    # Test set evaluation
    y_pred_test = prediction_model.predict([test_input, ])[-1]
    y_pred_test_tensor = tf.convert_to_tensor(y_pred_test)
    y_pred_test_tensor = tf.cast(y_pred_test_tensor, tf.float32)
    y_true_test = tf.cast(test_target, y_pred_test_tensor.dtype)
    y_pred_test_tensor = tf.squeeze(y_pred_test_tensor)
    y_true_test = tf.squeeze(y_true_test)

    temp_sum_test = 0
    ious_test = []
    for i in range(12):
        y_pred_pos_test = tf.reshape(y_pred_test_tensor[:, :, :, i], [-1])
        y_true_pos_test = tf.reshape(y_true_test[:, :, :, i], [-1])
        area_intersect = tf.reduce_sum(tf.multiply(y_true_pos_test, y_pred_pos_test))
        area_true = tf.reduce_sum(y_true_pos_test)
        area_pred = tf.reduce_sum(y_pred_pos_test)
        area_union = area_true + area_pred - area_intersect
        temp_sum_test = temp_sum_test + tf.math.divide_no_nan(area_intersect, area_union)
        ious_test.append(np.mean(tf.math.divide_no_nan(area_intersect, area_union)))

    # Full dataset evaluation
    all_input = np.append(np.append(valid_input, test_input, axis = 0), train_input, axis = 0)
    all_target = np.append(np.append(valid_target, test_target, axis = 0), train_target, axis = 0)

    y_pred = prediction_model.predict([all_input, ])[-1]
    y_pred_tensor = tf.convert_to_tensor(y_pred)
    y_pred_tensor = tf.cast(y_pred_tensor, tf.float32)
    y_true = tf.cast(all_target, y_pred_tensor.dtype)
    y_pred_tensor = tf.squeeze(y_pred_tensor)
    y_true = tf.squeeze(y_true)

    temp_sum = 0
    ious = []
    for i in range(12):
        y_true_pos = tf.reshape(y_true[:, :, :, i], [-1])
        y_pred_pos = tf.reshape(y_pred_tensor[:, :, :, i], [-1])
        area_intersect = tf.reduce_sum(tf.multiply(y_true_pos, y_pred_pos))
        area_true = tf.reduce_sum(y_true_pos)
        area_pred = tf.reduce_sum(y_pred_pos)
        area_union = area_true + area_pred - area_intersect
        temp_sum = temp_sum + tf.math.divide_no_nan(area_intersect, area_union)
        ious.append(np.mean(tf.math.divide_no_nan(area_intersect, area_union)))

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
    colors = colors / 255.0

    # IoU plot by class
    plt.figure(figsize = (20, 10))
    plt.rcParams['axes.facecolor'] = 'pink'
    plt.bar(np.arange(12), ious_test, color = colors)
    plt.yticks(fontsize = 17)
    plt.xticks(np.arange(12), ['Background', 'Artery', 'Vein', 'Artery, ICG', 'Stroma', 'Stroma, ICG', 'Umbil. cord', 'Suture', 'Sp.reflection', 'Blue dye', 'ICG', 'Red dye'], rotation = 75)
    plt.title('Test set IoU by class', fontsize = 20)
    plt.ylim(0, 1)
    plt.savefig('training_results/IoU_test_barplot.png', bbox_inches = 'tight')
    plt.clf()

    plt.figure(figsize = (20, 10))
    plt.rcParams['axes.facecolor'] = 'pink'
    plt.bar(np.arange(12), ious, color = colors)
    plt.yticks(fontsize = 17)
    plt.xticks(np.arange(12), ['Background', 'Artery', 'Vein', 'Artery, ICG', 'Stroma', 'Stroma, ICG', 'Umbil.', 'Suture', 'Sp.reflection', 'Blue dye', 'ICG', 'Red dye'], rotation = 75)
    plt.title('Whole set IoU by class', fontsize = 20)
    plt.ylim(0, 1)
    plt.savefig('training_results/IoU_whole_barplot.png', bbox_inches = 'tight')
    plt.clf()

    # Confusion matrix for test set
    y_test_label = np.zeros((y_pred_test.shape[0], y_pred_test.shape[1], y_pred_test.shape[2]))
    for i in range(y_pred_test.shape[0]):
        y_test_label[i] = np.argmax(y_pred_test[i], axis = 2)
    test_target_label = np.zeros(y_test_label.shape)
    for i in range(y_pred_test.shape[0]):
        test_target_label[i] = np.argmax(test_target[i], axis = 2)
    conf_mat_test = np.zeros((y_pred_test.shape[0], 12, 12))
    for k in range(y_pred_test.shape[0]):
        for i in range(12):
            for j in range(12):
                conf_mat_test[k][i][j] = np.where(np.logical_and(y_test_label[k] == i, test_target_label[k] == j))[0].shape[0]
    conf_mat_test = np.sum(conf_mat_test, axis = 0)
    conf_mat_test_print = conf_mat_test
    conf_mat_test = np.nan_to_num(conf_mat_test / np.sum(conf_mat_test, axis = 0))

    # Preccision & recall for test set
    predicted_num_test = np.sum(conf_mat_test_print.astype(int), axis = 1)
    correct_num_test = conf_mat_test_print.diagonal()
    individual_num_test = np.sum(conf_mat_test_print.astype(int), axis = 0)

    precision_test = correct_num_test / predicted_num_test
    recall_test = correct_num_test / individual_num_test

    # Confusion matrix for whole set
    y_label = np.zeros((y_pred.shape[0], y_pred.shape[1], y_pred.shape[2]))
    for i in range(y_pred.shape[0]):
        y_label[i] = np.argmax(y_pred[i], axis = 2)
    all_target_label = np.zeros(y_label.shape)
    for i in range(y_pred_test.shape[0]):
        all_target_label[i] = np.argmax(all_target[i], axis = 2)
    conf_mat = np.zeros((y_pred.shape[0], 12, 12))
    for k in range(y_pred.shape[0]):
        for i in range(12):
            for j in range(12):
                conf_mat[k][i][j] = np.where(np.logical_and(y_label[k] == i, all_target_label[k] == j))[0].shape[0]
    conf_mat = np.sum(conf_mat, axis = 0)
    conf_mat_print = conf_mat
    conf_mat = np.nan_to_num(conf_mat / np.sum(conf_mat, axis = 0))

    # Preccision & recall for test set
    predicted_num = np.sum(conf_mat_print.astype(int), axis = 1)
    correct_num = conf_mat_print.diagonal()
    individual_num = np.sum(conf_mat_print.astype(int), axis = 0)

    precision = correct_num / predicted_num
    recall = correct_num / individual_num

    np.set_printoptions(precision = 2, suppress = True)
    with open('training_results/evaluation_results.txt', 'w') as f:
        f.write('Training evaluation results of UNet 3+')
        f.write('\n')
        f.write('\n')
        f.write(f'Precision for test set: {precision_test}\n')
        f.write(f'Recall  for  test  set: {recall_test}\n')
        f.write('\n')
        f.write(f'Precision for whole set: {precision}\n')
        f.write(f'Recall  for  whole  set: {recall}\n')
        f.write('\n')
        f.write(f'Dice coefficient for test set: {np.mean(losses.dice_coef(test_target.astype(np.double), y_pred_test.astype(np.double)))}\n')
        f.write(f'Overall IoU for test set: {1 - np.mean(losses.iou_seg(test_target, y_pred_test))}\n')
        f.write('\n')
        f.write(f'Dice coefficient for whole set: {np.mean(losses.dice_coef(all_target.astype(np.double), y_pred.astype(np.double)))}\n')
        f.write(f'Overall IoU for whole set: {1 - np.mean(losses.iou_seg(all_target, y_pred))}\n')
        f.write('\n')
        f.write(f'Mean IoU for test set: {np.mean(temp_sum_test) / 12}\n')
        f.write(f'Mean IoU for whole set: {np.mean(temp_sum) / 12}\n')
        f.write('\n')
        f.write('\n')
        f.write('Test set confusion matrix:\n')
        np.savetxt(f, conf_mat_test_print, fmt = '%6d')
        f.write('\n')
        f.write('Whole set confusion matrix:\n')
        np.savetxt(f, conf_mat_print, fmt = '%6d')
        f.write('\n')
        f.close()

    # Plot confusion matrices
    plt.figure(figsize = (8, 7))
    plt.imshow(conf_mat_test)
    plt.xlabel('Predicted', fontsize = 15)
    plt.ylabel('True', fontsize = 15)
    plt.xticks(np.arange(12), ['Background', 'Artery', 'Vein', 'Artery, ICG', 'Stroma', 'Stroma, ICG', 'Umbil.', 'Suture', 'Sp.reflection', 'Blue dye', 'ICG', 'Red dye'], rotation = 75)
    plt.yticks(np.arange(12), ['Background', 'Artery', 'Vein', 'Artery, ICG', 'Stroma', 'Stroma, ICG', 'Umbil.', 'Suture', 'Sp.reflection', 'Blue dye', 'ICG', 'Red dye'])
    plt.title('Test set confusion matrix', fontsize = 18)
    plt.colorbar(shrink = 0.7)
    plt.savefig('training_results/test_confusion_matrix.png', bbox_inches = 'tight')
    plt.clf()

    plt.imshow(conf_mat)
    plt.xlabel('Predicted', fontsize = 15)
    plt.ylabel('True', fontsize = 15)
    plt.xticks(np.arange(12), ['Background', 'Artery', 'Vein', 'Artery, ICG', 'Stroma', 'Stroma, ICG', 'Umbil.', 'Suture', 'Sp.reflection', 'Blue dye', 'ICG', 'Red dye'], rotation = 75)
    plt.yticks(np.arange(12), ['Background', 'Artery', 'Vein', 'Artery, ICG', 'Stroma', 'Stroma, ICG', 'Umbil.', 'Suture', 'Sp.reflection', 'Blue dye', 'ICG', 'Red dye'])
    plt.title('Whole set confusion matrix', fontsize = 18)
    plt.colorbar(shrink = 0.7)
    plt.savefig('training_results/whole_confusion_matrix.png', bbox_inches = 'tight')
    plt.clf()



elif model_name == 'swinUnet':
    prediction_model = models.swin_unet_2d((128, 128, 38), filter_num_begin=64, n_labels=12, depth=4, stack_num_down=2, stack_num_up=2,
                            patch_size=(2, 2), num_heads=[4, 8, 8, 8], window_size=[4, 2, 2, 2], num_mlp=512,
                            output_activation='Softmax', shift_window=True, name='prediction_model')
    prediction_model.load_weights('saved_models/swin_weights.h5')

    valid_input1 = np.expand_dims(valid_input, axis = 1)
    valid_target1 = np.expand_dims(valid_target, axis = 1)
    test_input1 = np.expand_dims(test_input, axis = 1)
    test_target1 = np.expand_dims(test_target, axis = 1)
    train_input1 = np.expand_dims(train_input, axis = 1)
    train_target1 = np.expand_dims(train_target, axis = 1)

    test_input = np.load('data_arrays/test_data.npy')

    train_dataset = tf.data.Dataset.from_tensor_slices((train_input1, train_target1))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_input1, test_target1))
    val_dataset = tf.data.Dataset.from_tensor_slices((valid_input1, valid_target1))

    # Test set evaluation
    y_pred_test = prediction_model.predict(test_dataset)

    y_pred_test_tensor = tf.convert_to_tensor(y_pred_test)
    y_pred_test_tensor = tf.cast(y_pred_test_tensor, tf.float32)
    y_true_test = tf.cast(test_target, y_pred_test_tensor.dtype)
    y_pred_test_tensor = tf.squeeze(y_pred_test_tensor)
    y_true_test = tf.squeeze(y_true_test)

    temp_sum_test = 0
    ious_test = []
    for i in range(12):
        y_pred_pos_test = tf.reshape(y_pred_test_tensor[:, :, :, i], [-1])
        y_true_pos_test = tf.reshape(y_true_test[:, :, :, i], [-1])
        area_intersect = tf.reduce_sum(tf.multiply(y_true_pos_test, y_pred_pos_test))
        area_true = tf.reduce_sum(y_true_pos_test)
        area_pred = tf.reduce_sum(y_pred_pos_test)
        area_union = area_true + area_pred - area_intersect
        temp_sum_test = temp_sum_test + tf.math.divide_no_nan(area_intersect, area_union)
        ious_test.append(np.mean(tf.math.divide_no_nan(area_intersect, area_union)))

    # Full dataset evaluation
    all_input = np.append(np.append(valid_input, test_input, axis = 0), train_input, axis = 0)
    all_target = np.append(np.append(valid_target, test_target, axis = 0), train_target, axis = 0)

    all_target1 = np.expand_dims(all_target, axis = 1)
    all_input1 = np.expand_dims(all_input, axis = 1)
    all_dataset = tf.data.Dataset.from_tensor_slices((all_input1, all_target1))

    y_pred = prediction_model.predict(all_dataset)

    y_pred_tensor = tf.convert_to_tensor(y_pred)
    y_pred_tensor = tf.cast(y_pred_tensor, tf.float32)
    y_true = tf.cast(all_target, y_pred_tensor.dtype)
    y_pred_tensor = tf.squeeze(y_pred_tensor)
    y_true = tf.squeeze(y_true)

    temp_sum = 0
    ious = []
    for i in range(12):
        y_true_pos = tf.reshape(y_true[:, :, :, i], [-1])
        y_pred_pos = tf.reshape(y_pred_tensor[:, :, :, i], [-1])
        area_intersect = tf.reduce_sum(tf.multiply(y_true_pos, y_pred_pos))
        area_true = tf.reduce_sum(y_true_pos)
        area_pred = tf.reduce_sum(y_pred_pos)
        area_union = area_true + area_pred - area_intersect
        temp_sum = temp_sum + tf.math.divide_no_nan(area_intersect, area_union)
        ious.append(np.mean(tf.math.divide_no_nan(area_intersect, area_union)))

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
    colors = colors / 255.0

    # IoU plot by class
    plt.figure(figsize = (20, 10))
    plt.rcParams['axes.facecolor'] = 'pink'
    plt.bar(np.arange(12), ious_test, color = colors)
    plt.yticks(fontsize = 17)
    plt.xticks(np.arange(12), ['Background', 'Artery', 'Vein', 'Artery, ICG', 'Stroma', 'Stroma, ICG', 'Umbil. cord', 'Suture', 'Sp.reflection', 'Blue dye', 'ICG', 'Red dye'], rotation = 75)
    plt.title('Test set IoU by class', fontsize = 20)
    plt.ylim(0, 1)
    plt.savefig('training_results/IoU_test_barplot.png', bbox_inches = 'tight')
    plt.clf()

    plt.figure(figsize = (20, 10))
    plt.rcParams['axes.facecolor'] = 'pink'
    plt.bar(np.arange(12), ious, color = colors)
    plt.yticks(fontsize = 17)
    plt.xticks(np.arange(12), ['Background', 'Artery', 'Vein', 'Artery, ICG', 'Stroma', 'Stroma, ICG', 'Umbil.', 'Suture', 'Sp.reflection', 'Blue dye', 'ICG', 'Red dye'], rotation = 75)
    plt.title('Whole set IoU by class', fontsize = 20)
    plt.ylim(0, 1)
    plt.savefig('training_results/IoU_whole_barplot.png', bbox_inches = 'tight')
    plt.clf()

    # Confusion matrix for test set
    y_test_label = np.zeros((y_pred_test.shape[0], y_pred_test.shape[1], y_pred_test.shape[2]))
    for i in range(y_pred_test.shape[0]):
        y_test_label[i] = np.argmax(y_pred_test[i], axis = 2)
    test_target_label = np.zeros(y_test_label.shape)
    for i in range(y_pred_test.shape[0]):
        test_target_label[i] = np.argmax(test_target[i], axis = 2)
    conf_mat_test = np.zeros((y_pred_test.shape[0], 12, 12))
    for k in range(y_pred_test.shape[0]):
        for i in range(12):
            for j in range(12):
                conf_mat_test[k][i][j] = np.where(np.logical_and(y_test_label[k] == i, test_target_label[k] == j))[0].shape[0]
    conf_mat_test = np.sum(conf_mat_test, axis = 0)
    conf_mat_test_print = conf_mat_test
    conf_mat_test = np.nan_to_num(conf_mat_test / np.sum(conf_mat_test, axis = 0))

    # Preccision & recall for test set
    predicted_num_test = np.sum(conf_mat_test_print.astype(int), axis = 1)
    correct_num_test = conf_mat_test_print.diagonal()
    individual_num_test = np.sum(conf_mat_test_print.astype(int), axis = 0)

    precision_test = correct_num_test / predicted_num_test
    recall_test = correct_num_test / individual_num_test

    # Confusion matrix for whole set
    y_label = np.zeros((y_pred.shape[0], y_pred.shape[1], y_pred.shape[2]))
    for i in range(y_pred.shape[0]):
        y_label[i] = np.argmax(y_pred[i], axis = 2)
    all_target_label = np.zeros(y_label.shape)
    for i in range(y_pred_test.shape[0]):
        all_target_label[i] = np.argmax(all_target[i], axis = 2)
    conf_mat = np.zeros((y_pred.shape[0], 12, 12))
    for k in range(y_pred.shape[0]):
        for i in range(12):
            for j in range(12):
                conf_mat[k][i][j] = np.where(np.logical_and(y_label[k] == i, all_target_label[k] == j))[0].shape[0]
    conf_mat = np.sum(conf_mat, axis = 0)
    conf_mat_print = conf_mat
    conf_mat = np.nan_to_num(conf_mat / np.sum(conf_mat, axis = 0))

    # Preccision & recall for test set
    predicted_num = np.sum(conf_mat_print.astype(int), axis = 1)
    correct_num = conf_mat_print.diagonal()
    individual_num = np.sum(conf_mat_print.astype(int), axis = 0)

    precision = correct_num / predicted_num
    recall = correct_num / individual_num

    np.set_printoptions(precision = 2, suppress = True)
    with open('training_results/evaluation_results.txt', 'w') as f:
        f.write('Training evaluation results of Swin-Unet')
        f.write('\n')
        f.write('\n')
        f.write(f'Precision for test set: {precision_test}\n')
        f.write(f'Recall for test set: {recall_test}\n')
        f.write('\n')
        f.write(f'Precision for whole set: {precision}\n')
        f.write(f'Recall  for  whole  set: {recall}\n')
        f.write('\n')
        f.write(f'Dice coefficient for test set: {np.mean(losses.dice_coef(test_target.astype(np.double), y_pred_test.astype(np.double)))}\n')
        f.write(f'Overall IoU for test set: {1 - np.mean(losses.iou_seg(test_target, y_pred_test))}\n')
        f.write('\n')
        f.write(f'Dice coefficient for whole set: {np.mean(losses.dice_coef(all_target.astype(np.double), y_pred.astype(np.double)))}\n')
        f.write(f'Overall IoU for whole set: {1 - np.mean(losses.iou_seg(all_target, y_pred))}\n')
        f.write('\n')
        f.write(f'Mean IoU for test set: {np.mean(temp_sum_test) / 12}\n')
        f.write(f'Mean IoU for whole set: {np.mean(temp_sum) / 12}\n')
        f.write('\n')
        f.write('\n')
        f.write('Test set confusion matrix:\n')
        np.savetxt(f, conf_mat_test_print, fmt = '%6d')
        f.write('\n')
        f.write('Whole set confusion matrix:\n')
        np.savetxt(f, conf_mat_print, fmt = '%6d')
        f.write('\n')
        f.close()

    # Plot confusion matrices
    plt.figure(figsize = (8, 7))
    plt.imshow(conf_mat_test)
    plt.xlabel('Predicted', fontsize = 15)
    plt.ylabel('True', fontsize = 15)
    plt.xticks(np.arange(12), ['Background', 'Artery', 'Vein', 'Artery, ICG', 'Stroma', 'Stroma, ICG', 'Umbil.', 'Suture', 'Sp.reflection', 'Blue dye', 'ICG', 'Red dye'], rotation = 75)
    plt.yticks(np.arange(12), ['Background', 'Artery', 'Vein', 'Artery, ICG', 'Stroma', 'Stroma, ICG', 'Umbil.', 'Suture', 'Sp.reflection', 'Blue dye', 'ICG', 'Red dye'])
    plt.title('Test set confusion matrix', fontsize = 18)
    plt.colorbar(shrink = 0.7)
    plt.savefig('training_results/test_confusion_matrix.png', bbox_inches = 'tight')
    plt.clf()

    plt.imshow(conf_mat)
    plt.xlabel('Predicted', fontsize = 15)
    plt.ylabel('True', fontsize = 15)
    plt.xticks(np.arange(12), ['Background', 'Artery', 'Vein', 'Artery, ICG', 'Stroma', 'Stroma, ICG', 'Umbil.', 'Suture', 'Sp.reflection', 'Blue dye', 'ICG', 'Red dye'], rotation = 75)
    plt.yticks(np.arange(12), ['Background', 'Artery', 'Vein', 'Artery, ICG', 'Stroma', 'Stroma, ICG', 'Umbil.', 'Suture', 'Sp.reflection', 'Blue dye', 'ICG', 'Red dye'])
    plt.title('Whole set confusion matrix', fontsize = 18)
    plt.colorbar(shrink = 0.7)
    plt.savefig('training_results/whole_confusion_matrix.png', bbox_inches = 'tight')
    plt.clf()
