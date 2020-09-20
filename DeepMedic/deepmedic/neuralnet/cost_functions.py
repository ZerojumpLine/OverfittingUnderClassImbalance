# Copyright (c) 2016, Konstantinos Kamnitsas
# All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the BSD license. See the accompanying LICENSE file
# or read the terms at https://opensource.org/licenses/BSD-3-Clause.

from __future__ import absolute_import, print_function, division

import tensorflow as tf


def x_entr(p_y_given_x_train, y_gt, weightPerClass, eps=1e-6):
    # p_y_given_x_train : tensor5 [batchSize, classes, r, c, z]
    # y: T.itensor4('y'). Dimensions [batchSize, r, c, z]
    # weightPerClass is a vector with 1 element per class.

    # Weighting the cost of the different classes in the cost-function, in order to counter class imbalance.
    log_p_y_given_x_train = tf.log(p_y_given_x_train + eps)

    weightPerClass5D = tf.reshape(weightPerClass, shape=[1, tf.shape(p_y_given_x_train)[1], 1, 1, 1])
    weighted_log_p_y_given_x_train = log_p_y_given_x_train * weightPerClass5D

    y_one_hot = tf.one_hot(indices=y_gt, depth=tf.shape(p_y_given_x_train)[1], axis=1, dtype="float32")

    num_samples = tf.cast(tf.reduce_prod(tf.shape(y_gt)), "float32")

    return - (1. / num_samples) * tf.reduce_sum(weighted_log_p_y_given_x_train * y_one_hot)

def iou(p_y_given_x_train, y_gt, eps=1e-5):
    # Intersection-Over-Union / Jaccard: https://en.wikipedia.org/wiki/Jaccard_index
    # Analysed in: Nowozin S, Optimal Decisions from Probabilistic Models: the Intersection-over-Union Case, CVPR 2014
    # First computes IOU per class. Finally averages over the class-ious.
    # p_y_given_x_train : tensor5 [batchSize, classes, r, c, z]
    # y: T.itensor4('y'). Dimensions [batchSize, r, c, z]
    y_one_hot = tf.one_hot( indices=y_gt, depth=tf.shape(p_y_given_x_train)[1], axis=1, dtype="float32" )
    ones_at_real_negs = tf.cast( tf.less(y_one_hot, 0.0001), dtype="float32") # tf.equal(y_one_hot,0), but less may be more stable with floats.
    numer = tf.reduce_sum(p_y_given_x_train * y_one_hot, axis=(0,2,3,4)) # 2 * TP
    denom = tf.reduce_sum(p_y_given_x_train * ones_at_real_negs, axis=(0,2,3,4)) + tf.reduce_sum(y_one_hot, axis=(0,2,3,4)) # Pred + RP
    iou = (numer + eps) / (denom + eps) # eps in both num/den => dsc=1 when class missing.
    av_class_iou = tf.reduce_mean(iou) # Along the class-axis. Mean DSC of classes. 
    cost = 1. - av_class_iou
    return cost


def dsc(p_y_given_x_train, y_gt, eps=1e-5):
    # Similar to Intersection-Over-Union / Jaccard above.
    # Dice coefficient: https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
    y_one_hot = tf.one_hot( indices=y_gt, depth=tf.shape(p_y_given_x_train)[1], axis=1, dtype="float32" )
    numer = 2. * tf.reduce_sum(p_y_given_x_train * y_one_hot, axis=(0,2,3,4)) # 2 * TP
    denom = tf.reduce_sum(p_y_given_x_train, axis=(0,2,3,4)) + tf.reduce_sum(y_one_hot, axis=(0,2,3,4)) # Pred + RP
    dsc = (numer + eps) / (denom + eps) # eps in both num/den => dsc=1 when class missing.
    av_class_dsc = tf.reduce_mean(dsc) # Along the class-axis. Mean DSC of classes. 
    cost = 1. - av_class_dsc
    return cost

def focaloneside(p_y_given_x_train_network_output, y_gtmixp0, y_gtmixp1, gama, weightPerClass, mixupbiasmargin, marginm, mixuplambda, eps=1e-6 ):
    # p_y_given_x_train : tensor5 [batchSize, classes, r, c, z]
    # y: T.itensor4('y'). Dimensions [batchSize, r, c, z]
    # weightPerClass is a vector with 1 element per class.

    '''
    Watchout, p_y_given_x_train_network_output does not have non linear, we do softmax here.
    It now only supports binary class segmentation (brats/ ATLAS)
    However, it should not be difficult to extend, but needs good designs...
    '''

    ########################################### margin part #################################################
    '''
    needed hyperparamter: marginm
    '''

    inputToSoftmaxReshaped = tf.transpose(p_y_given_x_train_network_output, perm=[0, 2, 3, 4, 1])
    inputToSoftmaxFlattened = tf.reshape(inputToSoftmaxReshaped, shape=[-1])
    numberOfVoxelsDenselyClassified = p_y_given_x_train_network_output.shape[2] * p_y_given_x_train_network_output.shape[3] * p_y_given_x_train_network_output.shape[4]
    firstDimOfInputToSoftmax2d = p_y_given_x_train_network_output.shape[0] * numberOfVoxelsDenselyClassified  # b
    inputToSoftmax2d = tf.reshape(inputToSoftmaxFlattened, shape=[firstDimOfInputToSoftmax2d, p_y_given_x_train_network_output.shape[1]])  # Res

    inputToSoftmax2dt = tf.stack([inputToSoftmax2d[:, 0], inputToSoftmax2d[:, 1] - abs(marginm)])  # it is only for tumour
    inputToSoftmax2dt = tf.transpose(inputToSoftmax2dt, [1, 0])

    if marginm < 0:  # do it in both sides, it is for backgr
        inputToSoftmax2db = tf.stack([inputToSoftmax2d[:, 0] - abs(marginm), inputToSoftmax2d[:, 1]])  # it is only for backgr
        inputToSoftmax2db = tf.transpose(inputToSoftmax2db, [1, 0])
    else:
        inputToSoftmax2db = inputToSoftmax2d

    y_gt_flattened_tumour = tf.cast(tf.reshape(y_gtmixp0, shape=[-1]), tf.float32)  # 1 is for tumour
    y_gt_flattened_backgr = 1 - y_gt_flattened_tumour

    inputToSoftmax2dpost = tf.stack([inputToSoftmax2db[:, 0] * y_gt_flattened_backgr + inputToSoftmax2dt[:, 0] * y_gt_flattened_tumour,
                                     inputToSoftmax2db[:, 1] * y_gt_flattened_backgr + inputToSoftmax2dt[:, 1] * y_gt_flattened_tumour])
    inputToSoftmax2dpost = tf.transpose(inputToSoftmax2dpost, [1, 0])

    p_y_given_x_2d = tf.nn.softmax(inputToSoftmax2dpost, axis=-1)
    p_y_given_x_classMinor = tf.reshape(p_y_given_x_2d, shape=[p_y_given_x_train_network_output.shape[0], p_y_given_x_train_network_output.shape[2], p_y_given_x_train_network_output.shape[3],
                                                               p_y_given_x_train_network_output.shape[4], p_y_given_x_train_network_output.shape[1]])  # Result: batchSize, R,C,Z, Classes.
    p_y_given_x_train = tf.transpose(p_y_given_x_classMinor, perm=[0, 4, 1, 2, 3])  # Result: batchSize, Class, R, C, Z

    ########################################################################################################

    log_p_y_given_x_train = tf.log(p_y_given_x_train + eps)

    ################################################ mixup part ################################################
    '''
    needed hyperparameter: mixupbiasmargin, mixuplambda
    '''
    y_gtmix = y_gtmixp0 + y_gtmixp1
    y_gtmix_bool = tf.cast(y_gtmix, tf.bool)

    if mixupbiasmargin > 0: # asymmetric mixup
        lambdathreshold = tf.constant(1-mixupbiasmargin, dtype=tf.float32)

        y_gtmix0 = tf.cond(tf.less(mixuplambda, lambdathreshold), lambda: tf.cast(y_gtmix_bool, tf.int32), lambda: y_gtmixp0)
        y_gtmix1 = tf.cond(tf.less(1 - mixuplambda, lambdathreshold), lambda: tf.cast(y_gtmix_bool, tf.int32), lambda: y_gtmixp1)

        ## combine the two loss
        y_one_hot_mix0 = tf.one_hot(indices=y_gtmix0, depth=tf.shape(p_y_given_x_train)[1], axis=1, dtype="float32") * tf.cast(abs(mixuplambda), tf.float32)
        y_one_hot_mix1 = tf.one_hot(indices=y_gtmix1, depth=tf.shape(p_y_given_x_train)[1], axis=1, dtype="float32") * tf.cast((1 - abs(mixuplambda)), tf.float32)
        y_one_hot_mixpre = y_one_hot_mix0 + y_one_hot_mix1

        # if the sample is generated by mixup, I do not want middle results, such as "0.3 tumor"
        y_one_hot_mix = tf.floor(y_one_hot_mixpre)
    else: # traditional mixup, just mix the one-hot label
        y_gtmix0 = y_gtmixp0
        y_gtmix1 = y_gtmixp1
        y_one_hot_mix0 = tf.one_hot(indices=y_gtmix0, depth=tf.shape(p_y_given_x_train)[1], axis=1, dtype="float32") * tf.cast(abs(mixuplambda), tf.float32)
        y_one_hot_mix1 = tf.one_hot(indices=y_gtmix1, depth=tf.shape(p_y_given_x_train)[1], axis=1, dtype="float32") * tf.cast((1 - abs(mixuplambda)), tf.float32)
        y_one_hot_mix = y_one_hot_mix0 + y_one_hot_mix1

    ########################################################################################################

    ####################################  focal loss part  ########################################################
    '''
    needed hyperparameter: gamma (it is a typo, where I call it gama...)
    '''

    if gama < 0: # do it only for the background cls
        focal_conduct_active = (1 - p_y_given_x_train) ** abs(gama)
        a = tf.ones([focal_conduct_active.shape[0], 1, focal_conduct_active.shape[2], focal_conduct_active.shape[3], focal_conduct_active.shape[4]])
        focal_conduct = tf.concat([focal_conduct_active[:, 0:1, :, :, :], a], 1)
    else: # normal focal loss
        focal_conduct = (1 - p_y_given_x_train) ** abs(gama)
    m_log_p_y_given_x_train = focal_conduct * log_p_y_given_x_train
    weightPerClass5D = tf.reshape(weightPerClass, shape=[1, tf.shape(p_y_given_x_train)[1], 1, 1, 1])
    weighted_log_p_y_given_x_train = m_log_p_y_given_x_train * weightPerClass5D

    ########################################################################################################

    num_samples = tf.cast(tf.reduce_prod(tf.shape(y_gtmix0)), "float32")

    x_entr_mix = - (1. / num_samples) * tf.reduce_sum(weighted_log_p_y_given_x_train * y_one_hot_mix)

    return x_entr_mix