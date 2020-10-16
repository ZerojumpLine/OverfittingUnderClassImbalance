# Copyright (c) 2016, Konstantinos Kamnitsas
# All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the BSD license. See the accompanying LICENSE file
# or read the terms at https://opensource.org/licenses/BSD-3-Clause.

from __future__ import absolute_import, print_function, division
import numpy as np
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

def focaloneside(network_output, y_gtmixp0, y_gtmixp1, gama, weightPerClass, mixupbiasmargin, marginm, mixuplambda, eps=1e-6 ):
    # network_output : tensor5 [batchSize, classes, r, c, z]
    # y: T.itensor4('y'). Dimensions [batchSize, r, c, z]
    # weightPerClass is a vector with 1 element per class.

    '''
    Watchout, network_output does not have non linear, we do softmax here.
    It now only supports binary class segmentation (brats/ ATLAS)
    However, it should not be difficult to extend.
    '''

    '''
    tips, if you want to extend the code to your application, you can just define your /boldsymbol{r}
    '''

    ################################################ mixup part ################################################
    '''
    needed hyperparameters: mixupbiasmargin, mixuplambda
    '''

    if mixupbiasmargin > 0: # asymmetric mixup

        r = [0, 1]
        rall = [i for i, e in enumerate(r) if e == 1]
        y_comb0 = y_gtmixp0
        y_comb1 = y_gtmixp1
        # if the other one is taken as one of the rare classes, the combination should change
        for rindex in rall:
            rindextf = tf.constant(rindex)
            y_comb0 = tf.where(tf.equal(y_gtmixp1, rindextf), x = y_gtmixp1, y = y_comb0)
            y_comb1 = tf.where(tf.equal(y_gtmixp0, rindextf), x = y_gtmixp0, y = y_comb1)

        lambdathreshold = tf.constant(1-mixupbiasmargin, dtype=tf.float32)

        y_gtmix0 = tf.cond(tf.less(mixuplambda, lambdathreshold), lambda: y_comb0, lambda: y_gtmixp0)
        y_gtmix1 = tf.cond(tf.less(1 - mixuplambda, lambdathreshold), lambda: y_comb1, lambda: y_gtmixp1)

        ## combine the two loss
        y_one_hot_mix0 = tf.one_hot(indices=y_gtmix0, depth=tf.shape(network_output)[1], axis=1, dtype="float32") * tf.cast(abs(mixuplambda), tf.float32)
        y_one_hot_mix1 = tf.one_hot(indices=y_gtmix1, depth=tf.shape(network_output)[1], axis=1, dtype="float32") * tf.cast((1 - abs(mixuplambda)), tf.float32)
        y_one_hot_mixpre = y_one_hot_mix0 + y_one_hot_mix1

        # if the sample is generated by mixup, I do not want middle results, such as "0.3 tumor"
        y_one_hot_mix = tf.floor(y_one_hot_mixpre)
    else: # traditional mixup, just mix the one-hot label
        y_gtmix0 = y_gtmixp0
        y_gtmix1 = y_gtmixp1
        y_one_hot_mix0 = tf.one_hot(indices=y_gtmix0, depth=tf.shape(network_output)[1], axis=1, dtype="float32") * tf.cast(abs(mixuplambda), tf.float32)
        y_one_hot_mix1 = tf.one_hot(indices=y_gtmix1, depth=tf.shape(network_output)[1], axis=1, dtype="float32") * tf.cast((1 - abs(mixuplambda)), tf.float32)
        y_one_hot_mix = y_one_hot_mix0 + y_one_hot_mix1 # [batchSize, classes, r, c, z]

    ########################################################################################################

    # this is taken from the network code, to get the input to softmax
    inputToSoftmaxReshaped = tf.transpose(network_output, perm=[0, 2, 3, 4, 1])
    inputToSoftmaxFlattened = tf.reshape(inputToSoftmaxReshaped, shape=[-1])
    numberOfVoxelsDenselyClassified = network_output.shape[2] * \
                                      network_output.shape[3] * \
                                      network_output.shape[4]
    firstDimOfInputToSoftmax2d = network_output.shape[
                                     0] * numberOfVoxelsDenselyClassified  # batchSize*r*c*z.
    inputToSoftmax2d = tf.reshape(inputToSoftmaxFlattened, shape=[firstDimOfInputToSoftmax2d,
                                                                  network_output.shape[
                                                                      1]])  # N * cls

    ########################################### large margin part #################################################
    '''
    needed hyperparamter: marginm
    '''

    if marginm < 0: # do it in both classes
        r = [1, 1]
    else: # only for cls 1
        r = [0, 1]
    r = tf.constant(r, shape=[1, len(r)], dtype=tf.float32)
    y_one_hot_mixReshaped = tf.transpose(y_one_hot_mix, perm=[0, 2, 3, 4, 1])
    y_one_hot_mixFlattened = tf.reshape(y_one_hot_mixReshaped, shape=[firstDimOfInputToSoftmax2d,
                                                              network_output.shape[1]]) # N * cls
    y_one_hot_mixFlattened_M = y_one_hot_mixFlattened * abs(marginm)
    # multiply by the r
    # extent r from 1 * cls to N * cls
    rRepeat = tf.tile(r, [firstDimOfInputToSoftmax2d,1])
    # this is to get \hat{q}
    inputToSoftmax2d_M = inputToSoftmax2d - y_one_hot_mixFlattened_M * rRepeat

    ########################################################################################################

    # do the softmax
    p_y_given_x_2d = tf.nn.softmax(inputToSoftmax2d_M, axis=-1)
    p_y_given_x_classMinor = tf.reshape(p_y_given_x_2d, shape=[network_output.shape[0],
                                                               network_output.shape[2],
                                                               network_output.shape[3],
                                                               network_output.shape[4],
                                                               network_output.shape[
                                                                   1]])  # Result: batchSize, R,C,Z, Classes.
    p_y_given_x_train = tf.transpose(p_y_given_x_classMinor,
                                     perm=[0, 4, 1, 2, 3])  # Result: batchSize, Class, R, C, Z
    log_p_y_given_x_train = tf.log(p_y_given_x_train + eps)

    ####################################  focal loss part  ########################################################
    '''
    needed hyperparameter: gamma (it is a typo, where I call it gama...)
    '''

    if gama < 0: # do it only for the background cls
        r = [0, 1]
    else: # normal focal loss
        r = [0, 0]
    r = tf.constant(r, shape=[1, len(r), 1, 1, 1], dtype=tf.float32)
    # fill the shape from 1, Class,1, 1, 1 to batchSize, Class, R, C, Z
    rRepeat = tf.tile(r, [p_y_given_x_train.shape[0], 1, p_y_given_x_train.shape[2], p_y_given_x_train.shape[3],
                 p_y_given_x_train.shape[4]])

    focal_conduct_active = (1 - p_y_given_x_train + eps) ** abs(gama)
    focal_conduct_inactive = tf.ones([focal_conduct_active.shape[0], focal_conduct_active.shape[1], focal_conduct_active.shape[2], focal_conduct_active.shape[3],
                 focal_conduct_active.shape[4]])

    focal_conduct = focal_conduct_active * (1-rRepeat) + focal_conduct_inactive * rRepeat
    m_log_p_y_given_x_train = focal_conduct * log_p_y_given_x_train
    weightPerClass5D = tf.reshape(weightPerClass, shape=[1, tf.shape(p_y_given_x_train)[1], 1, 1, 1])
    weighted_log_p_y_given_x_train = m_log_p_y_given_x_train * weightPerClass5D

    ########################################################################################################

    num_samples = tf.cast(tf.reduce_prod(tf.shape(y_gtmix0)), "float32")

    x_entr_mix = - (1. / num_samples) * tf.reduce_sum(weighted_log_p_y_given_x_train * y_one_hot_mix)

    return x_entr_mix