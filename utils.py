# This module defines the utils for processing.

import numpy as np
import tensorflow as tf
from quadratic_weighted_kappa import *
import time
import random
# Preprocessing (for both training and validation):
# (1) Decode the image from jpg format
# (2) Resize the image so its smaller side is 256 pixels long
def parse_function(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)          # (1)
    image = tf.cast(image_decoded, tf.float32)

    # This the old settings
    # smallest_side = 256.0

    # set the image size as desired.
    smallest_side = 224.0
    height, width = tf.shape(image)[0], tf.shape(image)[1]
    height = tf.to_float(height)
    width = tf.to_float(width)

    scale = tf.cond(tf.greater(height, width),
                    lambda: smallest_side / width,
                    lambda: smallest_side / height)
    new_height = tf.to_int32(height * scale)
    new_width = tf.to_int32(width * scale)

    resized_image = tf.image.resize_images(image, [new_height, new_width])  # (2)

    return resized_image, label

# Preprocessing (for training)
# (3) Take a random 224x224 crop to the scaled image
# (4) Horizontally flip the image with probability 1/2
# (5) Substract the per color mean `VGG_MEAN`
# Note: we don't normalize the data here, as VGG was trained without normalization
def training_preprocess(image, label, VGG_MEAN=[123.68, 116.78, 103.94]):
    augment = False

    if augment:
        crop_image = tf.random_crop(image, [224, 224, 3])                       # (3)
        flip_image = tf.image.random_flip_left_right(crop_image)                # (4)
    else:
        flip_image = tf.random_crop(image, [224, 224, 3]) 

    means = tf.reshape(tf.constant(VGG_MEAN), [1, 1, 3])
    centered_image = flip_image - means                                     # (5)

    return centered_image, label





# Preprocessing (for validation)
# (3) Take a central 224x224 crop to the scaled image
# (4) Substract the per color mean `VGG_MEAN`
# Note: we don't normalize the data here, as VGG was trained without normalization
def val_preprocess(image, label, VGG_MEAN=[123.68, 116.78, 103.94]):

    # This is old setting
    # crop_image = tf.image.resize_image_with_crop_or_pad(image, 224, 224)    # (3)
    
    crop_image = tf.random_crop(image, [224, 224, 3])



    means = tf.reshape(tf.constant(VGG_MEAN), [1, 1, 3])
    centered_image = crop_image - means                                     # (4)

    return centered_image, label





def check_accuracy(sess, correct_prediction, labels, prediction, is_training, dataset_init_op):
    """
    Check the accuracy of the model on either train or val (depending on dataset_init_op).
    """
    # Initialize the correct dataset
    sess.run(dataset_init_op)
    num_correct, num_samples = 0, 0

    gt_labels = []
    predicted_labels = []

    while True:
        try:
            correct_pred, label_value, prediction_value = sess.run([correct_prediction, labels, prediction], {is_training: False})
            gt_labels.append(label_value)
            predicted_labels.append(prediction_value)
            num_correct += correct_pred.sum()
            num_samples += correct_pred.shape[0]
        except tf.errors.OutOfRangeError:
            break

    print("TOTAL number of samples is {}".format(num_samples))
    # Return the fraction of datapoints that were correctly classified
    acc = float(num_correct) / num_samples

    gt_labels = np.concatenate(gt_labels, axis=0).astype(int)
    predicted_labels = np.concatenate(predicted_labels, axis=0).astype(int)
    acc_weighted = quadratic_weighted_kappa(gt_labels, predicted_labels)

    # check the accuracy for each for classes:
    class_num = 5

    total_prediction = np.equal(gt_labels, predicted_labels)


    # The prediction information for each class
    # pred_each_class: Nx3 array
    #           the first col is the number of correct prediciton, 
    #           the second col is total number of samples
    #           the third col is the accuracy
    acc_for_each_class = np.zeros((class_num, 3))

    #time.sleep(100)
    for i in range(class_num):
        index_i = np.where(gt_labels == i)          # get all the index of class i
        pred_class_i = total_prediction[index_i]
        acc_for_each_class[i,0] = np.mean(pred_class_i.astype(float))
        acc_for_each_class[i,1] = np.sum(pred_class_i)       # The number of correct prediction for class i
        acc_for_each_class[i,2] = index_i[0].shape[0]        # The number of total samples of class i
        

    return acc, acc_weighted, acc_for_each_class






def duplicateSamples(sampleList, targetNum=700):

    sampleNum = len(sampleList)

    scale, remain = targetNum // sampleNum, targetNum % sampleNum

    # if sample number is greater than target number, return # targetNum
    if scale  == 0:
        return sampleList[:targetNum]

    if remain == 0:
        return sampleList * scale
    else:
        return sampleList * scale + sampleList[:remain]

def generateList(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    # init dictionary for saving all the images.
    cls_dict = {}
    for line in lines:
        splt = line.split(' ')
        path, label = splt[0], int(splt[1])

        if label not in cls_dict.keys():
            cls_dict[label] = [path]
        else:
            cls_dict[label].append(path)


    return cls_dict     # save all the data by class

def generateData(cls_dict, step=None, dpFlag=False, targetNum=700):
    # shuffle the list and pick 1/20 samples
    pathset = []
    labelset = []
    
    random.seed(2222)

    for key in cls_dict.keys():
        sample_num = len(cls_dict[key])
        #print("The {}-th class has {:5d} samples before downsample.".format(key, sample_num))
        #print("shuffle the list and pick 1/20 samples")
        random.shuffle(cls_dict[key])
        if step is not None:
            cls_dict[key] = cls_dict[key][::step]

        if dpFlag:
            cls_dict[key] = duplicateSamples(sampleList=cls_dict[key], targetNum=targetNum)


        sample_num = len(cls_dict[key])
        #print("The {:5}-th class has {:5d} samples after downsample.".format(key, sample_num))
        #print("First 3 samples\n {}".format(cls_dict[key][:3]))

        # get the downsampled list 
        pathset += cls_dict[key]
        labelset += sample_num * [key]


    pathAndLabel = zip(pathset, labelset)
    random.shuffle(pathAndLabel)


    pathset = []
    labelset = []

    for tmp in pathAndLabel:
        pathset.append(tmp[0])
        labelset.append(tmp[1])


    return pathset, labelset



def list_images(directory):
    """
    Get all the images and labels in directory/label/*.jpg
    """
    labels = os.listdir(directory)
    # Sort the labels so that training and validation get them in the same order
    labels.sort()

    files_and_labels = []
    for label in labels:
        for f in os.listdir(os.path.join(directory, label)):
            files_and_labels.append((os.path.join(directory, label, f), label))

    filenames, labels = zip(*files_and_labels)
    filenames = list(filenames)
    labels = list(labels)
    unique_labels = list(set(labels))

    label_to_int = {}
    for i, label in enumerate(unique_labels):
        label_to_int[label] = i

    labels = [label_to_int[l] for l in labels]

    return filenames, labels
