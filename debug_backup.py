"""
Example TensorFlow script for finetuning a VGG model on your own data.
Uses tf.contrib.data module which is in release v1.2
Based on PyTorch example from Justin Johnson
(https://gist.github.com/jcjohnson/6e41e8512c17eae5da50aebef3378a4c)

Required packages: tensorflow (v1.2)
Download the weights trained on ImageNet for VGG:
```
wget http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz
tar -xvf vgg_16_2016_08_28.tar.gz
rm vgg_16_2016_08_28.tar.gz
```
For this example we will use a tiny dataset of images from the COCO dataset.
We have chosen eight types of animals (bear, bird, cat, dog, giraffe, horse,
sheep, and zebra); for each of these categories we have selected 100 training
images and 25 validation images from the COCO dataset. You can download and
unpack the data (176 MB) by running:
```
wget cs231n.stanford.edu/coco-animals.zip
unzip coco-animals.zip
rm coco-animals.zip
```
The training data is stored on disk; each category has its own folder on disk
and the images for that category are stored as .jpg files in the category folder.
In other words, the directory structure looks something like this:
coco-animals/
  train/
    bear/
      COCO_train2014_000000005785.jpg
      COCO_train2014_000000015870.jpg
      [...]
    bird/
    cat/
    dog/
    giraffe/
    horse/
    sheep/
    zebra/
  val/
    bear/
    bird/
    cat/
    dog/
    giraffe/
    horse/
    sheep/
    zebra/
"""

import argparse
import os
import sys

import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets
import time
import random
import numpy as np


from utils import *

def _parse_function_backup(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)          # (1)
    image = tf.cast(image_decoded, tf.float32)

    smallest_side = 256.0
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

parser = argparse.ArgumentParser()
parser.add_argument('--train_dir', default='coco-animals/train')
parser.add_argument('--val_dir', default='coco-animals/val')
parser.add_argument('--model_path', default='./weights/vgg_16.ckpt', type=str)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--num_epochs1', default=20, type=int)
parser.add_argument('--num_epochs2', default=10, type=int)
parser.add_argument('--learning_rate1', default=1e-3, type=float)
parser.add_argument('--learning_rate2', default=1e-5, type=float)
parser.add_argument('--dropout_keep_prob', default=0.5, type=float)
parser.add_argument('--weight_decay', default=5e-4, type=float)
parser.add_argument('--restore_ckpt', default=0, type=int)      # 1 for True
parser.add_argument('--evaluation', default=0, type=int)        # 1 for True
parser.add_argument('--weightFile', default='./models/my-model', type=str)
parser.add_argument('--ckpt_dir', default='./models/alchemic', type=str)
parser.add_argument('--dn_train', default=20, type=int)
parser.add_argument('--dn_test', default=5, type=int)
parser.add_argument('--class_num', default=5, type=int)






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


def main(args):
    # Get the list of filenames and corresponding list of labels for training et validation
    # train_filenames, train_labels = list_images(args.train_dir)
    # val_filenames, val_labels = list_images(args.val_dir)


    def duplicateSamples(sampleList, targetNum=700):

        sampleNum = len(sampleList)

        scale, remain = targetNum // sampleNum, targetNum % sampleNum

        if remain == 0:
            return sampleList * scale
        else:
            return sampleList * scale + sampleList[:remain]

    def generateList(filename, step=None, dpFlag=False):
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

        # shuffle the list and pick 1/20 samples
        pathset = []
        labelset = []
        random.seed(2222)
        pos_weights = np.zeros((len(cls_dict.keys())))


        # duplicate samples 


        for key in cls_dict.keys():
            sample_num = len(cls_dict[key])
            #print("The {}-th class has {:5d} samples before downsample.".format(key, sample_num))
            #print("shuffle the list and pick 1/20 samples")
            random.shuffle(cls_dict[key])
            if step is not None:
                cls_dict[key] = cls_dict[key][::step]

            if dpFlag:
                cls_dict[key] = duplicateSamples(sampleList=cls_dict[key], targetNum=700)


            sample_num = len(cls_dict[key])
            #print("The {:5}-th class has {:5d} samples after downsample.".format(key, sample_num))
            #print("First 3 samples\n {}".format(cls_dict[key][:3]))

            # get the downsampled list 
            pathset += cls_dict[key]
            labelset += sample_num * [key]
            pos_weights[key] = sample_num

        pos_weights = 1 / pos_weights * len(pathset)

        pathAndLabel = zip(pathset, labelset)
        random.shuffle(pathAndLabel)


        pathset = []
        labelset = []

        for tmp in pathAndLabel:
            pathset.append(tmp[0])
            labelset.append(tmp[1])

        print(pos_weights)


        return pathset, labelset, pos_weights



    print("ONLY downsample the training dataset.")
    train_filenames, train_labels, pos_weights= generateList('train.txt', step=args.dn_train, dpFlag=True)
    if args.dn_test == 0:
        val_filenames, val_labels, _ = generateList('test.txt', step=None, dpFlag=False)
    else:
        val_filenames, val_labels, _ = generateList('test.txt', step=args.dn_test, dpFlag=False)


    assert set(train_labels) == set(val_labels),\
           "Train and val labels don't correspond:\n{}\n{}".format(set(train_labels),
                                                                   set(val_labels))

    num_classes = len(set(train_labels))

    # --------------------------------------------------------------------------
    # In TensorFlow, you first want to define the computation graph with all the
    # necessary operations: loss, training op, accuracy...
    # Any tensor created in the `graph.as_default()` scope will be part of `graph`
    graph = tf.Graph()
    with graph.as_default():


        # ----------------------------------------------------------------------
        # DATASET CREATION using tf.contrib.data.Dataset
        # https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/data

        # The tf.contrib.data.Dataset framework uses queues in the background to feed in
        # data to the model.
        # We initialize the dataset with a list of filenames and labels, and then apply
        # the preprocessing functions described above.
        # Behind the scenes, queues will load the filenames, preprocess them with multiple
        # threads and apply the preprocessing in parallel, and then batch the data





        def parse_function(filename, label):
            image_string = tf.read_file(filename)
            image_decoded = tf.image.decode_jpeg(image_string, channels=3)          # (1)
            image = tf.cast(image_decoded, tf.float32)

            smallest_side = 256.0
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



        # create placeholder to initialize data

        pathholder = tf.placeholder(tf.string, shape=[None, 1])
        labelholder = tf.placeholder(tf.string, shape=[None, 1])

        train_dataset = tf.data.Dataset.from_tensor_slices((train_filenames, train_labels))
        train_dataset = train_dataset.map(map_func=parse_function, num_parallel_calls=4)



        train_dataset = train_dataset.map(map_func=training_preprocess,num_parallel_calls=4)
        train_dataset = train_dataset.shuffle(buffer_size=10000)  # don't forget to shuffle
        batched_train_dataset = train_dataset.batch(args.batch_size)




        # Validation dataset
        val_dataset = tf.data.Dataset.from_tensor_slices((val_filenames, val_labels))
        val_dataset = val_dataset.map(map_func=parse_function, num_parallel_calls=4)
        val_dataset = val_dataset.map(map_func=val_preprocess, num_parallel_calls=4)
        batched_val_dataset = val_dataset.batch(args.batch_size)

        # Now we define an iterator that can operator on either dataset.
        # The iterator can be reinitialized by calling:
        #     - sess.run(train_init_op) for 1 epoch on the training set
        #     - sess.run(val_init_op)   for 1 epoch on the valiation set
        # Once this is done, we don't need to feed any value for images and labels
        # as they are automatically pulled out from the iterator queues.

        # A reinitializable iterator is defined by its structure. We could use the
        # `output_types` and `output_shapes` properties of either `train_dataset`
        # or `validation_dataset` here, because they are compatible.
        iterator = tf.data.Iterator.from_structure(batched_train_dataset.output_types,
                                                           batched_train_dataset.output_shapes)
        images, labels = iterator.get_next()

        train_init_op = iterator.make_initializer(batched_train_dataset)
        val_init_op = iterator.make_initializer(batched_val_dataset)

        # Indicates whether we are in training or in test mode
        is_training = tf.placeholder(tf.bool)

        # ---------------------------------------------------------------------
        # Now that we have set up the data, it's time to set up the model.
        # For this example, we'll use VGG-16 pretrained on ImageNet. We will remove the
        # last fully connected layer (fc8) and replace it with our own, with an
        # output size num_classes=8
        # We will first train the last layer for a few epochs.
        # Then we will train the entire model on our dataset for a few epochs.

        # Get the pretrained model, specifying the num_classes argument to create a new
        # fully connected replacing the last one, called "vgg_16/fc8"
        # Each model has a different architecture, so "vgg_16/fc8" will change in another model.
        # Here, logits gives us directly the predicted scores we wanted from the images.
        # We pass a scope to initialize "vgg_16/fc8" weights with he_initializer
        vgg = tf.contrib.slim.nets.vgg
        with slim.arg_scope(vgg.vgg_arg_scope(weight_decay=args.weight_decay)):
            logits, _ = vgg.vgg_16(images, num_classes=num_classes, is_training=is_training,
                                   dropout_keep_prob=args.dropout_keep_prob)

        # Specify where the model checkpoint is (pretrained weights).
        model_path = args.model_path
        assert(os.path.isfile(model_path))

        # Restore only the layers up to fc7 (included)
        # Calling function `init_fn(sess)` will load all the pretrained weights.

        variables_all = tf.contrib.framework.get_variables_to_restore()
        var_list = tf.trainable_variables()
        print(len(variables_all))
        for var in var_list:
            print(var.name)



        variables_to_restore = tf.contrib.framework.get_variables_to_restore(exclude=['vgg_16/fc8'])
        init_fn = tf.contrib.framework.assign_from_checkpoint_fn(model_path, variables_to_restore)

        # Initialization operation from scratch for the new "fc8" layers
        # `get_variables` will only return the variables whose name starts with the given pattern
        fc8_variables = tf.contrib.framework.get_variables('vgg_16/fc8')
        fc8_init = tf.variables_initializer(fc8_variables)


        #fc7_variables = tf.contrib.framework.get_variables('vgg_16/fc7')
        #fc7_init = tf.variables_initializer(fc7_variables)


        # iniitalize fc7

        """
        fc7_variables = tf.contrib.framework.get_variables('vgg_16/fc7')
        fc7_init = tf.variables_initializer(fc7_variables)
        """



        # ---------------------------------------------------------------------
        # Using tf.losses, any loss is added to the tf.GraphKeys.LOSSES collection
        # We can then call the total loss easily

        
        tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        loss = tf.losses.get_total_loss()
        
        """
        one_hot_label = tf.one_hot(tf.cast(labels, tf.int32), depth=5)

        # time.sleep(100)
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_label, logits=logits)
        loss = tf.reduce_mean(loss)
        """

        # First we want to train only the reinitialized last layer fc8 for a few epochs.
        # We run minimize the loss only with respect to the fc8 variables (weight and bias).
        # fc8_optimizer = tf.train.GradientDescentOptimizer(learning_rate=args.learning_rate1)

        # fc8_optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate1)
        fc8_optimizer = tf.train.MomentumOptimizer(learning_rate=args.learning_rate1, momentum=0.9)

        # only optimize fc8
        # fc8_train_op = fc8_optimizer.minimize(loss, var_list=fc8_variables)
        # Optimize all varialbes
        fc8_train_op = fc8_optimizer.minimize(loss)

        # Then we want to finetune the entire model for a few epochs.
        # We run minimize the loss only with respect to all the variables.
        full_optimizer = tf.train.GradientDescentOptimizer(args.learning_rate2)
        full_train_op = full_optimizer.minimize(loss)

        # Evaluation metrics
        prediction = tf.to_int32(tf.argmax(logits, 1))
        correct_prediction = tf.equal(prediction, labels)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        saver_all = tf.train.Saver()
        
        init_g = tf.global_variables_initializer()
        tf.get_default_graph().finalize()

    # --------------------------------------------------------------------------
    # Now that we have built the graph and finalized it, we define the session.
    # The session is the interface to *run* the computational graph.
    # We can call our training operations with `sess.run(train_op)` for instance

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True


    with tf.Session(graph=graph, config=config) as sess:
        sess.run(init_g)
        init_fn(sess)  # load the pretrained weights
        sess.run(fc8_init)  # initialize the new fc8 layer
        
        #sess.run(fc7_init)  # initialize the new fc8 layer
        if args.restore_ckpt:
            print("Restore from previous checkpoint file")
            saver_all.restore(sess, args.weightFile)
            print("Finish restoring")
            time.sleep(5)

        # only conduct evaluation and stop the program
        if args.evaluation:
            print("ONLY do evaluation")
            train_acc, train_acc_weighted, train_acc_for_each_class = check_accuracy(sess, correct_prediction, labels, prediction, is_training, train_init_op)
            print('Train, overall accuracy: {:6.5f}, weighted accuracy {:6.5f} >>>>\n\n'.format(train_acc, train_acc_weighted,
                    train_acc_for_each_class[0], train_acc_for_each_class[1], train_acc_for_each_class[2], 
                    train_acc_for_each_class[3], train_acc_for_each_class[4]))

            val_acc, val_acc_weighted, val_acc_for_each_class = check_accuracy(sess, correct_prediction, labels, prediction, is_training, val_init_op)
            print('Val, overall accuracy: {:6.5f}, weighted accuracy {:6.5f} >>>>\n\n \
                class 0: {:5.3f}\t class 1: {:5.3f}\t class 2: {:5.3f}\t class 3: {:5.3f}\t class 4: {:5.3f}\n'.format(val_acc, val_acc_weighted,
                    val_acc_for_each_class[0], val_acc_for_each_class[1], val_acc_for_each_class[2], 
                    val_acc_for_each_class[3], val_acc_for_each_class[4]))

            print("Finish evaluation")
            time.sleep(5)
            sys.exit()

        # Update only the last layer for a few epochs.
        for epoch in range(args.num_epochs1):

            # Run an epoch over the training data.
            print("*"*50 + '\n')
            print('Starting epoch %d / %d' % (epoch + 1, args.num_epochs1))
            # Here we initialize the iterator with the training set.
            # This means that we can go through an entire epoch until the iterator becomes empty.
            sess.run(train_init_op)

            epoch_loss = 0.

            while True:
                try:
                    _, loss_value = sess.run([fc8_train_op, loss], {is_training: True})

                    epoch_loss += loss_value
                except tf.errors.OutOfRangeError:
                    break
            # Check accuracy on the train and val sets every epoch.
            print("Loss: {:6.5f}".format(epoch_loss))

            train_acc, train_acc_weighted, train_acc_for_each_class = check_accuracy(sess, correct_prediction, labels, prediction, is_training, train_init_op)
            print('Train, overall accuracy: {:6.5f}, weighted accuracy {:6.5f} >>>>\n'.format(train_acc, train_acc_weighted))

            for cls_label in range(args.class_num):
                print("\t\tclass {:1d}: {:5.3f} [{:5d}/{:<5d}]".format(cls_label, train_acc_for_each_class[cls_label, 0], 
                    int(train_acc_for_each_class[cls_label, 1]), 
                    int(train_acc_for_each_class[cls_label, 2])))

            val_acc, val_acc_weighted, val_acc_for_each_class = check_accuracy(sess, correct_prediction, labels, prediction, is_training, val_init_op)
            print('Val, overall accuracy: {:6.5f}, weighted accuracy {:6.5f} >>>>\n'.format(val_acc, val_acc_weighted))

            for cls_label in range(args.class_num):
                print("\t\tclass {:1d}: {:5.3f} [{:5d}/{:<5d}]".format(cls_label, val_acc_for_each_class[cls_label, 0], 
                    int(val_acc_for_each_class[cls_label, 1]), 
                    int(val_acc_for_each_class[cls_label, 2])))


            saver_all.save(sess, os.path.join(args.ckpt_dir, 'models'), global_step=epoch)

        # Train the entire model for a few more epochs, continuing with the *same* weights.
        for epoch in range(args.num_epochs2):
            print('Starting epoch %d / %d' % (epoch + 1, args.num_epochs2))
            sess.run(train_init_op)
            iteration = 0
            while True:
                iteration += 1
                try:
                    _, loss_value = sess.run([full_train_op, loss], {is_training: True})
                    if iteration % 50 == 0 and iteration > 0:
                        print("Epoch [{:1d}], iteration [{:3d}], loss: {:6.5f}".format(epoch, iteration, loss_value))
                except tf.errors.OutOfRangeError:
                    break

            # Check accuracy on the train and val sets every epoch

            train_acc = check_accuracy(sess, correct_prediction, is_training, train_init_op)
            val_acc = check_accuracy(sess, correct_prediction, is_training, val_init_op)
            print('Train accuracy: %f' % train_acc)
            print('Val accuracy: %f\n' % val_acc)


if __name__ == '__main__':
    args = parser.parse_args()
    print("ALL the args information")
    print('args.train_dir = {}'.format(args.train_dir))
    print('args.val_dir = {}'.format(args.val_dir))
    print('args.model_path = {}'.format(args.model_path))
    print('args.batch_size = {}'.format(args.batch_size))
    print('args.num_workers = {}'.format(args.num_workers))
    print('args.num_epochs1 = {}'.format(args.num_epochs1))
    print('args.num_epochs2 = {}'.format(args.num_epochs2))
    print('args.learning_rate1 = {}'.format(args.learning_rate1))
    print('args.learning_rate2 = {}'.format(args.learning_rate2))
    print('args.dropout_keep_prob = {}'.format(args.dropout_keep_prob))
    print('args.restore_ckpt = {}'.format(args.restore_ckpt))
    print('args.evaluation = {}'.format(args.evaluation))
    print('args.weightFile = {}'.format(args.weightFile))
    print('args.ckpt_dir = {}'.format(args.ckpt_dir))
    print('args.dn_train = {}'.format(args.dn_train))
    print('args.dn_test = {}'.format(args.dn_test))

    if not os.path.isdir(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
    main(args)