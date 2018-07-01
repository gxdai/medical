import numpy as np
from random import shuffle
import scipy.io as sio
import argparse
import time
import sys
from PIL import Image


class Dataset:
    def __init__(self, train_list, test_list, width, height, channel, class_num=5):
        # Load training images (path) and labels
        """
        train_list:     training list file.
        test_list:      testing list file.
        class_num:      The total number of class
        """
        self.class_num = class_num
        self.width = width
        self.height = height
        self.channel = channel

        # Training data
        with open(train_list) as f:
            lines = f.readlines()
        self.train_data = [line.rstrip('\n') for line in lines]
        # shullfe all the data
        shuffle(self.train_data)
        self.train_num = len(self.train_data)


        # Testing data
        with open(test_list) as f:
            lines = f.readlines()
        self.test_data = [line.rstrip('\n') for line in lines]
        self.test_num = len(self.test_data)

        # all the pointer
        self.train_ptr = 0
        self.test_ptr = 0



    def next_batch(self, batch_size, phase):
        # Get next batch of image (path) and labels
        if phase == 'train':
            # Load training sketch
            if self.train_ptr + batch_size < self.train_num:
                batch_paths = self.train_data[self.train_ptr:self.train_ptr+batch_size]
                self.train_ptr += batch_size
            else:
                # shuffle the list
                shuffle(self.train_data)
                self.train_ptr = 0
                batch_paths = self.train_data[self.train_ptr:self.train_ptr+batch_size]
                self.train_ptr += batch_size
        elif phase == 'test':
            # Load training sketch
            if self.test_ptr + batch_size < self.test_num:
                batch_paths = self.test_data[self.test_ptr:self.test_ptr+batch_size]
                self.test_ptr += batch_size
            else:
                self.test_ptr = 0
                batch_paths = self.test_data[self.test_ptr:self.test_ptr+batch_size]
                self.test_ptr += batch_size

        return batch_paths

        """
        # Read images
        feaset = np.zeros((batch_size, self.width, self.height, self.channel))     ### 4096 is the feature size
        labelset = np.zeros((batch_size, 1))     ### 4096 is the feature size
        for i, path in enumerate(batch_paths):
            print(path)
            time.sleep(1)
            imgPath, label = path.split(' ')
            feaset[i] = self.loadimage(imgPath)
            labelset[i] = int(label)

        return feaset, labelset
        """



    def loadimage(self, filepath):
        img = Image.open(filepath)
        img = img.resize((self.width,self.height), Image.ANTIALIAS)
        img = np.asarray(img)

        return img


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='This is for loading shapenet partial data')
    parser.add_argument('--train_list', type=str, default='./train.txt', help='The training list file')
    parser.add_argument('--test_list', type=str, default='./test.txt', help='The testing list file')
    parser.add_argument('--class_num', type=int, default=5, help='the total number of class')
    parser.add_argument('--width', type=int, default=224)
    parser.add_argument('--height', type=int, default=224)
    parser.add_argument('--channel', type=int, default=3)
    args = parser.parse_args()
    data = Dataset(train_list=args.train_list, 
        test_list=args.test_list, 
        width=args.width,
        height=args.height,
        channel=args.channel,
        class_num=args.class_num)
    print('\n\n\n\n\n\n\n')
    for _ in range(1500):
        start_time = time.time()
        feaset, labelset = data.next_batch(5, 'train')
        print("time cost: {}".format(time.time() - start_time))
        print(feaset.shape)
        print(labelset.shape) 
