from __future__ import absolute_import
from __future__ import division
from __future__ import print_function  # force to use print as function print(args)
from __future__ import unicode_literals

import os
import pickle
import torchvision.datasets as datasets

if __name__ == '__main__':
    dataset_folder = os.path.join(os.path.dirname(__file__), '../datasets')

    datasets.CIFAR10(dataset_folder, train=True, download=True)
    datasets.CIFAR10(dataset_folder, train=False, download=True)
    datasets.CIFAR100(dataset_folder, train=True, download=True)
    datasets.CIFAR100(dataset_folder, train=False, download=True)

    # get labels
    # file = os.path.join(os.path.dirname(__file__), '../datasets/cifar-100-python/meta')
    # with open(file, 'rb') as fo:
    #     dict = pickle.load(fo, encoding='bytes')

    # print(dict)
