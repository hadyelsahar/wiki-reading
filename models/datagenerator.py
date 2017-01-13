"""
This is a class that reads training-test-validation datasets for the WikiReading dataset and
Generate training examples along side with the datasets

see more about data generators:
https://keras.io/getting-started/faq/#how-can-i-use-keras-with-datasets-that-dont-fit-in-memory

Author: hadyelsahar@gmail.com
"""

import os
from fnmatch import  fnmatch
import json
import  numpy as np

class WikiReadingDataGenerator:

    def __init__(self, datafolder, vectorize_function=None):
        """
        :param datafolder: path that contains all the unzipped data folders
        """
        self.datafolder = datafolder

        if vectorize_function is None:
            self.vectorize_function = self.demo_vectorizefunction()
        else:
            self.vectorize_function = vectorize_function


    def vectorizefile(self, filename):
        """
        a function takes a filename and returns input and output data for it
        in testing mode set TESTING = True
        :param filename: file name to extract data from and vectorize
        :param vectorize_function: function to extract vector input of the model given 1 json
                                    entry from the dataset
        :return:
        """

        lines = open(filename).readlines()

        X = None
        Y = np.zeros((len(lines),))

        for c, l in enumerate(lines):

            x, y = self.vectorize_function(json.loads(l))

            # define X shape in the first loop is faster than appending
            if X is None:
                X = np.zeros(np.concatenate([len(lines)], x.shape))

            X[c:] = x
            y[c] = y

        return X, Y


    def generate(self, filesregex, batchsize):
        """
        :param filesregex: regex to filter more files in datafolder
        :return:
        """

        if filesregex is not None:
            filenames = [os.path.join(self.datafolder, f) for f in os.listdir(self.datafolder) if fnmatch(f, filesregex)]
        else:
            filenames = [os.path.join(self.datafolder, f) for f in os.listdir(self.datafolder)]

        while 1:
            for filename in filenames:
                X, Y = self.vectorizefile(filename, batchsize)
                Xs =


    def demo_vectorizefunction(self, j):
        """
        a function taking one input json entry from dataset and return input vectors and labels
        for the model.

        :param j: one json file entry from the wikireading dataset
        :return: x, y :  x is the input vector for the dataset and
        """

        y = j['raw_answer_ids']
        x = j['document_sequence']

        return x, y


def chunks(l, n):
    """
    Yield successive n-sized chunks from l.
    :param l:
    :param n:
    :return:
    """
    for i in range(0, len(l), n):
        yield l[i:i + n]




















