"""
This is a class that reads training-test-validation datasets for the WikiReading dataset and
Generate training examples along side with the datasets

see more about data generators:
https://keras.io/getting-started/faq/#how-can-i-use-keras-with-datasets-that-dont-fit-in-memory

Author: hadyelsahar@gmail.com
"""
import json
import numpy as np

class WikiReadingDataGenerator:

    def vectorizefile(self, filename, vectorize_function):
        """
        a function takes a filename and returns input and output data for it
        in testing mode set TESTING = True
        :param filename: file name to extract data from and vectorize
        :param vectorize_function: function to extract vector input of the model given 1 json
                                    entry from the dataset
        :return:
        """

        lines = open(filename).readlines()

        X = []
        Y = []

        for c, l in enumerate(lines):

            x, y = vectorize_function(json.loads(l))
            X.append(x)
            Y.append(y)

        return X, Y


    def generate(self, file_names, batchsize, vectorize_function=None):
        """
        Generate data batch by batch
        The generator is expected to loop over its data indefinitely

        :param file_names: list of file names to read from
        :param batchsize: batch size to return each loop
        :return:
        """

        if vectorize_function is None:
            vectorize_function = WikiReadingDataGenerator.demo_vectorizefunction

        while 1:
            for filename in file_names:
                X, Y = self.vectorizefile(filename, vectorize_function)

                for x, y in zip(chunks(X, batchsize), chunks(Y, batchsize)):
                    yield x, y

    @staticmethod
    def demo_vectorizefunction(self, j):
        """
        a demo vectorizing function that takes a single json entry from wikireading dataset and return the
        input X and the label y for it

        :param j: one json file entry from the wikireading dataset
        :return: x, y :  x is the input vector for the dataset has to be a numpy array
                         y is a label
        """

        y = np.array(j['raw_answer_ids'])
        x = np.array(j['document_sequence'], dtype=int)
        return x, y




def chunks(a, n):
    """
    Yield successive n-sized chunks from a.
    :param a:
    :param n:
    :return:
    """
    for i in range(0, len(a), n):
        yield a[i:i + n]




















