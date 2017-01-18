"""
This is a class that reads training-test-validation datasets for the WikiReading dataset and
Generate training examples along side with the datasets

see more about data generators:
https://keras.io/getting-started/faq/#how-can-i-use-keras-with-datasets-that-dont-fit-in-memory

Author: hadyelsahar@gmail.com
"""
import json
import numpy as np
import pandas as pd
from utils.onehotencoder import OneHotEncoder

class WikiReadingDataGenerator:

    def __init__(self, x_dict=None, y_dict=None):

        if x_dict is not None:
            self.x_vocabulary = pd.read_csv(x_dict, sep="\t", names=["id", "word", "count"])
        else:
            self.x_vocabulary = None

        if y_dict is not None:
            self.y_vocabulary = pd.read_csv(y_dict, sep="\t", names=["id", "word", "count"])
        else:
            self.y_vocabulary = None

        self.y_onehot_encoder = OneHotEncoder()
        self.y_onehot_encoder.fit(self.y_vocabulary["id"].values)


    def generate(self, file_names, batchsize, x_vectorize_function, y_vectorize_function, **kwargs):
        """
        Generate data batch by batch
        The generator is expected to loop over its data indefinitely
        :param file_names: list of file names to read from
        :param batchsize: batch size to return each loop
        :param x_vectorize_function:
        :param y_vectorize_function:
        :param kwargs: kwargs are reserved to x_vectorizefunction since it needs more attributes
        :return:
        """

        while 1:
            x_batch = []
            y_batch = []
            for filename in file_names:
                lines = open(filename).readlines()

                for c, l in enumerate(lines):

                    x = x_vectorize_function(json.loads(l), **kwargs)
                    y = y_vectorize_function(json.loads(l))

                    x_batch.append(x)
                    y_batch.append(y)

                    if len(x_batch) == batchsize:
                        yield x_batch, y_batch
                        x_batch = []
                        y_batch = []

    def demo_x_vectorizefunction(self, j):
        """
        a demo vectorizing function that takes a single json entry from wikireading dataset and return the
        input X which is sequence of word indices

        :param j: one json file entry from the wikireading dataset
        :return: x :  x is the input vector for the dataset has to be a numpy array
        """

        x = np.array(j['document_sequence'], dtype=int)

        return x

    def demo_y_vectorizefunction(self, j):
        """
        a demo vectorizing function that takes a single json entry from wikireading dataset and return the
        the label y for it

        :param j: one json file entry from the wikireading dataset
        :return: y :  y is a label int
        """

        y = np.array(j['raw_answer_ids'][0])
        return y

    def bow_x_vectorizer(self, j, nb_words=None, maxlen=20, padding='post', truncating='post', pad_id=0, oov_id=0, dtype='int32'):
        """
        function to Vectorize each json entry
        each document words are first trimmed and then question sequence of words are added later afterwards

        :param j: one json file entry from the wikireading dataset
        :param nb_words: None or int. Maximum number of words to work with (if set, words that aren't included in the top words will be replaced by the index 0).
        :param maxlen: max sequence length of each document
        :param dtype: type of the array values
        :param padding: 'post' or 'pre'  either pad short sequences from the beginning fo the end
        :param truncating: 'post' or 'pre'  either truncate long sequences from the beginning fo the end
        :param pad_id: wordid to put for padding  default 0
        :param oov_id: out of vocabulary wordid to replace words that doesn't appear in the top nb_words
        :return: x, y :  x of size
                         y is a label
        """
        # pad or truncate
        docseq = j['document_sequence']
        qseq = j['question_sequence']

        diff = len(docseq) + len(qseq) - maxlen

        if diff > 0:
            # truncate
            if truncating == 'post':
                docseq = docseq[0:-diff]
            elif truncating == 'pre':
                docseq = docseq[diff:]

        elif diff < 0:
            # pad
            paddingseq = [pad_id] * abs(diff)
            if padding == 'post':
                docseq = docseq + paddingseq
            elif padding == 'pre':
                docseq = padding + docseq

        seq = docseq + qseq

        # limiting the vocab to top nb_words
        if nb_words is not None:
            seq = [i if i < nb_words else oov_id for i in seq]

        return np.array(seq, dtype=dtype)

    def onehot_y_vectorizer(self, j):
        """
        :param j:
        :return:
        """

        y = np.array(self.y_onehot_encoder.tranform_single(j['raw_answer_ids'][0]), dtype=int)
        return y



def chunks(a, n):
    """
    Yield successive n-sized chunks from a.
    :param a:
    :param n:
    :return:
    """
    for i in range(0, len(a), n):
        yield a[i:i + n]






















