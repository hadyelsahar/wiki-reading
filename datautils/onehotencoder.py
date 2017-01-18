import numpy as np

class OneHotEncoder:
    """
    Memory efficient one hot encoder when all labels are integers in a fixed range
    which is the case in wikireading dataset
    More memory efficient than Scikitlearn and Keras One hot vectorizers for labels
    """
    def __init__(self):

        self.max = None
        self.min = None
        self.shape = None

    def fit(self, vocabulary):
        """
        :param vocabulary: list of int
        :return:
        """

        voc = np.array(vocabulary)
        self.max = voc.max()
        self.min = voc.min()
        self.shape = self.max-self.min+1

    def transform(self, x):

        x_onehot = np.array([self.tranform_single(i) for i in x], dtype=int)
        return x_onehot

    def tranform_single(self, x):

        x_onehot = np.zeros(self.max-self.min+1)
        x_onehot[x-self.min] = 1
        return x_onehot
