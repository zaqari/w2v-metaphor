import gensim
import gensim.models.word2vec as w2v
import gensim.downloader as api
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

class mu():

    def __init__(self, vectors):
        self.vecs = vectors

        if len(self.vecs) > 2:
            self.vecs = self.vecs[:2]

    def gaussian(self):
        """
        returns the gaussian mean of the vectors
        :return:
        """
        numerator = self.vecs.sum(axis=0)
        denominator = self.vecs.std(axis=0).prod()
        return numerator/denominator

    def mean(self):
        """
        returns the euclidean mean of the vectors
        :return:
        """
        return self.vecs.mean(axis=0)

    def midway(self):
        """
        returns the midpoint between the two vectors
        :return:
        """
        return self.vecs[0] + ((self.vecs[0] - self.vecs[1])/2)
