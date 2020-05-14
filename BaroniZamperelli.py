import pandas as pd
import numpy as np
from sklearn.cross_decomposition import PLSRegression as pls
import w2vMods.mBERT as mb

class regressor(mb.axComp):

    def __init__(self, decomp_size):
        super(regressor, self).__init__()
        self.decomp_size = decomp_size
        self.target_vocab = {}
        self.vocabulary = {}

    def construct(self, X, Y):
        """

        :param X: a lexical item input as str. Typically an adjective.
        :param Y: a phrasal input as str. Typically an NP <- ADJ + NN
        :return: plsr weights
        """

        lex = self.lexeme(X)
        self.target_vocab[Y] = self.target(Y)

        plsr = pls(n_components=self.decomp_size)
        plsr.fit(lex, self.target_vocab[Y])

        return plsr