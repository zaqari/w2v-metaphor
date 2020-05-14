import gensim
import gensim.models.word2vec as w2v
import gensim.downloader as api
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import zipfile

#zip=zipfile.ZipFile('w2vMods/files/glove.6B.zip')

class GloveModel():

    def __init__(self):
        self.wv = []
        self.embedding_cols = None

    def loadGlove(self, dims=300):
        file = open("w2vMods/files/glove-6B-300d.txt", 'r')
        model = {}
        print('Loading Glove Model.')
        for _, line in enumerate(file.readlines()):
            splitline = line.split()
            self.wv.append(np.array([[splitline[0]]+[np.float(i) for i in splitline[1:]]]))

        self.embedding_cols = [str(i) for i in range(dims)]
        self.wv = pd.DataFrame(np.array(self.wv).reshape(-1,dims+1), columns=['lex']+self.embedding_cols)

        print('Glove Model loaded: {} lexemes complete.'.format(len(model)))

    def cosTopN(self, lex, N):
        lex = self.wv[self.embedding_cols].loc[self.wv['lex'].isin([lex])].values
        vals = self.wv[self.embedding_cols].values
        delta = cosine_similarity(lex, vals)
        i = np.argsort(delta)[1:1 + N]
        return [(self.wv['lex'].loc[j], delta[j]) for j in i]

    def eucTopN(self, lex, N):
        dif = self.wv[self.embedding_cols].loc[self.wv['lex'].isin([lex])].values - self.wv[self.embedding_cols].values
        delta = np.sum(dif * dif, axis=1)
        i = np.argsort(delta)[1:1+N]
        return [(self.wv['lex'].loc[j], delta[j]) for j in i]
