import pandas as pd
import numpy as np
from w2vMods.loadBERT import *

class axComp():

    def __init__(self):
        self.BERT = bert()

    def make_axes(self, col1, col2, df):
        return [self.BERT.translate(phrase[0])[0] - self.bert.translate(phrase[1])[0] for phrase in df[[col1, col2]].values]


    def targets(self, col1, col2, df):
        data = [' '.join(df[[col1, col2]].loc[i].values.tolist()) for i in df.index]
        return [self.BERT.translate(phrase)[0].sum(dim=0) for phrase in data]


