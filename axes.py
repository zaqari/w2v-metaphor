import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import w2vMods.mBERT as mb

class axs(mb.axComp):

    def __init__(self):
        super(axs, self).__init__()

        self.axes_embeds = None
        self.axes_names = None


    def make_axes_from_text(self, df, col, delim='<-v->'):
        self.axes_embeds = torch.zeros(size=(len(df[col].unique()), 768))
        self.axes_names = df[col].unique()

        for i, item in enumerate(df[col].unique()):
            w = item.split(delim)
            self.axes_embeds[i] = self.lexeme(w[0]).detach().sum(dim=0).view(-1) - self.lexeme(w[1]).detach().sum(dim=0).view(-1)


    def make_axes_from_columns(self, df, col1, col2):
        self.axes_embeds = torch.zeros(size=(len(df), 768))
        self.axes_names = np.array([' '.join(r) for r in df[[col1, col2]].values])

        for i in range(len(df)):
            w = df[[col1, col2]].loc[i].values
            self.axes_embeds[i] = self.lexeme(w[0]).detach().sum(dim=0).view(-1) - self.lexeme(w[1]).detach().sum(dim=0).view(-1)



    def compare(self, lexical_item):

        c = self.lexeme(lexical_item).detach().sum(dim=0).view(1, -1)

        axes = c @ self.axes_embeds

        sort = axes.argsort(descending=True).numpy()

        return axes, self.axes_names[sort]



