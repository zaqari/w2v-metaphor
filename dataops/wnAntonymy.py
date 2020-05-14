import pandas as pd
import numpy as np
from nltk.corpus import wordnet as wn

class antonym_finder():

    def __init__(self):
        super(antonym_finder, self).__init__()
        self.dic = {}

    def create_antonym_list(self, wordlist):
        print('creating list of antonyms from list provided')
        for lex in wordlist:
            for syn in wn.synsets(lex):
                for lemma in syn.lemmas():
                    if lemma.antonyms():
                        self.dic[lemma.name()] = lemma.antonyms()[0].name()
        print('antonym list complete')

    def create_antonym_list_all_synsets(self, pos='n'):
        print('creating list of antonyms from all of wordnet')

        for syn in wn.all_synsets(pos=pos):
            for lemma in syn.lemmas():
                if lemma.antonyms():
                    if (lemma.name() not in self.dic.values()) and (lemma.name() not in self.dic.keys()):
                        self.dic[lemma.name()] = lemma.antonyms()[0].name()
        print('antonym list complete')

    def return_df(self):
        return pd.DataFrame(np.array(self.dic.items()).reshape(-1,2), columns=['key', 'value'])
    
    def save_df(self, save_file_path):
        df = pd.DataFrame(np.array(list(self.dic.items())).reshape(-1,2), columns=['lex', 'antonym'])
        df.to_csv(save_file_path, index=False, encoding='utf-8')



