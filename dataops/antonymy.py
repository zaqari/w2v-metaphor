import pandas as pd
import numpy as np

initial = pd.read_csv('w2vMods/files/MasterStimuliList - Kao et al 2014.csv')

antonyms = [[], [],[]]
for i in range(3):
    antonyms[0]+=initial['vehicle'].values.tolist()
    antonyms[1]+=initial['ADJ'+str(i+1)].values.tolist()
    antonyms[2]+=initial['aADJ'+str(i+1)].values.tolist()

df = pd.DataFrame()
df['vehicle'] = antonyms[0]
df['pos'] = antonyms[1]
df['anti'] = antonyms[2]

df.to_csv('w2vMods/files/axes-list.csv', index=False, encoding='utf-8')