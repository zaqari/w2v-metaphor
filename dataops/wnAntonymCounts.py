from w2vMods.dataops.wnAntonymy import *
df = pd.read_csv('w2vMods/files/adj-antonyms.csv')
axes = orthoganal_antonyms('lex', 'antonym', df)
axes.import_similarity_matrix('w2vMods/files/similarity-matrix.csv')

outputs=[]
for axs in axes.matrix['lex'].unique():
    ants = axs.split('<-v->')
    ct=0
    for lex in ants:
        for i in wn.lemmas(lex, pos='a'):
            ct+=int(i.count())
    outputs.append(ct)

dfcounts = pd.DataFrame()
dfcounts['axis']=axes.matrix['lex'].unique()
dfcounts['count']=outputs

dfcounts.to_csv('w2vMods/files/axis-counts.csv', index=False, encoding='utf-8')

dfcounts.sort_values(by='count', ascending=False)