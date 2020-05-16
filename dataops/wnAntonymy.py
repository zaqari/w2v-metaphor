import pandas as pd
import numpy as np
from nltk.corpus import wordnet as wn

class antonym_list_builder():

    def __init__(self):
        super(antonym_list_builder, self).__init__()
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

class orthoganal_antonyms(mb.axComp):

    def __init__(self, w1, w2, data, vec_size=768):
        super(orthoganal_antonyms, self).__init__()

        """self.vocab = list(
            set(
                data[w1].unique().tolist() + data[w2].unique().tolist()
            )
        )"""

        self.vec_size = vec_size
        self.cos = nn.CosineSimilarity(dim=-1)

        self.df = data[[w1, w2]]
        self.vocab = self._initialize_vocabulary_()
        self.matrix = None


    def _initialize_vocabulary_(self):
        print('initializing vocabulary of antonym axes')
        no_repeats=[]
        for r in self.df.values.tolist():
            if (r not in no_repeats) and (r[::-1] not in no_repeats):
                no_repeats.append(r)
        print('{} antonym axes built'.format(len(no_repeats)))
        return pd.DataFrame(np.array(no_repeats).reshape(-1, 2), columns=['w1', 'w2'])


    def similarity_matrix(self):
        """
        creates a matrix of cosine similarity comparisons between
        all lexical items in the list.

        :return: Does not return a value, but sets self.matrix
        """
        print('initializing cosine similarity matrix')

        #(1) Using BERT.lexeme(), create a tensor of all the vocab
        #    items in our vocabulary
        vocabulary = torch.zeros(size=(len(self.vocab), self.vec_size))
        for i, w in enumerate(self.vocab.values):
            vocabulary[i] = self.lexeme(w[0]).detach().sum(dim=0).view(-1) - self.lexeme(w[1]).detach().sum(dim=0).view(-1)
        print('vocaulary of shape {} calculated'.format(vocabulary.shape))

        #(2) Generate Cosine Similarity Matrix for each lexeme compared to total dataset.
        cossim = np.array([self.cos(i, vocabulary).view(-1).tolist() for i in vocabulary]).reshape(-1, len(vocabulary))

        #(3) Convert results from step (2) into a dataframe
        self.matrix = pd.DataFrame(cossim, columns=self.vocab)
        vocab_names = ['<-v->'.join(i) for i in self.vocab.values]
        self.matrix.columns = vocab_names
        self.matrix['lex'] = vocab_names
        print('cosine similarity matrix complete')

    def export_similarity_matrix(self, datapath):
        self.matrix.to_csv(datapath, index=False, encoding='utf-8')

    def import_similarity_matrix(self, datapath):
        self.matrix = pd.read_csv(datapath)

    def generate_axes(self, max_similarity=.5):
        open_columns = self.matrix['lex'].unique().tolist()

        interpretable_axes = []

        # (1) Randomly select columns for comparison
        column_order = np.random.choice(open_columns, size=len(open_columns), replace=False)
        for col in column_order:
            if col in open_columns:
                # if the item is in the available list of columns, we'll proceed. But first,
                # we remove it from the list to avoid future issues.
                interpretable_axes.append(open_columns.pop(col))

                # I believe this is correct. Because the magnitude in cosine space defines
                # directionality of the vector, but not similarity (.5 and -.5 are both 50%
                # similar, but -.5 is going in the opposite direction) I opt to use the .abs()
                # of the values in the matrix.
                distant_axes = self.matrix[col].values.__abs__()
                # This little bit of linear algebra makes it so that values that are above our
                # cutoff for maxmimum similarity end up being even higher and thus pushed to
                # the end of the step in which we sort our values from smallest cosine
                # similarity to largest.
                distant_axes = ((distant_axes > max_similarity) == 0) + distant_axes

                # Now, we sort our values, and remove and values that are not in the available
                # columns to select from. We then pick the smallest one of these values and send
                # it to interpretable axes.
                selection = [i for i in self.matrix['lex'].values[distant_axes.argsort()] if i in open_columns][0]
                interpretable_axes.append(selection)

        return interpretable_axes

    def viterbi_walk_generate_axes(self, max_similarity=.5, n_steps=300):
        """
        Uses a viterbi like algorithm to "walk" through the matrix
        and select the items that have the lowest cosine similarity
        that it hasn't seen yet.

        :param max_similarity:
        :param n_steps:
        :return:
        """

        open_columns = self.matrix['lex'].unique().tolist()

        current_column = np.random.choice(open_columns[:-1], size=1, replace=False)[0]
        interpretable_axes = [open_columns.pop(open_columns.index(current_column))]

        for _ in range(n_steps):
            x = self.matrix[current_col].loc[~self.matrix['lex'].isin(interpretable_axes)].values
            
            z = self.matrix[interpretable_axes].loc[self.matrix[current_col].isin(x)].values.sum(axis=1)
            
            xz = x + z + (x > max_similarity)

            organized = self.matrix['lex'].loc[self.matrix[current_col].isin(x)].values[xz.argsort()]
            current_col = [col for col in organized if col in open_columns][0]

            interpretable_axes.append(open_columns.pop(open_columns.index(current_col)))

        return interpretable_axes
