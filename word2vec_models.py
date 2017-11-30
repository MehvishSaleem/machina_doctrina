from read_and_select import read_and_select
from gensim.models.word2vec import Word2Vec

class w2v():
    def __init__(self, train_df, test_df):

        train_text = train_df['processed_text']
        test_text = test_df['processed_text']

        train_sentences = [x.split() for x in train_text.values.tolist()]
        test_sentences = [x.split() for x in test_text.values.tolist()]

        sentences = train_sentences + test_sentences

        self.sentences = sentences

        self.params = dict()
        self.params['feat_vec_size'] = 200 # length of vector representation of words
        self.params['window'] = 10 # Context around the word (10 words near the current word in the sentence)
        self.params['min_count'] = 1 # Don't change this!
        self.params['iter'] = 30 # Number of iterations to run the model for training

    def train_model(self):
        self.model = Word2Vec(self.sentences,
                             size=self.params['feat_vec_size'],
                             window=self.params['window'],
                             min_count=self.params['min_count'],
                             iter=self.params['iter'])

    def get_word_vec(self, word):
        if not hasattr(self, "model"):
            raise Exception("Model not found!")

        if not hasattr(self, "sentences"):
            raise Exception("Sentences not found!")

        return self.model.wv.word_vec(word)

    def update_param(self, param, val):
        self.params[param] = val

    def set_sentences(self, sentences):
        self.sentences = sentences

if __name__ == '__main__':
    train_df, test_df = read_and_select([], False)

    w2v_model = w2v(train_df, test_df)
    w2v_model.train_model()

    processed_text_row0 = train_df['processed_text'].values.tolist()[0]

    for word in processed_text_row0.split():
        print("\n\nWord:", word)
        print("============\n")
        print(w2v_model.get_word_vec(word))