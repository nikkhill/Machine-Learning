import numpy as np


class EmbWeights():

    def __init__(self, glove_path):
        self.glove_path = glove_path
        self.glove = {}
        self.glove_dim = None
        self.initialized = False

    def read_glove(self):
        if not self.initialized:
            with open(self.glove_path, 'rb') as f:
                for l in f:
                    line = l.decode().split()
                    word = line[0]
                    vect = np.array(line[1:]).astype(np.float)
                    self.glove[word] = vect
                    if not self.glove_dim:
                        self.glove_dim = len(vect)
            self.initialized = True

    def create_emb_matrix(self, word2index):
        """
        Reads glove file and returns the embedding weights
        :param glove_path: path to glove file
        :param word2index: word to index dictionary
        :return: emb weight np array
        """
        self.read_glove()
        emb_matrix = np.zeros((len(word2index), self.glove_dim))
        for word in word2index:
            idx = word2index[word]
            if word in self.glove:
                emb_matrix[idx] = self.glove[word]
            elif word != "<pad>":
                emb_matrix[idx] = np.random.rand(self.glove_dim)
        return emb_matrix


# test
if __name__ == '__main__':
    w2i = {
        "<pad>": 0,
        "<bos>": 1,
        "<eos>": 2,
        "<unk>": 3,
        ".": 4,
        ",": 5,
        "the": 6,
        "to": 7,
        "and": 8,
        "someunknownword1": 9,
        "someunknownword2": 10,
    }

    ew = EmbWeights(r"C:\MyFiles\cmu coursework\e2e\glove.6B\glove.6B.50d.txt")
    print(ew.create_emb_matrix(w2i))
    print(ew.create_emb_matrix(w2i))
