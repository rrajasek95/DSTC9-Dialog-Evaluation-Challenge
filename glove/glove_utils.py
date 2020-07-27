from scipy import spatial
from nltk.tokenize import word_tokenize
import nltk
import numpy as np

score_dict = {
    'NN': 10,
    'JJ': 10,
    'VB': 5,
}


def load_embeddings():
    # load the whole embedding into memory
    glove_path = 'glove/glove.840B.300d.txt'
    embeddings_index = dict()
    f = open(glove_path)
    count = 0
    for line in f.readlines():
        count += 1
        values = line.split()
        word = values[0]
        try:
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        except ValueError:
            print(count)

    f.close()
    print('Loaded %s word vectors.' % len(embeddings_index))
    return embeddings_index


class GloveUtils:
    def __init__(self, data):
        self.tokenizer = {}
        count = 0
        for sentence in data:
            words = word_tokenize(sentence)
            for word in words:
                if word.lower() not in self.tokenizer:
                    self.tokenizer[word.lower()] = count
                    count += 1
        self.embeddings_index = load_embeddings()
        self.embedding_matrix = self.make_embedding_matrix()

    def make_embedding_matrix(self):
        # create a weight matrix for words in training docs
        embedding_matrix = np.zeros((len(self.tokenizer), 300))
        # did_not_find = []
        for word, i in self.tokenizer.items():
            embedding_vector = self.embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
            # else:
            #     did_not_find.append(word)
        return embedding_matrix


def get_cosine_similarity(v1, v2):
    result = 1 - spatial.distance.cosine(v1, v2)
    return result


def get_sentence_glove_embedding(sentence, embedding_matrix, tokenizer):
    sentence = word_tokenize(sentence.lower())
    pos = nltk.pos_tag(sentence)
    arr = np.zeros(300)
    count = 0
    for word in sentence:
        index = tokenizer.get(word)
        score = score_pos(pos[count][1])
        if index is not None:
            arr = arr + (score * embedding_matrix[index])
        count += 1
    return arr / count


def get_max_cosine_similarity(message, knowledge_list, embedding_matrix, tokenizer):
    message_embed = get_sentence_glove_embedding(message, embedding_matrix, tokenizer)
    max_sim = 0
    max_sim_fact = ""
    for element in knowledge_list:
        sim = get_cosine_similarity(message_embed, element[1])
        if sim > max_sim:
            max_sim = sim
            max_sim_fact = element[0]
    return max_sim_fact, max_sim


def get_max_cosine_similarity_infersent(message, knowledge_list, model, knowledge_policy="infersent"):
    if knowledge_policy == "infersent":
        embeddings = model.encode([message], tokenize=True)
    else:
        embeddings = model.encode([message])
    message_embed = embeddings[0]
    max_sim = 0
    max_sim_fact = ""
    for element in knowledge_list:
        sim = get_cosine_similarity(message_embed, element[1])
        if sim > max_sim:
            max_sim = sim
            max_sim_fact = element[0]
    return max_sim_fact, max_sim


def get_cosine_similarity_embs_all(message, knowledge_list, model, knowledge_policy="infersent"):
    if knowledge_policy == "infersent":
        embeddings = model.encode([message], tokenize=True)
    else:
        embeddings = model.encode([message])
    message_embed = embeddings[0]
    return_array = []
    knowledge_set = set()
    for element in knowledge_list:
        if (element[0]) not in knowledge_set:
            knowledge_set.add(element[0])
            new_array = [element[0]]
            sim = get_cosine_similarity(message_embed, element[1])
            new_array.append(sim)
            return_array.append(new_array)
    return return_array

def score_pos(pos):
    if pos in score_dict:
        return score_dict[pos]
    return 1
