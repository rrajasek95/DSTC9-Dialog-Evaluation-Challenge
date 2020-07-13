from scipy import spatial
from nltk.tokenize import word_tokenize
import nltk
import json
import os
import numpy as np
import csv
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from collections import OrderedDict
import random
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt


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
        # Tokenize the sentences
        self.tokenizer = Tokenizer()
        # preparing vocabulary
        self.vocab = set()
        for sentence in data:
            words = word_tokenize(sentence)
            for word in words:
                self.vocab.add(word.lower())
        self.embeddings_index = load_embeddings()
        self.tokenizer.fit_on_texts(list(self.vocab))
        self.embedding_matrix = self.make_embedding_matrix()

    def make_embedding_matrix(self):
        # create a weight matrix for words in training docs
        embedding_matrix = np.zeros((len(self.vocab) + 1, 300))
        did_not_find = []
        for word, i in self.tokenizer.word_index.items():
            embedding_vector = self.embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
            else:
                did_not_find.append(word)
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
        index = tokenizer.word_index.get(word)
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
    # print("S: ", str(max_sim))
    return max_sim_fact, max_sim


def score_pos(pos):
    if pos == 'NN':
        return 10
    elif pos == 'JJ':
        return 10
    elif pos == 'VB':
        return 5
    else:
        return 1
