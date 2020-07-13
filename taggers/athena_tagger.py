import json
import os
import pickle
import pprint

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC


def load_json_data(filename):
    data_dir = 'data'
    file_path = os.path.join(data_dir, filename)

    with open(file_path, 'r') as all_json_file:
        label_utt_dict = json.load(all_json_file)

        dataset = []

        for label, utterances in label_utt_dict.items():
            print("Label {} Count {}".format(label, len(utterances)))
            # if len(utterances) < 10:  # Skip sparse labels
            #     print("Skipped label ", label)
            #     continue
            for utt in utterances:
                dataset.append((utt, label))


    unified_df = pd.DataFrame(dataset, columns=['text', 'label'])
    return unified_df

def load_dataframe():
    return load_json_data('all_augmented.json')


def train_json_models():
    df = load_dataframe()

    texts = df['text']
    texts = texts.str.replace(" ' ", "")
    labels = df['label']

    for test_size in [0.1]:
        texts_train, texts_test, labels_train, labels_test = train_test_split(texts, labels, stratify=labels)
        label_list = list(set(labels_test))

        vec = TfidfVectorizer(ngram_range=(1, 4))

        tfidf_train = vec.fit_transform(texts_train, labels_train)
        tfidf_test = vec.transform(texts_test)

        test_split_percent = int(test_size * 100)

        tfidf_test_data = {
            'input': tfidf_test,
            'target': labels_test
        }

        svm = LinearSVC()
        svm.fit(tfidf_train, labels_train)

        with open('svm_model_%d.pkl' % test_split_percent, 'wb') as model_file:
            models_dict = {
                "svm": svm
            }

            pickle.dump(models_dict, model_file)

        print(f"Models and vectorizers saved for split {test_split_percent}")

def get_model_keys():
    for label in set(load_dataframe()['label'].tolist()):
        print(label)

if __name__ == '__main__':
    # train_json_models()
    get_model_keys()