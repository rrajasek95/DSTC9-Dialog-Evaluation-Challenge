import json
import os
import pickle

import numpy as np
import pandas as pd

from nltk import word_tokenize

from knowledge_index import extract_fact_set, clean


def generate_knowledge_selections(test_freq_conversations, test_freq_reading_set, vectorizer):

    turn_knowledge = []
    for conv_id, conv_data in test_freq_conversations.items():
        conv_reading_set = test_freq_reading_set[conv_id]
        fact_set_1 = set(extract_fact_set(conv_reading_set["agent_1"]))
        fact_set_2 = set(extract_fact_set(conv_reading_set["agent_2"]))

        article_data = conv_reading_set["article"]

        article_indices = ['AS1', 'AS2', 'AS3', 'AS4']

        common_knowledge_set = set()
        if "AS1" in article_data:
            for idx in article_indices:
                sentence = article_data[idx]
                if len(word_tokenize(sentence)) < 5:
                    continue
                common_knowledge_set.add(clean(sentence))
        fact_set_1.update(common_knowledge_set)
        fact_set_2.update(common_knowledge_set)

        agent_knowledge = {
            "agent_1": list(fact_set_1),
            "agent_2": list(fact_set_2)
        }

        for i, turn in enumerate(conv_data["content"]):
            text = turn["message"]

            available_knowledge = agent_knowledge[turn["agent"]]


            text_tfidf = vectorizer.transform([clean(text)])
            knowledge_tfidf = vectorizer.transform(available_knowledge)
            similarity = np.squeeze(np.asarray(text_tfidf.dot(knowledge_tfidf.transpose()).todense()))

            top_n_indices = similarity.argsort()[-3:][::-1].tolist()

            knowledge_1 = available_knowledge[top_n_indices[0]]
            if i > 0:
                same_as_prev_knowledge = knowledge_1 == turn_knowledge[-1]["knowledge_1"]
            else:
                same_as_prev_knowledge = False

            data = {
                "conversation_id": conv_id,
                "turn": (i + 1),
                "text": text,
                "knowledge_1": knowledge_1,
                "knowledge_1_similarity": similarity[top_n_indices[0]],
                "knowledge_2": available_knowledge[top_n_indices[1]],
                "knowledge_2_similarity": similarity[top_n_indices[1]],
                "knowledge_3": available_knowledge[top_n_indices[2]],
                "knowledge_3_similarity": similarity[top_n_indices[2]],
                "same_as_prev_knowledge": same_as_prev_knowledge
            }

            turn_knowledge.append(data)

    test_freq_knowledge_dataframe = pd.DataFrame(turn_knowledge,
                                                 columns=[
                                                     'conversation_id',
                                                     'turn',
                                                     'text',
                                                     'knowledge_1',
                                                     'knowledge_1_similarity',
                                                     'knowledge_2',
                                                     'knowledge_2_similarity',
                                                     'knowledge_3',
                                                     'knowledge_3_similarity',
                                                     'same_as_prev_knowledge'
                                                 ])

    test_freq_knowledge_dataframe.to_csv(os.path.join(
        'tc_processed',
        'test_freq_tfidf.csv'
    ))


if __name__ == '__main__':
    topical_chats_path = 'alexa-prize-topical-chat-dataset'

    test_freq_conv_file = os.path.join(
        topical_chats_path,
        'conversations',
        'test_freq.json'
    )

    test_freq_reading_set_path = os.path.join(
        topical_chats_path,
        'reading_sets',
        'post-build',
        'test_freq.json'
    )

    knowledge_index_path = os.path.join(
        'tc_processed',
        'tc_knowledge_index.pkl'
    )

    with open(test_freq_conv_file, 'r') as test_freq_conv_f:
        test_freq_conversations = json.load(test_freq_conv_f)

    with open(test_freq_reading_set_path, 'r') as test_freq_reading_set_file:
        test_freq_reading_set = json.load(test_freq_reading_set_file)

    with open(knowledge_index_path, 'rb') as knowledge_index_file:
        index_dict = pickle.load(knowledge_index_file)
        vectorizer = index_dict["vectorizer"]

    generate_knowledge_selections(test_freq_conversations, test_freq_reading_set, vectorizer)