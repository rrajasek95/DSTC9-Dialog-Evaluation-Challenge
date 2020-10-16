import itertools
import json
import logging
import os
import pickle

import torch
from nltk import word_tokenize
from sklearn.metrics.pairwise import linear_kernel
from tqdm import tqdm
from glove.glove_utils import get_max_cosine_similarity, get_max_cosine_similarity_embs_models
from knowledge_index import clean
from sentence_transformers import SentenceTransformer
from spacy.lang.en import English
from pd_nrg.ranker import BertRankerRetriever, InfersentRankerRetriever, EmbRankerRetriever
from copy import deepcopy
CONFIG_NAME = 'config.json'

logger = logging.getLogger(__file__)


def transform_da(da_str):
    return " ".join([f"<{da}>" for da in da_str.split(" ")])

def segment_src(src):
    nlp = English()
    sentencizer = nlp.create_pipe("sentencizer")
    nlp.add_pipe(sentencizer)
    segmented_conversations = []
    for conversation in src:
        segmented_conversation = []
        for turn in conversation:
            doc = nlp(turn)
            segmented_conversation.append([segment.text for segment in doc.sents])
        segmented_conversations.append(segmented_conversation)
    return segmented_conversations

def segment_tgt(tgt):
    nlp = English()
    sentencizer = nlp.create_pipe("sentencizer")
    nlp.add_pipe(sentencizer)
    output = []
    for turn in tgt:
        output.append([each.text for each in nlp(turn).sents])
    return output

def load_data(dataset_path, split, training_configuration, generation_config):

    path_prefix = os.path.join(dataset_path, split)

    # Splitting history into multiple sentences for ease of further processing
    src = [l.strip().split("_eos")[:-1] for l in open(path_prefix + '.src').readlines()]
    tgt = [l.strip().replace("_go", "").replace("_eos", "") for l in open(path_prefix + '.tgt').readlines()]
    fct = [l.strip() for l in open(path_prefix + '.fct').readlines()]

    if generation_config != "sentence":
        return prepare_turn_wise_data(fct, path_prefix, src, tgt, training_configuration)
    elif generation_config == "sentence":
        return prepare_sentence_wise_data(fct, path_prefix, src, tgt, training_configuration)


def load_data_for_sentence_generation(dataset_path, split, training_configuration):
    path_prefix = os.path.join(dataset_path, split)

    # Splitting history into multiple sentences for ease of further processing
    src = [l.strip().split("_eos")[:-1] for l in open(path_prefix + '.src').readlines()]
    tgt = [l.strip().replace("_go", "").replace("_eos", "") for l in open(path_prefix + '.tgt').readlines()]
    fct = [l.strip() for l in open(path_prefix + '.fct').readlines()]

    segmented_conversation_contexts = segment_src(src)
    segmented_responses = segment_tgt(tgt)

    if training_configuration != "baseline":
        history_da, history_knowledge, resp_da = load_da_history_data(path_prefix, training_configuration)
    else:
        history_da, history_knowledge, resp_da = load_dummy_da_history(
            segmented_conversation_contexts,
            segmented_responses)

    segmented_sent = segment_src(src)
    segmented_tgt = segment_tgt(tgt)

    context = [zip(s, h, k) for (s, h, k) in zip(segmented_sent, history_da, history_knowledge)]

    return list(zip(context, zip(segmented_tgt, resp_da, fct)))

def load_da_history_data(path_prefix, training_configuration):

    history_da_file = path_prefix + (".src.da" if training_configuration == "kd-pd-nrg" else ".src.swbd3.da")
    history_resp_file = path_prefix + (".tgt.da" if training_configuration == "kd-pd-nrg" else ".tgt.swbd3.da")

    history_da = [list(map(transform_da, l.strip().split("_eos")[:-1])) for l in
                  open(history_da_file).readlines()]
    history_da = [[each.replace("<>", "") for each in history_da[i]] for i in
                  range(len(history_da))]
    history_knowledge = itertools.repeat(itertools.repeat(""))
    # history_knowledge = [l.strip().split("_eos")[:-1] for l in open(path_prefix + ".src.fct")]
    # We load the DAs as an iterable to make it compatible with the baseline itertools repeat logic
    resp_da = [transform_da(l.strip().replace("_go ", "").replace(" _eos", "")).split(" ") for l in open(history_resp_file).readlines()]

    return history_da, history_knowledge, resp_da

def load_dummy_da_history(segmented_conversations, segmented_responses):
    # We need src information to produce the right number of segments

    dummy_data = []
    for conversation in segmented_conversations:
        conversation_dummy = []
        for turn in conversation:
            turn_dummy = []
            for segment in turn:
                turn_dummy.append(None)
            conversation_dummy.append(turn_dummy)

        dummy_data.append(conversation_dummy)

    dummy_resp_data = []
    for response in segmented_responses:
        response_dummy = []
        for segment in response:
            response_dummy.append(None)
        dummy_resp_data.append(response_dummy)
    return dummy_data, dummy_data, dummy_resp_data

def prepare_sentence_wise_data(fct, path_prefix, src, tgt, training_configuration):
    segmented_conversation_contexts = segment_src(src)
    segmented_responses = segment_tgt(tgt)
    if training_configuration != "baseline":
        history_da, history_knowledge, resp_da = load_da_history_data(path_prefix, training_configuration)
    else:
        history_da, history_knowledge, resp_da = load_dummy_da_history(
            segmented_conversation_contexts,
            segmented_responses)

    examples = []

    for i in range(len(segmented_conversation_contexts)):
        conversation_context = segmented_conversation_contexts[i]
        for j in range(len(segmented_responses[i])):
            # Previous turns + user's sentences
            sentence_history = conversation_context + segmented_responses[:j]

            if training_configuration != "baseline":
                sentence_act_history = history_da[i] + resp_da[i][:j]
                sentence_knowledge_history = [fct[i] if j != 0 else ""]
            else:
                sentence_act_history = None
                sentence_knowledge_history = [fct[i] if j != 0 else ""]
            examples.append(
                ((sentence_history, sentence_act_history, sentence_knowledge_history),
                 (segmented_responses[i][j], resp_da[i][j], fct[i])))

    return examples


def prepare_turn_wise_data(fct, path_prefix, src, tgt, training_configuration):
    if training_configuration != "baseline":
        history_da_file = path_prefix + (".src.da" if training_configuration == "kd-pd-nrg" else ".src.swbd3.da")
        history_resp_file = path_prefix + (".tgt.da" if training_configuration == "kd-pd-nrg" else ".tgt.swbd3.da")

        history_da = [list(map(transform_da, l.strip().split("_eos")[:-1])) for l in open(history_da_file).readlines()]
        history_da = [[each.replace("<>", "") for each in history_da[i]] for i in
                      range(len(history_da))]
        history_knowledge = itertools.repeat(itertools.repeat(""))
        # history_knowledge = [l.strip().split("_eos")[:-1] for l in open(path_prefix + ".src.fct")]
        resp_da = [transform_da(l.strip()) for l in open(history_resp_file).readlines()]
    else:
        history_da = itertools.repeat(itertools.repeat(None))
        history_knowledge = itertools.repeat(itertools.repeat(None))
        resp_da = itertools.repeat(None)
    context = [zip(s, h, k) for (s, h, k) in zip(src, history_da, history_knowledge)]
    return list(zip(context, zip(tgt, resp_da, fct)))


def get_dataset(tokenizer, dataset_path, dataset_cache, training_configuration, generation_config):
    dataset_cache = dataset_cache + '_' + type(tokenizer).__name__
    if dataset_cache and os.path.isfile(dataset_cache):
        logger.info("Load tokenized dataset from cache at %s", dataset_cache)
        dataset = torch.load(dataset_cache)
    else:
        logger.info("Loading dataset from %s", dataset_path)

        splits = ['train', 'valid_freq', 'test_freq', 'test_rare', 'valid_rare']
        dataset = dict()

        for split in splits:
            data_items = load_data(dataset_path, split, training_configuration, generation_config)

            def tokenize(obj):
                if obj is None:
                    return None
                if isinstance(obj, str):
                    return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
                if isinstance(obj, tuple):
                    return tuple(tokenize(o) for o in obj)
                return list(tokenize(o) for o in obj)

            logger.info("Tokenize the dataset")
            dataset[split] = tokenize(data_items)
        torch.save(dataset, dataset_cache)
    return dataset

def get_dataset_sentence_generation(tokenizer, dataset_path, dataset_cache, training_configuration):
    dataset_cache = dataset_cache + '_' + type(tokenizer).__name__
    if dataset_cache and os.path.isfile(dataset_cache):
        logger.info("Load tokenized dataset from cache at %s", dataset_cache)
        dataset = torch.load(dataset_cache)
    else:
        logger.info("Loading dataset from %s", dataset_path)

        splits = ['train', 'valid_freq', 'test_freq', 'test_rare', 'valid_rare']
        dataset = dict()

        for split in splits:
            data_items = load_data_for_sentence_generation(dataset_path, split, training_configuration)

            # def tokenize(obj):
            #     if obj is None:
            #         return None
            #     if isinstance(obj, str):
            #         return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
            #     if isinstance(obj, tuple):
            #         return tuple(tokenize(o) for o in obj)
            #     return list(tokenize(o) for o in obj)

            logger.info("Prepare but not tokenize the dataset")
            dataset[split] = data_items
        torch.save(dataset, dataset_cache)
    return dataset


def extract_fact_set_mapped(factsets):
    sentences = []
    original_sentences = dict()
    for idx, data in factsets.items():

        fun_facts = data.get("fun_facts")

        if fun_facts:
            for fact in fun_facts:
                cleaned_fact = clean(fact)
                sentences.append(cleaned_fact)
                original_sentences[cleaned_fact] = fact

        short_wiki = data.get("shortened_wiki_lead_section")

        if short_wiki:
            cleaned_swiki = clean(short_wiki)
            sentences.append(cleaned_swiki)
            original_sentences[cleaned_swiki] = short_wiki

        summarized_wiki = data.get("summarized_wiki_lead_section")

        if summarized_wiki:
            cleaned_sum_wiki = clean(summarized_wiki)
            sentences.append(cleaned_sum_wiki)
            original_sentences[cleaned_sum_wiki] = summarized_wiki
    return original_sentences


def process_split_turn(dataset_path, split, tokenizer, index, knowledge_policy, sentiment=False, ranker=None):
    vec, dialog_act = index
    path_prefix = os.path.join(dataset_path, split)
    reading_set_path = os.path.join(dataset_path, 'reading_sets', f'{split}.json')
    data = []
    with open(path_prefix + '.json', 'r') as annotated_split_file, \
            open(reading_set_path, 'r') as reading_set_file:
        annotated_data = json.load(annotated_split_file)
        reading_set = json.load(reading_set_file)
        for conv_id, conv_data in tqdm(annotated_data.items()):
            context = []
            agent_knowledge, agent_mapping = prepare_reading_set_for_conversation(conv_id, reading_set)
            for turn in conv_data["content"]:
                response = turn["message"]

                available_knowledge = agent_knowledge[turn["agent"]]
                current_turn_data = prepare_turn_data(agent_mapping, available_knowledge, conv_id,
                                                      dialog_act, knowledge_policy, response,
                                                      tokenizer, turn, vec, sentiment, ranker=ranker)
                data.append((context, current_turn_data))
                context = context + [current_turn_data]
    return data


# context: (history_turn_info, history_da, history_facts)
# [[int]] - [[[int]]] (list of turns where each turn array contains a list of segments,
# each segment contains a list of tokens)
# [context, (sentence, DA, fact)]

"""
Input format for sentence generation:
<bos> (<sot>/<mot>/<eot>) (<DA>) [knowledge] <speaker1> S1 <end> S2 <eot> <speaker2> S3 <end> S4 <end> S5 <eot> <speaker1> R
history: [[[S1], [S2]], [[S3], [S4], [S5]]] - List[List[List[int]]]
DA: [int] : ["statement-opinion"]
Response: [int]: [5, 2, 3, 4, 1] - List[int]
Fact: [int]: [6, 2, 3, 4] - List[int]
"""
def process_split_sentence(dataset_path, split, tokenizer, index, ranker):
    vec, dialog_act = index
    path_prefix = os.path.join(dataset_path, split)
    reading_set_path = os.path.join(dataset_path, 'reading_sets', f'{split}.json')
    data = []

    # history_da = itertools.repeat(itertools.repeat(None))
    # history_knowledge = itertools.repeat(itertools.repeat(None))
    # resp_da = itertools.repeat(itertools.repeat(None))

    eot_tag = tokenizer.encode("<eot>")
    with open(path_prefix + '.json', 'r') as annotated_split_file, \
            open(reading_set_path, 'r') as reading_set_file:
        annotated_data = json.load(annotated_split_file)
        reading_set = json.load(reading_set_file)
        for conv_id, conv_data in tqdm(annotated_data.items()):
            convo_history_segments = []
            agent_knowledge, agent_mapping = prepare_reading_set_for_conversation(conv_id, reading_set)
            turn_counter = 0
            for turn in conv_data["content"]:
                turn_history = []
                da_index = 0
                for segment in turn["segments"]:
                    sentence = segment["text"]
                    current_segment_data = prepare_sentence_knowledge_data(agent_mapping, conv_id, dialog_act, tokenizer,
                                                                           turn, sentence, ranker, da_index)

                    if len(convo_history_segments) == turn_counter:
                        convo_history_segments.append(turn_history)
                    else:
                        convo_history_segments[turn_counter] = turn_history
                    convo_history_segments_copy = deepcopy(convo_history_segments)
                    context = (convo_history_segments_copy, None, None)

                    data.append((context, current_segment_data))
                    turn_history.append(current_segment_data[0])
                    da_index += 1
                turn_counter += 1
    return data


def prepare_sentence_knowledge_data(agent_mapping, conv_id, dialog_act, tokenizer, turn, sentence, ranker, da_index):
    knowledge_sentence = ranker.get_top_fact(clean(sentence), conv_id, threshold=True)
    original_knowledge_sentence = agent_mapping[turn["agent"]].get(knowledge_sentence, "")
    return tokenizer.encode(sentence), None, tokenizer.encode(original_knowledge_sentence)



# TODO : Refactor Knowledge Selection for tf-idf into ranker class - Rishi
def prepare_turn_data(agent_mapping, available_knowledge, conv_id, dialog_act, knowledge_policy,
                      response, tokenizer, turn, vec, sentiment=None, ranker=None):

    knowledge_sentence = ""
    for segment in turn["segments"]:
        sentence = segment["text"]

        if knowledge_policy == "none":
            # Always return an empty sentence
            break
        if knowledge_policy == "tf_idf":
            # With regards to knowledge selection, this is a highly approximate heuristic.
            # Both Gopalakrishnan et al. 2019 and Hedayatnia et al. 2020
            # acknowledge they don't have anything better for this issue
            text_tfidf = vec.transform([clean(sentence)])
            """
            In this section, we find the knowledge sentence that is closest
            to the ground truth response expected from the model.
            This is so that the model learns to appropriately condition on
            the knowledge
            """
            knowledge_tfidf = vec.transform(available_knowledge)

            similarities = linear_kernel(knowledge_tfidf, text_tfidf).flatten()
            closest_knowledge_index = similarities.argsort()[-1]

            if similarities[closest_knowledge_index] > 0.3:
                knowledge_sentence = available_knowledge[closest_knowledge_index]
                break
        else:
            knowledge_sentence = ranker.get_top_fact(clean(sentence), conv_id, threshold=True)
            if knowledge_sentence != "":
                break
    else:
        if knowledge_policy == "tf_idf":
            text_tfidf = vec.transform([clean(response)])
            knowledge_tfidf = vec.transform(available_knowledge)
            similarities = linear_kernel(knowledge_tfidf, text_tfidf).flatten()
            closest_knowledge_index = similarities.argsort()[-1]

            knowledge_sentence = available_knowledge[closest_knowledge_index] \
                if similarities[closest_knowledge_index] > 0.3 else ""
    original_knowledge_sentence = agent_mapping[turn["agent"]].get(knowledge_sentence, "")
    if sentiment:
        current_turn_data = (
        tokenizer.encode(response), turn["sentiment_vader"], tokenizer.encode(original_knowledge_sentence))
    else:
        current_turn_data = (
        tokenizer.encode(response), turn[dialog_act], tokenizer.encode(original_knowledge_sentence))
    return current_turn_data


def get_ranker_retriever(knowledge_policy, vec):
    if knowledge_policy == "bert" or knowledge_policy == "bert_sentence":
        return BertRankerRetriever(vec)
    elif knowledge_policy == "infersent":
        return InfersentRankerRetriever(vec)
    elif knowledge_policy == "embeddings":
        return EmbRankerRetriever(vec)
    else:
        return None

def prepare_reading_set_for_conversation(conv_id, reading_set):
    conv_reading_set = reading_set[conv_id]
    fact_mapping_1 = extract_fact_set_mapped(conv_reading_set["agent_1"])
    fact_mapping_2 = extract_fact_set_mapped(conv_reading_set["agent_2"])
    fact_set_1 = set(fact_mapping_1.keys())
    fact_set_2 = set(fact_mapping_2.keys())
    article_data = conv_reading_set["article"]
    article_indices = ['AS1', 'AS2', 'AS3', 'AS4']
    common_knowledge_mapping = dict()
    if "AS1" in article_data:
        for idx in article_indices:
            sentence = article_data[idx]
            if len(word_tokenize(sentence)) < 5:
                continue

            cleaned_sentence = clean(sentence)
            common_knowledge_mapping[cleaned_sentence] = sentence
    common_knowledge_set = set(common_knowledge_mapping.keys())
    fact_set_1.update(common_knowledge_set)
    fact_set_2.update(common_knowledge_set)
    fact_mapping_1.update(common_knowledge_mapping)
    fact_mapping_2.update(common_knowledge_mapping)
    agent_knowledge = {
        "agent_1": list(fact_set_1),
        "agent_2": list(fact_set_2)
    }
    agent_mapping = {
        "agent_1": fact_mapping_1,
        "agent_2": fact_mapping_2
    }
    return agent_knowledge, agent_mapping


def load_infersent_vecs(knowledge_index_path):
        splits = ['train', 'valid_freq', 'test_freq', 'test_rare', 'valid_rare']
        vecs = {}
        for split in splits:
            data_path = os.path.join(knowledge_index_path, f'tc_knowledge_index_facebook_{split}.pkl')
            with open(data_path, 'rb') as knowledge_index_file:
                index_data = pickle.load(knowledge_index_file)
                vecs.update(index_data['knowledge_vecs'])
        return vecs

def load_knowledge_vecs(knowledge_policy, knowledge_index_path):
    if knowledge_policy == "infersent":
        vec = load_infersent_vecs(knowledge_index_path)
    else:
        with open(knowledge_index_path, 'rb') as knowledge_index_file:
            index_data = pickle.load(knowledge_index_file)
        vec = index_data["vectorizer"] if knowledge_policy == "tf_idf" else index_data

    return vec

def augmented_tc_dataset(tokenizer, dataset_path, dataset_cache, knowledge_index_path, dialog_act, knowledge_policy):
    sentiment_flag = dialog_act == "sentiment"
    dataset_cache = dataset_cache + '_augmented_' + type(tokenizer).__name__

    if dataset_cache and os.path.isfile(dataset_cache):
        logger.info("Load tokenized dataset from cache at %s", dataset_cache)
        dataset = torch.load(dataset_cache)
    else:
        vec = load_knowledge_vecs(knowledge_policy, knowledge_index_path)
        ranker = get_ranker_retriever(knowledge_policy, vec)

        logger.info("Loading dataset from %s", dataset_path)

        splits = ['train', 'valid_freq', 'test_freq', 'test_rare', 'valid_rare']

        dataset = {}
        for split in splits:
            if knowledge_policy == "bert_sentence":
                dataset[split] = process_split_sentence(dataset_path, split, tokenizer, (vec, dialog_act), ranker)
            else:
                dataset[split] = process_split_turn(dataset_path, split, tokenizer, (vec, dialog_act), knowledge_policy,
                                                    sentiment=sentiment_flag, ranker=ranker)
            logger.info("Processed split %s", split)
        torch.save(dataset, dataset_cache)

    return dataset


class GlobalStepCounter(object):
    def __init__(self):
        self.num_steps = 0

    def get(self):
        return self.num_steps

    def step(self):
        self.num_steps += 1


def generate_references_for_split(dataset_path, dataset_cache, split, output_path):
    path_prefix = os.path.join(dataset_path, split)
    responses = []
    with open(path_prefix + '.json', 'r') as annotated_split_file:
        annotated_data = json.load(annotated_split_file)
        for conv_id, conv_data in tqdm(annotated_data.items()):
            for turn in conv_data["content"]:
                text = turn["message"]
                if text:
                    responses.append(text.strip() + "\n")

    with open(output_path, 'w') as references_file:
        references_file.writelines(responses[:-1])


def make_path(path):
    """
    Based off: https://stackoverflow.com/a/600612
    Recursively make directories for the given path.
    Replicates "mkdir -p" functionality

    :param path:
    :return:
    """
    os.makedirs(path, exist_ok=True)


if __name__ == '__main__':
    pass
    # generate_references_for_split('tc_processed', None, 'valid_freq', 'tc_processed/valid_freq.tgt')
    # tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
    # dataset_path = 'processed_output'
    # dataset_cache = './dataset_cache'
    #
    #
    # get_dataset(tokenizer, dataset_path, dataset_cache)
