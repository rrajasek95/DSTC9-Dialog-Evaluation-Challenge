import itertools
import json
import logging
import os
import pickle

import torch
from nltk import word_tokenize
from sklearn.metrics.pairwise import linear_kernel
from tqdm import tqdm
from glove.glove_utils import get_max_cosine_similarity, get_max_cosine_similarity_infersent
from encoder.fb_models import InferSent
from knowledge_index import clean
from sentence_transformers import SentenceTransformer
from spacy.lang.en import English

CONFIG_NAME = 'config.json'

logger = logging.getLogger(__file__)


def transform_da(da_str):
    return " ".join([f"<{da}>" for da in da_str.split(" ")])

def segment_src(src):
    nlp = English()
    sentencizer = nlp.create_pipe("sentencizer")
    nlp.add_pipe(sentencizer)
    output = []
    for each in src:
        current_context = []
        for turn in each:
            doc = nlp(turn)
            current_context.append([each.text for each in doc.sents])
        output.append(current_context)
    return output

def segment_tgt(tgt):
    nlp = English()
    sentencizer = nlp.create_pipe("sentencizer")
    nlp.add_pipe(sentencizer)
    output = []
    for turn in tgt:
        doc = nlp(turn)
        output.append([each.text for each in doc.sents])
    return output

def load_data(dataset_path, split, training_configuration, generation_config):

    path_prefix = os.path.join(dataset_path, split)

    # Splitting history into multiple sentences for ease of further processing
    src = [l.strip().split("_eos")[:-1] for l in open(path_prefix + '.src').readlines()]
    tgt = [l.strip().replace("_go", "").replace("_eos", "") for l in open(path_prefix + '.tgt').readlines()]
    fct = [l.strip() for l in open(path_prefix + '.fct').readlines()]
    segmented_sent = segment_src(src)
    segmented_tgt = segment_tgt(tgt)
    if generation_config != "sentence":
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
    elif generation_config == "sentence":
        if training_configuration == "baseline":
            history_da = itertools.repeat(itertools.repeat(None))
            history_knowledge = itertools.repeat(itertools.repeat(None))
            resp_da = itertools.repeat(None)
            new_src = []
            new_tgt = []
            new_fct = []
            for i in range(len(src)):
                for j in range(len(segmented_tgt[i])):
                    # target [j] is not the beginning of the turn
                    if j != 0:
                        new_src.append(segmented_sent[i] + [segmented_tgt[i][:j]])
                    # the target is the beginning of the turn
                    # the appended history will have a <eot> token to distinguish the turn boundary
                    else:
                        new_src.append(segmented_sent[i] + ["<eot>"])
                    new_tgt.append(segmented_tgt[i][j])
                    new_fct.append(fct[i])
            context = [zip(s, h, k) for (s, h, k) in zip(new_src, history_da, history_knowledge)]
            return list(zip(context, zip(new_tgt, resp_da, new_fct)))


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


def process_split(dataset_path, split, tokenizer, index, knowledge_policy, sentiment=False):
    infersent = None
    bert_model = None
    if knowledge_policy == "infersent":
        infersent = load_infersent_model()
    elif knowledge_policy == "bert":
        bert_model = SentenceTransformer('bert-base-nli-stsb-mean-tokens')
    vec, dialog_act = index
    path_prefix = os.path.join(dataset_path, split)
    reading_set_path = os.path.join(dataset_path, 'reading_sets', f'{split}.json')
    data = []
    with open(path_prefix + '_full_anno.json', 'r') as annotated_split_file, \
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
                                                      tokenizer, turn, vec, bert_model, infersent, sentiment)
                data.append((context, current_turn_data))
                context = context + [current_turn_data]

    return data


def prepare_turn_data(agent_mapping, available_knowledge, conv_id, dialog_act, knowledge_policy,
                      response, tokenizer, turn, vec, bert_model=None, infersent=None, sentiment=None):
    knowledge_sentence = ""
    for segment in turn["segments"]:
        sentence = segment["text"]
        # With regards to knowledge selection, this is a highly approximate heuristic.
        # Both Gopalakrishnan et al. 2019 and Hedayatnia et al. 2020
        # acknowledge they don't have anything better for this issue

        if knowledge_policy == "tf_idf":
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
        elif knowledge_policy == "embeddings":
            knowledge_sentence = emb_knowledge_selection(conv_id, sentence, vec)
            break
        elif knowledge_policy == "bert":
            knowledge_sentence = bert_knowledge_selection(conv_id, sentence, vec, bert_model)
            break
        else:
            knowledge_sentence = infersent_knowledge_selection(conv_id, sentence, vec, infersent)
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


def load_infersent_model():
    V = 2
    MODEL_PATH = 'encoder/infersent%s.pkl' % V
    params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                    'pool_type': 'max', 'dpout_model': 0.0, 'version': V}
    infersent = InferSent(params_model)
    infersent.load_state_dict(torch.load(MODEL_PATH))
    W2V_PATH = 'fastText/crawl-300d-2M.vec'
    infersent.set_w2v_path(W2V_PATH)
    infersent.build_vocab_k_words(K=100000)
    return infersent


def emb_knowledge_selection(conv_id, sentence, vec):
    knowledge = vec["knowledge_vecs"][conv_id]
    fact, sim = get_max_cosine_similarity(clean(sentence), knowledge, vec["emb_matrix"],
                                          vec["tokenizer"])
    if sim > 0.7:
        knowledge_sentence = fact
    else:
        knowledge_sentence = ""
    return knowledge_sentence


def infersent_knowledge_selection(conv_id, sentence, vec, infersent):
    knowledge = vec[conv_id]
    fact, sim = get_max_cosine_similarity_infersent(clean(sentence), knowledge, infersent)
    if sim > 0.6:
        knowledge_sentence = fact
    else:
        knowledge_sentence = ""
    return knowledge_sentence


def bert_knowledge_selection(conv_id, sentence, vec, bert):
    knowledge = vec["knowledge_vecs"][conv_id]
    fact, sim = get_max_cosine_similarity_infersent(clean(sentence), knowledge, bert, knowledge_policy="bert")
    if sim > 0.35:
        knowledge_sentence = fact
    else:
        knowledge_sentence = ""
    return knowledge_sentence

def load_infersent_vecs(knowledge_index_path):
    splits = ['train', 'valid_freq', 'test_freq', 'test_rare', 'valid_rare']
    vecs = {}
    for split in splits:
        data_path = os.path.join(knowledge_index_path, f'tc_knowledge_index_facebook_{split}.pkl')
        with open(data_path, 'rb') as knowledge_index_file:
            index_data = pickle.load(knowledge_index_file)
            vecs.update(index_data['knowledge_vecs'])
    return vecs


def augmented_tc_dataset(tokenizer, dataset_path, dataset_cache, knowledge_index_path, dialog_act, knowledge_policy):
    sentiment_flag = dialog_act == "sentiment"
    dataset_cache = dataset_cache + '_augmented_' + type(tokenizer).__name__
    if knowledge_policy == "infersent":
        vec = load_infersent_vecs(knowledge_index_path)
    else:
        with open(knowledge_index_path, 'rb') as knowledge_index_file:
            index_data = pickle.load(knowledge_index_file)
        if knowledge_policy == "tf_idf":
            vec = index_data["vectorizer"]
        else:
            vec = index_data
    if dataset_cache and os.path.isfile(dataset_cache):
        logger.info("Load tokenized dataset from cache at %s", dataset_cache)
        dataset = torch.load(dataset_cache)
    else:
        logger.info("Loading dataset from %s", dataset_path)

        splits = ['train', 'valid_freq', 'test_freq', 'test_rare', 'valid_rare']

        dataset = {}
        for split in splits:
            dataset[split] = process_split(dataset_path, split, tokenizer, (vec, dialog_act), knowledge_policy,
                                           sentiment=sentiment_flag)
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
    with open(path_prefix + '_full_anno.json', 'r') as annotated_split_file:
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
