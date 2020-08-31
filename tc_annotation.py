import argparse
import json
import os
import pickle
import string

import more_itertools
import spacy
import torch
from flair.data import Sentence
from flair.models import SequenceTagger
from nltk import word_tokenize
from tqdm.auto import tqdm
import neuralcoref
from spacy.tokenizer import Tokenizer


from annotators.spotlight import SpotlightTagger
from annotators.vader import VaderSentimentTagger

from taggers.models import InferSentClassifier

"""
Script to perform annotation of Topical Chats data.

Performs the following kinds of annotations:
1. Sentence (sub-turn) level Named Entity recognition (NER)
2. Dialog act annotation
"""

def clean(s):
    return ''.join([c if c not in string.punctuation else ' ' for c in s.lower()])

def flair_annotate(tagger, split_data):
    for conv_id, dialog_data in tqdm(split_data.items()):

        for turn in dialog_data["content"]:

            message = turn["message"]

            sentence = Sentence(message)

            tagger.predict(sentence)

            flair_entities = []

            for entity in sentence.get_spans('ner'):
                flair_entities.append({
                    "surface": entity.to_original_text(),
                    "start_pos": entity.start_pos,
                    "end_pos": entity.end_pos,
                    "labels": [label.to_dict() for label in entity.labels]
                })
            if flair_entities:
                turn["flair_entities"] = flair_entities

    return split_data


def vader_annotate(tagger, split_data):
    for conv_id, dialog_data in tqdm(split_data.items()):
        for turn in dialog_data["content"]:
            sentiment_segments = []
            for segment in turn["segments"]:
                sentiment = tagger.extract_sentiment(segment["text"])
                sentiment_segments.append(sentiment)
            turn["sentiment_vader"] = sentiment_segments
    return split_data


def vader_annotate_turn(tagger, split_data):
    for conv_id, dialog_data in tqdm(split_data.items()):
        for turn in dialog_data["content"]:
            turn["sentiment_vader_turn"] = tagger.extract_sentiment(turn["message"])
    return split_data


def perform_vader_annotation(args, turn_anno=True):
    data_dir = os.path.join(
        args.data_dir,
        'tc_processed'
    )

    splits = [
        'train',
        'valid_freq',
        'valid_rare',
        'test_freq',
        'test_rare'
    ]

    tagger = VaderSentimentTagger()

    for split in splits:
        with open(os.path.join(data_dir, 'new_swbd_training_data', split + '_full_anno.json'), 'r') as data_file:
            split_data = json.load(data_file)

        if turn_anno:
            annotated_split = vader_annotate_turn(tagger, split_data)
        else:
            annotated_split = vader_annotate(tagger, split_data)
        with open(os.path.join(data_dir, "new_vader_anno_turn", split + '_anno_vader_arg_max_turn.json'), 'w') as annotated_file:
            json.dump(annotated_split, annotated_file)


def load_athena_tagger(args):
    with open('taggers/checkpoints/infersent_config.pkl', 'rb') as infersent_config:
        cfg = pickle.load(infersent_config)

    state_dict = torch.load('taggers/checkpoints/infersent_clf_8.pt')

    MODEL_PATH = 'taggers/encoder/infersent2.pkl'
    W2V_PATH = 'taggers/fastText/crawl-300d-2M.vec'

    classifier = InferSentClassifier(len(cfg["vocab"]), MODEL_PATH, W2V_PATH, cfg["params"], device=args.device)
    classifier.load_state_dict(state_dict)
    classifier.to(args.device)
    return classifier, cfg["vocab"]

def athena_annotate(tagger, split_data):
    classifier, vocab = tagger
    for conv_id, dialog_data in tqdm(split_data.items()):
        for turn in dialog_data["content"]:
            dacts = []
            for segment in turn["segments"]:
                dact = vocab.itos[classifier.predict([segment["text"]])]
                dacts.append(dact)
            turn["athena_das"] = dacts
    return split_data

def perform_athena_da_annotation(args):
    data_dir = os.path.join(
        args.data_dir,
        'tc_processed'
    )

    splits = [
        'train',
        'valid_freq',
        'valid_rare',
        'test_freq',
        'test_rare'
    ]

    tagger = load_athena_tagger(args)
    for split in splits:

        with open(os.path.join(data_dir, split + '_full_anno.json'), 'r') as data_file:
            split_data = json.load(data_file)

        annotated_split = athena_annotate(tagger, split_data)

        with open(os.path.join(data_dir, split + '_anno_athena.json'), 'w') as annotated_file:
            json.dump(annotated_split, annotated_file)

def annotate_split(nlp, split_data, split):
    for conv_id, dialog_data in tqdm(split_data.items()):

        for turn in dialog_data["content"]:
            message = turn["message"]
            doc = nlp(message)

            segments = []
            for sent in doc.sents:
                segment_info = {"text": sent.text}
                segments.append(segment_info)
            entity_list = []
            for ent in doc.ents:
                entity_list.append({
                    "surface": ent.text,
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "label": ent.label_
                })
            turn["segments"] = segments

            if entity_list:
                turn["entities"] = entity_list

    with open(os.path.join('tc_processed', split + '_anno.json'), 'w') as annotated_file:
        json.dump(split_data, annotated_file)


def annotate_fresh_tc_data(args):
    nlp = spacy.load("en_core_web_lg")

    data_dir = os.path.join(
        args.data_dir,
        'alexa-prize-topical-chat-dataset',
        'conversations'
    )

    splits = [
        'train',
        'valid_freq',
        'valid_rare',
        'test_freq',
        'test_rare'
    ]

    for split in splits:
        with open(os.path.join(data_dir, split + '.json'), 'r') as data_file:
            split_data = json.load(data_file)

        annotate_split(nlp, split_data, split)


def perform_flair_enhanced_anno(args):
    data_dir = os.path.join(
        args.data_dir,
        'tc_processed'
    )

    splits = [
        'train',
        'valid_freq',
        'valid_rare',
        'test_freq',
        'test_rare'
    ]

    tagger = SequenceTagger.load('ner')

    for split in splits:
        with open(os.path.join(data_dir, split + '_anno.json'), 'r') as data_file:
            split_data = json.load(data_file)

        annotated_split = flair_annotate(tagger, split_data)

        with open(os.path.join(data_dir, split + '_anno_flair.json'), 'w') as annotated_file:
            json.dump(annotated_split, annotated_file)


def spotlight_annotate(tagger, split_data):
    for conv_id, dialog_data in tqdm(split_data.items()):
        for turn in dialog_data["content"]:

            message = turn["message"]

            spotlight_entities = tagger.get_spotlight_annotation(message, confidence=0.5)

            if spotlight_entities:
                turn["dbpedia_entities"] = spotlight_entities
            print(spotlight_entities)

    return split_data

def lengthbin_annotate(split_data):
    bins = {
        0: "S",
        1: "M",
        2: "L"
    }

    for conv_id, dialog_data in tqdm(split_data.items()):

        for turn in dialog_data["content"]:
            segments = turn["segments"]

            for segment in segments:
                text = segment["text"]
                tokens = word_tokenize(text)
                length_bin_index = len(tokens) // 10

                segment["length_bin"] = bins.get(length_bin_index, "L")
                segment["num_tokens"] = len(tokens)

    return split_data


def perform_spotlight_anno(args):
    data_dir = os.path.join(
        args.data_dir,
        'tc_processed'
    )

    splits = [
        'train',
        'valid_freq',
        'valid_rare',
        'test_freq',
        'test_rare'
    ]

    tagger = SpotlightTagger(
        ontology_json='annotators/ontology_classes.json',
        spotlight_server_url='http://localhost:2222/rest/annotate')
    for split in splits:
        with open(os.path.join(data_dir, split + '_anno.json'), 'r') as data_file:
            split_data = json.load(data_file)

        annotated_split = spotlight_annotate(tagger, split_data)

        with open(os.path.join(data_dir, split + '_anno_spotlight.json'), 'w') as annotated_file:
            json.dump(annotated_split, annotated_file)


def load_split_reading_set(data_file_path, split):
    split_reading_set_path = os.path.join(
        data_file_path,
        'reading_sets',
        'post-build',
        f'{split}.json'
    )
    with open(split_reading_set_path, 'r') as reading_set_file:
        split_reading_set = json.load(reading_set_file)

    return split_reading_set


def spotlight_annotate_knowledge(tagger, split_data):
    agents = ["agent_1", "agent_2"]

    for conv_id, dialog_data in tqdm(split_data.items()):
        for agent in agents:
            for idx, data in dialog_data[agent].items():
                fun_facts = data.get("fun_facts")
                if fun_facts:
                    facts = []
                    for fact in fun_facts:
                        fact_dict = {}
                        fact_dict["text"] = fact
                        spotlight_entities = tagger.get_spotlight_annotation(clean(fact), confidence=0.5)
                        if spotlight_entities:
                            fact_dict["dbpedia_entities"] = spotlight_entities
                        facts.append(fact_dict)
                    data["fun_facts"] = facts

                short_wiki = data.get("shortened_wiki_lead_section")
                if short_wiki:
                    short = {}
                    short["text"] = short_wiki
                    spotlight_entities = tagger.get_spotlight_annotation(clean(short_wiki), confidence=0.5)
                    if spotlight_entities:
                        short["dbpedia_entities"] = spotlight_entities
                    data["shortened_wiki_lead_section"] = short

                summarized_wiki = data.get("summarized_wiki_lead_section")
                if summarized_wiki:
                    summ = {}
                    summ["text"] = summarized_wiki
                    spotlight_entities = tagger.get_spotlight_annotation(clean(summarized_wiki), confidence=0.5)
                    if spotlight_entities:
                        summ["dbpedia_entities"] = spotlight_entities
                    data["summarized_wiki_lead_section"] = summ

            article_data = dialog_data["article"]
            article_indices = ['AS1', 'AS2', 'AS3', 'AS4']

            # Article information
            if "AS1" in article_data:
                for idx in article_indices:
                    sentence = article_data[idx]
                    if len(word_tokenize(sentence)) < 5:
                        continue
                    art_dict = {}
                    art_dict["text"] = sentence
                    spotlight_entities = tagger.get_spotlight_annotation(clean(summarized_wiki), confidence=0.5)
                    if spotlight_entities:
                        art_dict["dbpedia_entities"] = spotlight_entities
                    article_data[idx] = art_dict



def perform_spotlight_anno_knowledge(args):
    splits = [
        # 'train',
        # 'valid_freq',
        # 'valid_rare',
        'test_freq',
        # 'test_rare'
    ]
    data_file_path = "alexa-prize-topical-chat-dataset"


    tagger = SpotlightTagger(
        ontology_json='annotators/ontology_classes.json',
        spotlight_server_url='http://localhost:2222/rest/annotate')
    for split in splits:
        reading_set = {}
        reading_set.update(load_split_reading_set(data_file_path, split))

        annotated_split = spotlight_annotate_knowledge(tagger, reading_set)

        with open(os.path.join(data_file_path, split + '_spotlight.json'), 'w') as annotated_file:
            json.dump(annotated_split, annotated_file)


def perform_length_binning_anno(args):
    data_dir = os.path.join(
        args.data_dir,
        'tc_processed'
    )

    splits = [
        'train',
        'valid_freq',
        'valid_rare',
        'test_freq',
        'test_rare'
    ]

    for split in splits:
        with open(os.path.join(data_dir, split + '_anno.json'), 'r') as data_file:
            split_data = json.load(data_file)

        annotated_split = lengthbin_annotate(split_data)

        with open(os.path.join(data_dir, split + '_anno_length_bin.json'), 'w') as annotated_file:
            json.dump(annotated_split, annotated_file)


def merge_data(merged_split, merging_data, fields):
    for conversation_id, data in merged_split.items():
        merging_conv = merging_data[conversation_id]

        for (m1, m2) in zip(data["content"], merging_conv["content"]):
            # Since all the fields are at the segment level, we can merge it directly into the segments

            for field in fields:
                m1[field] = m2[field]



def merge_all_annotations(args):
    splits = [
        'train',
        'valid_freq',
        'valid_rare',
        'test_freq',
        'test_rare'
    ]

    suffix_field_map = {
        '_anno_flair_mezza_da': ['mezza_da'],
        '_anno_vader': ['sentiment_vader', 'switchboard_da'],
    }

    # Reminder about what fields have been annotated:
    # In '_anno_spotlight'
    #   we have ['entities', 'dbpedia_entities', 'flair_entities']
    # In '_anno_vader'
    #   we have ['switchboard_da', 'sentiment_vader']
    # In '_anno_flair_mezza_da'
    #   we have ['mezza_da']

    for split in splits:
        merged_split = None

        with open(os.path.join(
                args.data_dir,
                'tc_processed',
                f"{split}_anno_spotlight.json"), 'r') as spotlight_anno_file:
            merged_split = json.load(spotlight_anno_file)
        for suffix, fields in suffix_field_map.items():
            with open(os.path.join(
                    args.data_dir,
                    'tc_processed',
                    f"{split}{suffix}.json"), "r") as annotated_file:
                merging_data = json.load(annotated_file)

            merge_data(merged_split, merging_data, fields)

        with open(os.path.join(
                args.data_dir,
                'tc_processed',
                f'{split}_full_anno.json'), 'w') as merged_anno_file:
            json.dump(merged_split, merged_anno_file)


def partition_training_conversations(data_dir, num_splits=16):
    train_file = os.path.join(
        data_dir,
        'processed_output',
        'train.src'
    )

    with open(train_file, 'r') as train_data_file:

        conversations = []
        current_conversation = []
        prev_num_turns = 0
        for line in train_data_file:
            turns = line.strip().split("_eos")[:-1]

            num_turns = len(turns)

            if num_turns <= prev_num_turns:

                conversations.append(current_conversation)
                current_conversation = []

            current_conversation.append(line)
            prev_num_turns = num_turns

        chunks = more_itertools.divide(num_splits, conversations)

        for i, chunk in enumerate(chunks):
            file_path = os.path.join(
                data_dir,
                'processed_output',
                f'train_{i + 1}.src'
            )
            with open(file_path, 'w') as split_file:
                conversations = list(chunk)
                for lines in conversations:
                    split_file.writelines(lines)

def perform_coref_anno(args):
    data_dir = os.path.join(
        args.data_dir,
        'tc_processed'
    )
    nlp = spacy.load('en_core_web_lg')
    coref = neuralcoref.NeuralCoref(nlp.vocab)
    nlp.add_pipe(coref, name='neuralcoref')

    splits = [
        # 'train',
        # 'valid_freq',
        # 'valid_rare',
        'test_freq',
        # 'test_rare'
    ]

    for split in splits:
        with open(os.path.join(data_dir, split + '_anno.json'), 'r') as data_file:
            split_data = json.load(data_file)

        annotated_split = coref_anno(nlp, split_data)

        with open(os.path.join(data_dir, split + '_anno_coref_large.json'), 'w') as annotated_file:
            json.dump(annotated_split, annotated_file)


def coref_anno(nlp, split_data):
    for conv_id, dialog_data in tqdm(split_data.items()):
        messages = ""
        # end of turn span, non inclusive
        spans = []
        spanEnd = -1
        for turn in dialog_data["content"]:
            messages += turn["message"] + " "
            spanEnd += len(turn["message"]) + 1
            spans.append(spanEnd)


        messages = messages.strip()
        doc = nlp(messages)
        if doc._.has_coref:
            corefs = []
            for coref in doc._.coref_clusters:
                coref_dict = {}
                coref_dict["main"] = coref.main.string
                start_char_main = coref.main.start_char
                end_char_main = coref.main.end_char
                main_turn = find_span_turn(start_char_main, end_char_main, spans)
                # TODO: BUG!! This should not happen but some annotators didn't use punctuation
                # So coref marks entities across different turns and that's a problem and annoying
                if main_turn is None:
                    continue
                coref_dict["turn"] = main_turn

                turn_start = 0
                if main_turn != 1:
                    turn_start = spans[main_turn - 2]
                # coref_dict["span_within_turn_start"] = start_char_main - turn_start - 1
                # coref_dict["span_within_turn_end"] = end_char_main - turn_start - 1

                start_turn_span = start_char_main - turn_start - 1
                end_turn_span = end_char_main - turn_start - 1
                doc_turn = nlp(dialog_data["content"][main_turn - 1]["message"])
                start_mes_span = 0
                index = 0
                for sent in doc_turn.sents:
                    if end_turn_span < start_mes_span + sent.end_char:
                        coref_dict["segment"] = index + 1
                        if index > 0:
                            coref_dict["span_within_segment_start"] = start_turn_span - start_mes_span - 1
                            coref_dict["span_within_segment_end"] = end_turn_span - start_mes_span - 1
                        else:
                            coref_dict["span_within_segment_start"] = start_turn_span - start_mes_span
                            coref_dict["span_within_segment_end"] = end_turn_span - start_mes_span
                        break
                    start_mes_span = sent.end_char
                    index += 1

                single_refs = []
                for i in range(1, len(coref.mentions)):
                    single_dict = {}
                    single_dict["text"] = coref.mentions[i].string
                    ref_turn = find_span_turn(coref.mentions[i].start_char, coref.mentions[i].end_char, spans)
                    # TODO: BUG!! This should not happen but some annotators didn't use punctuation
                    # So coref marks entities across different turns and that's a problem and annoying
                    if ref_turn is None:
                        continue
                    single_dict["turn"] = ref_turn


                    turn_start_ref = 0
                    if ref_turn != 1:
                        turn_start_ref = spans[ref_turn - 2]
                    start_ref_turn_span = coref.mentions[i].start_char - turn_start_ref - 1
                    end_ref_turn_span = coref.mentions[i].end_char - turn_start_ref - 1
                    doc_turn_ref = nlp(dialog_data["content"][ref_turn - 1]["message"])
                    start_mes_span_ref = 0
                    index = 0
                    for sent in doc_turn_ref.sents:
                        if end_ref_turn_span < start_mes_span_ref + sent.end_char:
                            single_dict["segment"] = index + 1
                            if index > 0:
                                single_dict["span_within_segment_start"] = start_ref_turn_span - start_mes_span_ref - 1
                                single_dict["span_within_segment_end"] = end_ref_turn_span - start_mes_span_ref - 1
                            else:
                                single_dict["span_within_segment_start"] = start_ref_turn_span - start_mes_span_ref
                                single_dict["span_within_segment_end"] = end_ref_turn_span - start_mes_span_ref
                            break
                        start_mes_span_ref = sent.end_char
                        index += 1
                    single_refs.append(single_dict)
                coref_dict["references"] = single_refs
                corefs.append(coref_dict)
            split_data[conv_id]["corefs"] = corefs

    return split_data


def find_span_turn(start_char, end_char, spans_array):
    start = 0
    for j in range(len(spans_array)):
        if start_char >= start and end_char < spans_array[j]:
            return j + 1
        start = spans_array[j]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./',
                        help='Base directory for the data')
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    args = parser.parse_args()
    # perform_athena_da_annotation(args)
    # merge_all_annotations(args)
    # perform_vader_annotation(args, True)

    # perform_spotlight_anno(args)
    # perform_spotlight_anno_knowledge(args)
    perform_coref_anno(args)
    # try:
    #     perform_flair_enhanced_anno(args)
    # except:
    #     # Lazy hacky way to perform flair annotation on existing data
    #     annotate_fresh_tc_data(args)
    #     perform_flair_enhanced_anno(args)
    # perform_length_binning_anno(args)