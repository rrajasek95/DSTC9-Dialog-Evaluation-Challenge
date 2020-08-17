import argparse
import json
import os
import pickle

import more_itertools
import spacy
import torch
from flair.data import Sentence
from flair.models import SequenceTagger
from nltk import word_tokenize
from tqdm.auto import tqdm

from annotators.spotlight import SpotlightTagger
from annotators.vader import VaderSentimentTagger

from taggers.models import InferSentClassifier

"""
Script to perform annotation of Topical Chats data.

Performs the following kinds of annotations:
1. Sentence (sub-turn) level Named Entity recognition (NER)
2. Dialog act annotation
"""


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./',
                        help='Base directory for the data')
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    args = parser.parse_args()
    # perform_athena_da_annotation(args)
    # merge_all_annotations(args)
    perform_vader_annotation(args, True)

    # perform_spotlight_anno(args)
    # try:
    #     perform_flair_enhanced_anno(args)
    # except:
    #     # Lazy hacky way to perform flair annotation on existing data
    #     annotate_fresh_tc_data(args)
    #     perform_flair_enhanced_anno(args)
    # perform_length_binning_anno(args)