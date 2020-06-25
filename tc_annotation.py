import argparse
import json
import os

import spacy
from flair.data import Sentence
from flair.models import SequenceTagger

from tqdm.auto import tqdm

from annotators.spotlight import SpotlightTagger
from annotators.vader import VaderSentimentTagger

"""
Script to perform annotation of Topical Chats data.

Performs the following kinds of annotations:
1. Sentence (sub-turn) level Named Entity recognition (NER)
2. Dialog act annotation (WIP; Need to set up the DA tagger module)
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


def perform_vader_annotation(args):
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
        with open(os.path.join(data_dir, split + '_anno_switchboard.json'), 'r') as data_file:
            split_data = json.load(data_file)

        annotated_split = vader_annotate(tagger, split_data)

        with open(os.path.join(data_dir, split + '_anno_vader.json'), 'w') as annotated_file:
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./',
                        help='Base directory for the data')

    args = parser.parse_args()
    perform_vader_annotation(args)

    # perform_spotlight_anno(args)
    # try:
    #     perform_flair_enhanced_anno(args)
    # except:
    #     # Lazy hacky way to perform flair annotation on existing data
    #     annotate_fresh_tc_data(args)
    #     perform_flair_enhanced_anno(args)
