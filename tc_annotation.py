import json
import os

import spacy

from tqdm.auto import tqdm

"""
Script to perform annotation of Topical Chats data.

Performs the following kinds of annotations:
1. Sentence (sub-turn) level Named Entity recognition (NER)
2. Dialog act annotation (WIP; Need to set up the DA tagger module)
"""


def annotate_split(nlp, split_data, split):

    for conv_id, dialog_data in tqdm(split_data.items()):

        for turn in dialog_data["content"]:
            message = turn["message"]
            doc = nlp(message)

            segments = []
            for sent in doc.sents:
                segment_info = {"text": sent.text}

                entity_list = []
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


if __name__ == '__main__':
    nlp = spacy.load("en_core_web_lg")

    data_dir = os.path.join(
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