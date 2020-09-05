import argparse
import json
import os
import sys

from DA_Classifier.DA_Tagger import DATagger

import spacy
from tqdm.auto import tqdm

from annotators.spotlight import SpotlightTagger


def swbd_v3_tag_dstc9(tagger, data_path):
    nlp = spacy.load('en_core_web_sm')
    splits = [
        'train',
        'valid_freq',
        'valid_rare',
        'test_freq',
        'test_rare'
    ]

    for split in splits:
        src_file_path = os.path.join(data_path, split + '.src')
        tgt_file_path = os.path.join(data_path, split + '.tgt')

        with open(src_file_path, 'r') as split_file:
            src_examples = [line.strip().split('_eos')[:-1] for line in split_file]
        with open(tgt_file_path, 'r') as tgt_file:
            tgt_responses = [line.strip().replace('_eos', '').replace('_go', '') for line in tgt_file]

        convs = []
        prev_example = []

        for i, example in tqdm(enumerate(src_examples)):
            if len(prev_example) > len(example):
                # Special handling to add the last turn of the response
                response_turn = tgt_responses[i]
                doc = nlp(response_turn)
                turn_info = {"segments": [{"text": sent.text} for sent in doc.sents]}
                prev_example.append(turn_info)

                convs.append(prev_example)
                prev_example = []

            last_turn = example[-1]


            doc = nlp(last_turn)

            turn_info = {"segments": [{"text": sent.text} for sent in doc.sents]}
            prev_example.append(turn_info)

        # This could be avoided entirely by adding an extra empty example
        # but it's better to handle it explicitly
        response_turn = tgt_responses[-1]
        doc = nlp(response_turn)
        turn_info = {"segments": [{"text": sent.text} for sent in doc.sents]}

        prev_example.append(turn_info)

        convs.append(prev_example)

        for conv in tqdm(convs):
            tagger.tag_tc_conversation(conv, lower=True)

        # Prepare for output

        lines = []
        response_lines = []
        for conv in convs:
            turn_das = []
            for turn in conv[:-1]:
                da_dicts = turn["swbd_da_v3"]
                das = [da["label"] for da in da_dicts]

                turn_das.append(" ".join(das))

                lines.append(" _eos ".join(turn_das) + "\n")

            for turn in conv[1:]:
                da_dicts = turn["swbd_da_v3"]
                das = [da["label"] for da in da_dicts]

                response_lines.append("_go " + " ".join(das) + " _eos\n")

        with open(os.path.join(data_path, f'{split}.src.swbd3.da'), 'w') as da_file:
            da_file.writelines(lines)
        with open(os.path.join(data_path, f'{split}.tgt.swbd3.da'), 'w') as tgt_da_file:
            tgt_da_file.writelines(response_lines)

def perform_spotlight_anno(tagger, data_path):
    splits = [
        'train',
        'valid_freq',
        'valid_rare',
        'test_freq',
        'test_rare'
    ]

    for split in splits:
        src_file_path = os.path.join(data_path, split + '.src')
        tgt_file_path = os.path.join(data_path, split + '.tgt')

        with open(src_file_path, 'r') as split_file:
            src_examples = [line.strip().split('_eos')[:-1] for line in split_file]
        with open(tgt_file_path, 'r') as tgt_file:
            tgt_responses = [line.strip().replace('_eos', '').replace('_go', '') for line in tgt_file]
        with open(tgt_file_path, 'r') as fct_file:
            fcts = [line.strip() for line in fct_file]
        convs_anno = []
        prev_example = []

        response_anno = []
        for i, example in tqdm(enumerate(src_examples)):
            if len(example) == 1:
                prev_example = []
                response_anno.append("_go " + json.dumps(tagger.get_spotlight_annotation(tgt_responses[i])) + "\n")
            else:
                response_anno.append("_go " + prev_example[-1] + "\n")

            anno = tagger.get_spotlight_annotation(example[-1])
            tagged_line = prev_example + [json.dumps(anno)]
            convs_anno.append(" _eos ".join(tagged_line) + "\n")
            prev_example = tagged_line

        facts_anno = []

        for i, fact in tqdm(enumerate(fcts)):
            facts_anno.append(json.dumps(tagger.get_spotlight_annotation(fact)) + "\n")

        with open(os.path.join(data_path, f'{split}.src.spotlight'), 'w') as da_file:
            da_file.writelines(convs_anno)
        with open(os.path.join(data_path, f'{split}.tgt.spotlight'), 'w') as tgt_da_file:
            tgt_da_file.writelines(response_anno)
        with open(os.path.join(data_path, f'{split}.fct.spotlight'), 'w') as tgt_da_file:
            tgt_da_file.writelines(facts_anno)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--annotation',
                        help='Type of annotation to perform Dialog Act Tagging (swbd) or NER (spotlight)',
                        type=str,
                        default='swbd',
                        choices=['swbd', 'spotlight'])
    parser.add_argument('--data_path',
                        default='processed_output')

    tagger_config = [
        "-load_model", "DA_Classifier/models/m6_acc80.04_loss0.57_e4.pt",
        "-cuda",
    ]

    args = parser.parse_args()

    if args.annotation == "swbd":
        tagger = DATagger(tagger_config)
        swbd_v3_tag_dstc9(tagger, args.data_path)
    else:
        tagger = SpotlightTagger('annotators/ontology_classes.json', 'http://localhost:2222/rest/annotate')
        perform_spotlight_anno(tagger, args.data_path)