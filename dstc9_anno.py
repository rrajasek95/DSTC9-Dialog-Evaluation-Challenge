import os
import sys

from DA_Classifier.DA_Tagger import DATagger

import spacy
from tqdm.auto import tqdm

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

if __name__ == '__main__':
    data_path = 'processed_output/'
    tagger_config = [
        "-load_model", "DA_Classifier/models/m6_acc80.04_loss0.57_e4.pt",
        "-cuda",
    ]
    tagger = DATagger(tagger_config)
    swbd_v3_tag_dstc9(tagger, data_path)