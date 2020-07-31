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
        file_path = os.path.join(data_path, split + '.src')

        with open(file_path, 'r') as split_file:
            examples = [line.strip().split('_eos')[:-1] for line in split_file]

        convs = []
        prev_example = []

        for example in tqdm(examples):
            if len(prev_example) > len(example):
                convs.append(prev_example)
                prev_example = []

            last_turn = example[-1]

            turn_info = {}

            doc = nlp(last_turn)

            turn_info["segments"] = [{"text": sent.text} for sent in doc.sents]
            prev_example.append(turn_info)
        convs.append(prev_example)

        for conv in tqdm(convs):
            tagger.tag_tc_conversation(conv, lower=True)

        # Prepare for output

        lines = []
        for conv in convs:
            turn_das = []
            for turn in conv:
                da_dicts = turn["swbd_da_v3"]
                das = [da["label"] for da in da_dicts]

                turn_das.append(" ".join(das))

                lines.append(" _eos ".join(turn_das) + "\n")

        with open(os.path.join(data_path, f'{split}.src.swbd3.da'), 'w') as da_file:
            da_file.writelines(lines)

if __name__ == '__main__':
    data_path = 'processed_output/'
    tagger_config = [
        "-load_model", "DA_Classifier/models/m6_acc80.04_loss0.57_e4.pt",
        "-cuda",
    ]
    tagger = DATagger(tagger_config)
    swbd_v3_tag_dstc9(tagger, data_path)