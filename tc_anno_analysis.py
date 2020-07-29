import argparse
import json
import os
import random
from collections import defaultdict



def analyze_da_information_for_split(da_mapping):

    for da, segments in da_mapping.items():
        print("Label:", da)
        print("Examples:")
        sample_segments = random.sample(segments, min(20, len(segments)))

        for segment in sample_segments:
            print(segment['text'])
        print("\n")


def analyze_split(split_data):
    mezza_da_segment_mapping = defaultdict(list)
    switchboard_da_segment_mapping = defaultdict(list)

    for conv_id, conv_data in split_data.items():
        turns = conv_data["content"]

        for turn in turns:
            segments = turn['segments']
            mezza_das = turn['mezza_da']
            swbd_das = turn['switchboard_da']

            for (segment, mezza_da, swbd_da) in zip(segments, mezza_das, swbd_das):
                mezza_da_segment_mapping[mezza_da['da']].append(segment)
                switchboard_da_segment_mapping[swbd_da['label']].append(segment)
    print("Mezza DA:")
    analyze_da_information_for_split(da_mapping=mezza_da_segment_mapping)
    print("Switchboard DA:")
    analyze_da_information_for_split(da_mapping=switchboard_da_segment_mapping)

def analyze_split_ath(split_data):
    athena_da_segment_mapping = defaultdict(list)

    for conv_id, conv_data in split_data.items():
        turns = conv_data["content"]

        for turn in turns:
            segments = turn['segments']
            athena_das = turn['athena_das']

            for (segment, athena_da) in zip(segments, athena_das):
                athena_da_segment_mapping[athena_da].append(segment)

    print("Athena DA:")
    analyze_da_information_for_split(athena_da_segment_mapping)

def analyze_tc_data(args):

    splits = ['train' ] # , 'valid_freq', 'test_freq']

    for split in splits:
        if args.ath :
            split_file = f'{split}_anno_athena.json'
        else:
            split_file = f'{split}_full_anno.json'
        split_file_path = os.path.join(args.data_path, split_file)

        with open(split_file_path, 'r') as split_f:
            split_data = json.load(split_f)
        print("Split:", split)

        if args.ath:
            analyze_split_ath(split_data)
        else:
            analyze_split(split_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='./tc_processed')
    parser.add_argument('--ath', action='store_true')
    args = parser.parse_args()
    analyze_tc_data(args)
