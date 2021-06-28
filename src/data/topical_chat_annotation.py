import argparse
import os.path

import pandas as pd

from annotators.nltk_segmenter import NltkSentenceSegmenter
from annotators.swda_tagger import SwDATagger

SPLITS = (
    "train",
    "valid_freq",
    "valid_rare",
    "test_freq",
    "test_rare"
)

def annotate_topical_chat_parquet(split_file):
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--messages_path', default="data/intermediate/topical_chat_parquet/conversations")
    parser.add_argument('--swda_tagger_model_path', default="models/swda_tagger/m6_acc80.04_loss0.57_e4.pt")
    parser.add_argument('--output_path', default="data/intermediate/topical_chat_annotation/conversations")


    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    nltk_sentence_segmenter = NltkSentenceSegmenter()
    swda_annotator = SwDATagger(args.swda_tagger_model_path)


    for split in SPLITS:
        split_messages_parquet = os.path.join(args.messages_path, f"{split}.parquet")
        split_messages_dataframe = pd.read_parquet(split_messages_parquet)

        print(f"Annotating segments for split '{split}'")
        split_messages_dataframe['segments'] = nltk_sentence_segmenter.annotate_series(split_messages_dataframe['message'])

        swda_tagged_with_split_messages_dataframe = swda_annotator.annotate_df(split_messages_dataframe)


        annotated_split_data_path = os.path.join(args.output_path, f"{split}.parquet")
        swda_tagged_with_split_messages_dataframe.to_parquet(annotated_split_data_path)
