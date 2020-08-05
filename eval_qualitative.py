import argparse
from collections import defaultdict
import random

import spacy
from tqdm.auto import tqdm


def analyze_da_utterances(args):
    da_to_utterance_map = defaultdict(list)
    nlp = spacy.load('en_core_web_sm')

    with open(args.predictions_file, 'r') as predictions_f:
        pred_lines = [pred.strip() for pred in predictions_f]

    with open(args.plan_file, 'r') as plan_f:
        plan_lines = [plan.strip() for plan in plan_f]

    for (pred, plan) in tqdm(zip(pred_lines, plan_lines)):
        doc = nlp(pred)

        for (sent, da) in zip(doc.sents, plan.split(" ")):
            da_to_utterance_map[da].append(sent)


    for (da, sents) in da_to_utterance_map.items():
        print("DA: ", da)
        print("Examples: ")

        sample = random.sample(sents, k=min(len(sents), 20))
        for sent in sample:
            print(sent.text)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions_file',
                        type=str,
                        default="submissions/submissions.txt",
                        help='File containing output predictions')
    # parser.add_argument('--references_file',
    #                     type=str,
    #                     default='processed_output/valid_freq.tgt',
    #                     help='File containing the reference responses')
    parser.add_argument('--plan_file',
                        type=str,
                        default='processed_output/valid_freq.tgt,da')
    args = parser.parse_args()
    analyze_da_utterances(args)