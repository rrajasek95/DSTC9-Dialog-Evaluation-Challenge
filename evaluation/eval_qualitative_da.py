import argparse
from collections import defaultdict
import random

import spacy
from tqdm.auto import tqdm

import jiwer

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, classification_report


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

def analyze_da_realization(sentence_plan_path, realized_acts_path):
    with open(sentence_plan_path, 'r') as valid_freq_plan:
        sentence_plans = [line.strip() for line in valid_freq_plan]

    with open(realized_acts_path, 'r') as realized_acts:
        realized_acts = [line.strip() for line in realized_acts]


    print(jiwer.compute_measures(sentence_plans, realized_acts))

    all_plan_das = []
    all_real_das = []

    for plan, realized in zip(sentence_plans, realized_acts):
        plan_acts = plan.split(" ")
        real_das = realized.split(" ")

        len_diff = len(plan_acts) - len(real_das)

        if len_diff < 0:
            plan_acts = plan_acts + ["None"] * -len_diff
        elif len_diff > 0:
            real_das = real_das + ["None"] * len_diff

        all_plan_das += plan_acts
        all_real_das += real_das

    labels = list(set(all_plan_das))
    datoi = {label: i for i, label in enumerate(labels)}
    plan_da_idx = [datoi[label] for label in all_plan_das]
    real_da_idx = [datoi[label] for label in all_real_das]
    print(classification_report(plan_da_idx, real_da_idx, target_names=labels))



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
                        default='processed_output/valid_freq.tgt.da')
    args = parser.parse_args()
    # analyze_da_utterances(args)
    analyze_da_realization(args.plan_file, args.predictions_file)