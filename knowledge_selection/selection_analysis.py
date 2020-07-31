import argparse
import pandas as pd
import numpy as np

from sklearn.metrics import roc_curve

import matplotlib.pyplot as plt

def analyze_knowledge_selection_data(args):
    knowledge_df = pd.read_csv(args.data_file)

    explicit_knowledge_data = knowledge_df.dropna(subset=['uses_knowledge_explicitly'])
    filtered_knowledge_data = knowledge_df.dropna(subset=['top_knowledge_relevant'])

    relevance = filtered_knowledge_data['top_knowledge_relevant']

    knowledge_usage = explicit_knowledge_data['uses_knowledge_explicitly']

    print("Number of utterances using knowledge explicitly: ", knowledge_usage.sum() / len( knowledge_usage))
    explicit_knowledge_utterances = explicit_knowledge_data[explicit_knowledge_data['uses_knowledge_explicitly'] == 1.]
    explicit_knowledge_relevant = explicit_knowledge_utterances['top_knowledge_relevant']

    print("Unconditional relevance without threshold, accuracy: ", relevance.sum() / len(relevance))
    print("Relevance where knowledge usage is explicit", explicit_knowledge_relevant.sum() / len(explicit_knowledge_relevant))
    score = filtered_knowledge_data['knowledge_1_similarity']
    print("Number of utterances considered for relevance", len(filtered_knowledge_data))
    print("Number of unique conversations", len(filtered_knowledge_data["conversation_id"].unique()))

    fpr, tpr, thresholds = roc_curve(relevance, score)
    lw = 2
    plt.figure()
    plt.plot(fpr, tpr, lw=lw, color='darkorange')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0., 1.])
    plt.ylim([0., 1.])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')

    plt.title('ROC curve for knowledge selection threshold')
    threshold_df = pd.DataFrame({'true_positive_rate': tpr, 'false_positive_rate': fpr, 'threshold': thresholds})
    threshold_df.to_csv('bert_threshold.csv',index=False)

    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file',
                        default='test_freq_bert - test_freq_bert.csv',
                        help='File that contains knowledge data to analyze')

    args = parser.parse_args()

    analyze_knowledge_selection_data(args)