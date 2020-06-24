
"""
This script allows for the standardized evaluation metrics
such as F1, BLEU-4, ROUGE-L, METEOR, etc. to be computed
against a set of reference responses.
"""
import argparse
from collections import Counter

import nlp

class ReferenceMetric(object):
    """
    Metric that requires a reference sentence for each
    hypothesis to compute
    """
    def compute(self, hypotheses, references):
        raise NotImplementedError("Implement the compute method!")


class ReferenceFreeMetric(object):
    """
    Metric that does not require a reference sentence
    """

    def compute(self, hypotheses):
        raise NotImplementedError("Implement the compute method!")


class NLPReferenceMetric(ReferenceMetric):
    """
    Reference dependent metrics that are part of the
    Huggingface NLP library
    """
    def __init__(self, module, compute_args={}):
        self.scorer = nlp.load_metric(module)
        self.compute_args = compute_args

    def compute(self, hypotheses, references):
        return self.scorer.compute(hypotheses, references, **self.compute_args)


class BLEUMetric(NLPReferenceMetric):
    def __init__(self):
        super().__init__('bleu')

    def __repr__(self):
        return 'BLEU-4'


class RougeMetric(NLPReferenceMetric):
    def __init__(self):
        super().__init__('rouge')

    def __repr__(self):
        return 'ROUGE'

class BertScoreMetric(NLPReferenceMetric):
    def __init__(self):
        self.arg_dict = {"lang": "en"}
        super().__init__('bertscore', self.arg_dict)

    def __repr__(self):
        return f'BertScore({self.arg_dict})'


class UnigramFScoreMetric(ReferenceMetric):
    def __init__(self, beta=1):
        self.beta = beta
        self.beta_squared = beta * beta  # Fbeta uses (beta^2 as weighting factor)

    def _f_beta(self, pred, true):
        common = Counter(true) & Counter(pred)
        num_same = sum(common.values())

        if num_same == 0:
            return 0

        prec = num_same / len(pred)
        rec = num_same / len(true)

        f_beta = ((self.beta_squared + 1) * prec * rec) / (self.beta_squared * prec + rec)
        return f_beta

    def compute(self, hypotheses, references):
        return sum([self._f_beta(hyp.split(), ref.split()) for hyp, ref in zip(hypotheses, references)]) / len(references)

    def __repr__(self):
        return f'F{self.beta}-score'

class NGramDiversity(ReferenceFreeMetric):
    def __init__(self, n=1):
        self.n = n

    def _diversity(self, pred):
        n_grams = []

        for i in range(len(pred) - self.n + 1):
            n_grams.append(' '.join(pred[i:i + self.n]))

        if len(n_grams) == 0:
            return 0

        return len(set(n_grams)) / len(n_grams)

    def compute(self, hypotheses):
        return sum([self._diversity(hyp.split()) for hyp in hypotheses])

    def __repr__(self):
        return f'{self.n}-gram diversity'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Currently support only single hypotheses scoring
    parser.add_argument('--predictions_file',
                        type=str,
                        default="submissions/submissions.txt",
                        help='File containing output predictions')
    parser.add_argument('--references_file',
                        type=str,
                        default='processed_output/valid_freq.tgt',
                        help='File containing the reference responses')
    args = parser.parse_args()

    with open(args.predictions_file, 'r') as predictions_file:
        predictions = [line.strip() for line in predictions_file]

    with open(args.references_file, 'r') as references_file:
        references = [line.replace("_go", "").replace("_eos", "") for line in references_file]

    assert len(predictions) == len(references), "The number of predictions and references do not match!"

    metrics = [
        BLEUMetric(),
        RougeMetric(),
        BertScoreMetric(),
        UnigramFScoreMetric(),
        NGramDiversity(n=1),
        NGramDiversity(n=2)
    ]

    print(f"Number of examples n={len(predictions)}\n")

    for metric in metrics:
        if isinstance(metric, ReferenceFreeMetric):
            print(metric, ":")
            print(metric.compute(predictions))
        else:
            print(metric, ":")
            print(metric.compute(predictions, references))