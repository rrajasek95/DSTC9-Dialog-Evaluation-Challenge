import nltk
from annotators.base_annotator import AnnotatorBase

class NltkSentenceSegmenter(AnnotatorBase):
    def annotate_series(self, series):
        return series.apply(nltk.sent_tokenize)
