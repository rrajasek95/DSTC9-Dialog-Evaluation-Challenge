from flair.data import Sentence
from flair.models import SequenceTagger


class FlairNamedEntityTagger(object):
    """
    Wrapper class that performs Flair NER on a given text
    """
    def __init__(self):
        self.tagger = SequenceTagger.load('ner')

    def extract_entities(self, text):
        sentence = Sentence(text)

        self.tagger.predict(sentence)

        flair_entities = []

        for entity in sentence.get_spans('ner'):
            flair_entities.append({
                "surface": entity.to_original_text(),
                "start_pos": entity.start_pos,
                "end_pos": entity.end_pos,
                "labels": [label.to_dict() for label in entity.labels]
            })

        return flair_entities