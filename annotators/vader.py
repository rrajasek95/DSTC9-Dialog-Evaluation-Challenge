import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer


class VaderSentimentTagger(object):
    """
    Wrapper class that performs NLTK Vader Sentiment Analysis on a given text
    """

    def __init__(self):
        nltk.download('vader_lexicon')
        self.tagger = SentimentIntensityAnalyzer()

    def extract_sentiment(self, text):
        sentiment_dict = self.tagger.polarity_scores(text)

        # decide sentiment as positive, negative and neutral
        if sentiment_dict['compound'] >= 0.05:
            sentiment = "POS"

        elif sentiment_dict['compound'] <= - 0.05:
            sentiment = "NEG"

        else:
            sentiment = "NEU"
        return {"label": sentiment}
