import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer


class VaderSentimentTagger(object):
    """
    Wrapper class that performs NLTK Vader Sentiment Analysis on a given text
    """

    def __init__(self):
        nltk.download('vader_lexicon')
        self.tagger = SentimentIntensityAnalyzer()

    def extract_sentiment(self, text, arg_max=True):
        sentiment_dict = self.tagger.polarity_scores(text)

        if arg_max:
            max_label_val = sentiment_dict["neu"]
            sentiment = "NEU"
            if sentiment_dict["pos"] >= max_label_val:
                sentiment = "POS"
                max_label_val = sentiment_dict["pos"]
            if sentiment_dict["neg"] >= max_label_val:
                sentiment = "NEG"
        else:
            # decide sentiment as positive, negative and neutral
            if sentiment_dict['compound'] >= 0.05:
                sentiment = "POS"

            elif sentiment_dict['compound'] <= - 0.05:
                sentiment = "NEG"

            else:
                sentiment = "NEU"
        return {"label": sentiment}
