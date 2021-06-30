
class AnnotatorBase:

    def annotate_df(self, messages_df):
        raise NotImplementedError("Method annotate_df not implemented!")

    def annotate_series(self, series):
        raise NotImplementedError("Method annotate_series not implemented!")