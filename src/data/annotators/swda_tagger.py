import dill
import nltk
import pandas as pd
import torch
import torch.nn as nn

from annotators.bcrf import BiLSTM_CRF
from annotators.base_annotator import AnnotatorBase

# A hardcoded constant that was taken from the BiLSTM-CRF training project
from tqdm import tqdm

PAD = "_PAD_"
UNK = "_UNK_"

class SwDATagger(AnnotatorBase):

    def _build_embeddings(self, opt, field, embedding_size):
        vocab_size = len(field.vocab)
        pad_idx = field.vocab.stoi[PAD]

        embeddings = nn.Embedding(vocab_size, embedding_size, padding_idx=pad_idx)

        return embeddings

    def _build_model(self, opt, fields, embeddings):
        embedding_size = embeddings.weight.size(1)

        num_tags = len(fields["labels"].vocab)

        model = BiLSTM_CRF(embeddings, embedding_size, opt.rnn_size, num_tags, num_layers=opt.layers, dropout=opt.dropout)

        return model

    def _setup_model(self):
        model_data = torch.load(self._model_checkpoint, pickle_module=dill)

        fields = model_data["fields"]
        opt = model_data["opt"]
        embedding_size = opt.word_vec_size

        embeddings = self._build_embeddings(opt, fields["conversation"], embedding_size)

        model = self._build_model(opt, fields, embeddings)

        model.load_state_dict(model_data["model"])
        model.eval()

        return model, fields

    def __init__(self, model_checkpoint):
        self._model_checkpoint = model_checkpoint

        model, fields = self._setup_model()

        self.model = model
        self.fields = fields

    def annotate_df(self, messages_dataframe):
        tqdm.pandas()

        # Tokenize the segments and put empty tokens wherever we have a 0-length segment
        print("Tokenizing...")
        tokenized_segments = messages_dataframe['segments']\
            .progress_apply(lambda segments: [nltk.word_tokenize(segment.lower()) for segment in segments])\
            .apply(lambda tok_segments: [segment_tokens if len(segment_tokens) > 1 else [UNK] for segment_tokens in tok_segments])


        messages_dataframe['segment_tokens'] = tokenized_segments

        messages_dataframe['segment_to_turn_idx'] = messages_dataframe.apply(
            lambda row: [row['turn_index'] for _ in row['segments']], axis=1)

        grouped_conversation_messages = messages_dataframe.groupby(["conversation_id"])[['segment_tokens', 'segment_to_turn_idx']].agg(list)

        grouped_conversation_messages['flattened_segment_to_turn_idx_list'] = grouped_conversation_messages['segment_to_turn_idx'].apply(lambda turns: [segment_turn_idx for turn in turns for segment_turn_idx in turn])
        grouped_conversation_messages['flattened_segment_token_list'] = grouped_conversation_messages['segment_tokens'].apply(lambda turns: [segment_tokens for turn in turns for segment_tokens in turn])

        model_conversation_input = grouped_conversation_messages['flattened_segment_token_list'].apply(lambda segments: self.fields["conversation"].process([segments]))

        print("Performing predictions...")
        grouped_conversation_messages['segment_predictions'] = model_conversation_input.progress_apply(lambda x: self.model(x[0], x[2])[0][0]).apply(lambda preds: [self.fields["labels"].vocab.itos[pred] for pred in preds])

        exploded_df = grouped_conversation_messages[['flattened_segment_to_turn_idx_list', 'segment_predictions']].apply(pd.Series.explode).rename(columns={"flattened_segment_to_turn_idx_list": "turn_index"})  # Flatten it completely

        # Re-group by conversation_id and turn, just like in the original_df
        turn_wise_predictions = exploded_df.groupby(["conversation_id", "turn_index"]).agg(list)


        merged_dataframe = messages_dataframe.merge(turn_wise_predictions, on=["conversation_id", "turn_index"])

        return merged_dataframe