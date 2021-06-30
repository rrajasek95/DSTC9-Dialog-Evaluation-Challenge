import torch
import numpy as np
from torch import nn
try:
    from .Crf import CRF
except ModuleNotFoundError as e:
    from DA_Classifier.Crf import CRF


from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

class BiLSTM_CRF(nn.Module):

    def __init__(self, embeddings, embedding_size, hidden_size, num_tags, num_layers=1,
                 rnn_dropout=0., dropout=0.):

        super(BiLSTM_CRF, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        #self.dropout = nn.Dropout(dropout)

        self.embedding = embeddings
        # Utterance level encoder
        self.lstm1 = nn.LSTM(
            self.embedding_size,
            self.hidden_size//2,
            batch_first=False,
            bidirectional=True,
            num_layers=num_layers,
            dropout=rnn_dropout,
        )
        # Turn level encoder
        self.lstm2 = nn.LSTM(
            self.hidden_size,
            self.hidden_size//2,
            batch_first=True,
            bidirectional=True,
            num_layers=num_layers,
            dropout=rnn_dropout,
        )

        #self.lstm1 = nn.LSTM(self.embedding_size, self.hidden_size//2, batch_first=False, bidirectional=True)
        # Turn level encoder
        #self.lstm2 = nn.LSTM(self.hidden_size, self.hidden_size//2, batch_first=True, bidirectional=True)



        self.linear = nn.Linear(self.hidden_size, num_tags)

        #self.linear = nn.Sequential(
        #    nn.Dropout(dropout),
        #    nn.Linear(self.hidden_size, self.hidden_size),
        #    nn.ReLU(),
        #    nn.Linear(self.hidden_size, num_tags)
        #)

        self.crf = CRF(num_tags, batch_first=True)

    def rnn_params(self):
        params = dict(self.lstm1.parameters())
        params.update(self.lstm2.parameters())
        return params

    def crf_score(self, crf_input, labels):
        """

        :param crf_input:
        :param labels:
        :return:
        """

        # Experiment with different ways to calculate loss
        score = self.crf(crf_input, labels, reduction="mean")
        # Score is log-likelihood, so negate
        score = score * (-1)
        return score

    def forward(self, x, lengths):
        """

        :param x:
        :param lengths:
        :return:
        """
        # input shape (batch_size, batch_conv_len, batch_max_utt_len), lengths = [[]]
        batch_size = x.shape[0]

        # output shape (batch_size, batch_conv_len, batch_max_utt_len, embedding_dim)
        emb = self.embedding(x)

        #emb = self.dropout(emb)

        # output shape = (batch_size * batch_conv_len, batch_max_utt_len, embedding_dim)
        emb = emb.view(-1, emb.shape[2], emb.shape[3])

        # output shape = (batch_size * batch_conv_len)
        lengths_reshaped = lengths.view(-1)
        packed_utterance = pack(emb, lengths_reshaped, batch_first=True, enforce_sorted=False)

        # output shape = (batch_size * batch_conv_len)
        _rnn_out, (ht, ct) = self.lstm1(packed_utterance)

        # output shape (batch_size * batch_conv_len, hid_size*2)
        lstm1_out = ht.permute(1, 2, 0).contiguous()
        lstm1_out = lstm1_out.view(emb.shape[0], self.hidden_size)
        lstm2_input = lstm1_out.view(batch_size, -1, self.hidden_size)
        #lstm2_input = self.dropout(lstm2_input)


        # Need all hidden states
        lstm2_out, _ = self.lstm2(lstm2_input)

        # Map to tag-space
        logits = self.linear(lstm2_out)

        with torch.no_grad():
            output = self.crf.decode(logits)

        return output, logits


