from itertools import chain, cycle, islice

import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()

        self.use_embed = params['use_embed']

        if self.use_embed:
            self.embedding = nn.Embedding(
                params['input_size'],
                params['embed_dim']
            )

        self.rnn = nn.GRU(
            input_size=params['embed_dim'],
            hidden_size=params['hidden_dim'],
            num_layers=params['num_layers'],
            batch_first=True,
            bidirectional=True
        )

    def forward(self, x):
        # x: [batch_size, max_seq_len]
        # lengths: [batch_size]
        if self.use_embed:
            embed = self.embedding(x)
        else:
            embed = x
        # embed: [batch_size, max_seq_len, embed_dim]
        return self.rnn(embed)

class HierarchicalDaPredictor(nn.Module):
    def __init__(self, params):
        super(HierarchicalDaPredictor, self).__init__()
        self.speaker_embeddings = nn.Embedding(3, params['speaker_embedding_dim'], padding_idx=0)

        # TODO: utterance and da encoders need embeddings, since they operate on words and labels
        self.utterance_encoder = Encoder(params['utt_encoder'])
        self.da_encoder = Encoder(params['da_encoder'])

        # TODO: Context encoder should not have an embedding layer
        self.context_encoder = Encoder(params['context_encoder'])

        context_hidden_size = params['context_encoder']['hidden_dim']
        self.fc1 = nn.Linear(context_hidden_size, params['num_labels'])
        self.device = params['device']

    def _build_packed_seq(self, x, lengths):
        return nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True)

    def _unpack_seq(self, x):
        return nn.utils.rnn.pad_packed_sequence(x)

    def generate_speaker_embeddings(self, conv_len, max_len):
        speakers = list(islice(cycle([1, 2]), conv_len))
        padding = [0 for i in range(max_len - conv_len)]

        return nn.Embedding(torch.LongTensor(speakers + padding)).to(self.device)


    def forward(self, dialogue, tag_sequence, conv_lengths, sent_lengths):

        batch_size = dialogue.shape[0]
        max_conv_len = dialogue.shape[0]
        # dialogue: [batch_size, max_conv_len, max_sent_len]
        sentences = dialogue.view(batch_size * max_conv_len, -1)
        flat_sent_lengths = list(chain(sent_lengths))

        packed_sentences = self._build_packed_seq(sentences, flat_sent_lengths)

        sent_out_packed = self.utterance_encoder(packed_sentences) # TODO: extract only the last hidden state for each sentence
        sent_out, _ = self._unpack_seq(sent_out_packed)
        # sent_out: [batch_size * max_conv_len, encoder_hidden_size]

        sent_out_unwound = sent_out.view(batch_size, max_conv_len, -1)
        # sent_out_unwound: [batch_size, max_conv_len, encoder_hidden_size]

        # speaker embeds: [batch_size, max_conv_len, speaker_embedding_dim]

        speaker_embeds = torch.stack([self.generate_speaker_embeddings(l, max_conv_len) for l in conv_lengths])
        context = torch.concat((sent_out_unwound, speaker_embeds), dim=-1)
        # context: [batch_size, max_conv_len, encoder_hidden_size + speaker_embedding_dim]

        # tag_sequence: torch.LongTensor with dimensions [batch_size, max_conv_len]
        tag_packed = self._build_packed_seq(tag_sequence, conv_lengths)
        da_out_packed = self.da_encoder(tag_packed)  # TODO: get hidden state for each turn of dialogue
        da_out, _ = self._unpack_seq(da_out_packed)
        # da_out [batch_size, max_conv_len, tag_encoder

        packed_context = self._build_packed_seq(context, conv_lengths)
        context_out_packed = self.context_encoder(packed_context) # TODO: extract only the last hidden state
        context_out, _ = self._unpack_seq(context_out_packed)
        # context_out: [batch_size, max_conv_len, context_encoder_hidden_size]

        logits = self.fc1(context_out)
        # logits: [batch_size, max_conv_len, num_labels]

        # To get prediction for each turn, compute argmax along the final dimension

        return logits