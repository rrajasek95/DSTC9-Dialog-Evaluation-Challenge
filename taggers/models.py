import torch
import torch.nn as nn

import numpy as np

from taggers.infersent.models import InferSent


class InferSentClassifier(nn.Module):
    """
    A classifier that performs classification using
    infersent sentence embeddings
    """
    def __init__(self, num_dacts, model_path, w2v_path, infersent_params, device="cpu", join_train=True, verbose=False):
        super(InferSentClassifier, self).__init__()

        self.infersent = InferSent(infersent_params)
        self.infersent.load_state_dict(torch.load(model_path))
        self.infersent.set_w2v_path(w2v_path)
        self.infersent.build_vocab_k_words(K=100000)
        if not join_train:
            # Disable fine-tuning of the infersent model
            for p in self.infersent.parameters():
                p.requires_grad = False

        self.linear = nn.Linear(infersent_params['enc_lstm_dim'] * 2, out_features=num_dacts)
        self.ce_loss = nn.CrossEntropyLoss()

        self.device = device
        self.verbose = verbose

    def forward(self, sentences, target=None):
        # sentences: [batch_size]
        sents, lengths, idx_sort = self.infersent.prepare_samples(sentences, bsize=len(sentences), tokenize=True, verbose=self.verbose)

        batch = self.infersent.get_batch(sents).to(self.device)

        embed = self.infersent((batch, lengths))

        idx_unsort = np.argsort(idx_sort)
        embeddings = embed[idx_unsort]

        # embed: [batch_size, embed_dim]
        logits = self.linear(embeddings)

        # logits: [batch_size, num_dacts]

        loss = self.ce_loss(logits, target.to(self.device)) if target is not None else None

        return loss, logits

    def visualize(self, sentence):
        self.infersent.visualize(sentence, tokenize=True)

    def predict(self, sentences):
        with torch.no_grad():
            _, logits = self.forward(sentences)
            predictions = logits.argmax(dim=-1)
        return predictions
