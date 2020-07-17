import numpy as np
import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2Model

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

class GPT2Classifier(nn.Module):
    def __init__(self,
                 num_labels,
                 pretrained_model='microsoft/DialoGPT-medium',
                 joint_train=False,
                 device="cpu"):
        super(GPT2Classifier, self).__init__()
        self.tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.encoder = GPT2Model.from_pretrained(pretrained_model).to(device)
        embed_size = self.encoder.config.hidden_size
        self.linear = nn.Linear(embed_size, num_labels)
        self.device = device
        self.ce_loss = nn.CrossEntropyLoss()

        if not joint_train:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, sents, labels=None):
        inp_dict = self.tokenizer.batch_encode_plus(sents,
                                                    pad_to_max_length=True,
                                                    return_tensors="pt")

        hidden, _ = self.encoder(input_ids=inp_dict["input_ids"].to(self.device),
                                 attention_mask=inp_dict["attention_mask"].to(self.device),
                                 token_type_ids=inp_dict["token_type_ids"].to(self.device))
        embed = hidden[:, -1, :]
        logits = self.linear(embed)

        loss = self.ce_loss(logits, labels.to(self.device)) if labels is not None else None

        return loss, logits

    def predict(self, sentences):
        with torch.no_grad():
            _, logits = self.forward(sentences)
            predictions = logits.argmax(dim=-1)
        return predictions