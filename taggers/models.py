import numpy as np
import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2Model

from taggers.Crf import CRF
from taggers.infersent.models import InferSent


class InferSentClassifier(nn.Module):
    """
    A classifier that performs classification using
    infersent sentence embeddings
    """
    def __init__(self, num_labels, model_path, w2v_path, infersent_params, device="cpu", join_train=True, verbose=False):
        super(InferSentClassifier, self).__init__()
        self.infersent = InferSent(infersent_params)
        self.infersent.load_state_dict(torch.load(model_path))
        self.infersent.set_w2v_path(w2v_path)
        self.infersent.build_vocab_k_words(K=100000)
        if not join_train:
            # Disable fine-tuning of the infersent model
            for p in self.infersent.parameters():
                p.requires_grad = False

        self.linear = nn.Linear(infersent_params['enc_lstm_dim'] * 2, out_features=num_labels)
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
        """
        TODO: Implement version that makes use of hidden 
            states from all layers 
            
        TODO: We are currently picking hidden state at max_seq_len, instead pick last state for each turn
        """
        embed = hidden[:, -1, :]
        logits = self.linear(embed)

        loss = self.ce_loss(logits, labels.to(self.device)) if labels is not None else None

        return loss, logits

    def predict(self, sentences):
        with torch.no_grad():
            _, logits = self.forward(sentences)
            predictions = logits.argmax(dim=-1)
        return predictions


class PPLMGPT2Classifier(nn.Module):
    """
    A version of a classifier that performs classification
    based on the average representation as described in Dathathri et al. 2020
    """
    def __init__(
            self,
            num_labels,
            pretrained_model='microsoft/DialoGPT-medium',
            cached_mode=False,
            device="cpu"
    ):
        super(PPLMGPT2Classifier, self).__init__()

        self.tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.transformer = GPT2Model.from_pretrained(pretrained_model)
        self.transformer.to(device)
        # Freeze transformer parameters to train only classification head
        for param in self.transformer.parameters():
            param.requires_grad = False

        self.embed_size = self.transformer.config.hidden_size

        self.linear = nn.Linear(self.embed_size, num_labels)
        self.linear.to(device)
        self.cached_mode = cached_mode
        self.device = device

        self.ce_loss = nn.CrossEntropyLoss()


    def _avg_representation(self, x):
        mask = x.ne(0).unsqueeze(2).repeat(1, 1, self.embed_size).float().to(self.device).detach()
        hidden, _ = self.transformer(x)
        masked_hidden = hidden * mask
        avg_hidden = torch.sum(masked_hidden, dim=1) / (torch.sum(mask, dim=1).detach() + 1e-12)

        return avg_hidden

    def forward(self, x, labels=None):
        if self.cached_mode:
            avg_hidden = x.to(self.device)
        else:
            inp_dict = self.tokenizer.batch_encode_plus(x,
                                                        pad_to_max_length=True,
                                                        return_tensors="pt")
            x = inp_dict['input_ids'].to(self.device)
            avg_hidden = self._avg_representation(x)

        logits = self.linear(avg_hidden)
        loss = self.ce_loss(logits, labels.to(self.device)) if labels is not None else None
        return loss, logits

class InferSentCRFTagger(nn.Module):
    def __init__(self, num_labels, model_path, w2v_path, infersent_params,
                 utterance_encoder_hidden_size,
                 device="cpu", join_train=True, verbose=False):
        super(InferSentCRFTagger, self).__init__()

        self.infersent = InferSent(infersent_params).to(device)
        self.infersent.load_state_dict(torch.load(model_path))
        self.infersent.set_w2v_path(w2v_path)
        self.infersent.build_vocab_k_words(K=100000)
        if not join_train:
            # Disable fine-tuning of the infersent model
            for p in self.infersent.parameters():
                p.requires_grad = False

        self.dialog_encoder = nn.LSTM(input_size=infersent_params['enc_lstm_dim'] * 2,
                                      hidden_size=utterance_encoder_hidden_size,
                                      bidirectional=True,
                                      batch_first=True).to(device)

        self.linear = nn.Linear(utterance_encoder_hidden_size * 2, num_labels).to(device)
        self.crf = CRF(num_labels, batch_first=True).to(device)
        self.verbose = verbose
        self.device = device

    def bilstm_forward(self, dialogs):
        """
        We have dialogs as a List[List[str]] where outer list is of size batch_size.
        The inner list will have lists of varying lengths
        """
        batch_size = len(dialogs)
        max_seq_len = max(map(len, dialogs))
        dlens = []

        # We create a linearized sentence array
        sentences = []

        for i in range(batch_size):
            dlen = len(dialogs[i])
            dlens.append(dlen)

            for j in range(max_seq_len):
                if j < dlen:
                    sentences.append(dialogs[i][j])
                else:
                    sentences.append("")

        sents, lengths, idx_sort = self.infersent.prepare_samples(sentences, bsize=len(sentences), tokenize=True,
                                                                  verbose=self.verbose)

        batch = self.infersent.get_batch(sents).to(self.device)

        embed = self.infersent((batch, lengths))
        idx_unsort = np.argsort(idx_sort)
        embeddings = embed[idx_unsort]
        # embed: [batch_size * max_seq_len, sentence_embed_size]


        lstm2_input = embeddings.view(batch_size, -1, embeddings.shape[-1])
        # lstm2_input: [batch_size, max_seq_len, sentence_embed_size]

        dialog_embedding, _ = self.dialog_encoder(lstm2_input)
        # dialog_embedding: [batch_size, max_seq_len, utterance_encoder_hidden_size]

        logits = self.linear(dialog_embedding)
        # logits: [batch_size, max_seq_len, num_labels]
        return logits

    def forward(self, dialogs, tags, mask):
        logits = self.bilstm_forward(dialogs)

        score = self.crf(logits, tags, mask, reduction="mean")  # Mean reduction averages log-likelihood over the batch
        loss = score * -1

        return loss, logits

    def decode(self, dialogs, mask):
        logits = self.bilstm_forward(dialogs)

        with torch.no_grad():
            outputs = self.crf.decode(logits, mask)

        return outputs