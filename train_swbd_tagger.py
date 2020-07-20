
"""
This new tagger for the switchboard corpus relies on pretrained
sentence embeddings along with capturing label dependencies. My hypothesis
is that a richer sentence representation coupled with label dependencies
can help improve the model accuracy.

The key question is whether InferSent which is trained on translation corpora
can still be used to capture representations for spoken conversation.
- Rishi
"""
import json
import math
import pickle
import random

import torch
from sklearn.metrics import classification_report
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from taggers.dataset import SwdaDataset
from taggers.models import InferSentCRFTagger
from train_util.metrics import RunningLambdaMetric, MetricLambda


def load_swbd_data(args):
    with open(args.switchboard_data_path, 'r') as switchboard_corpus_file:
        switchboard_data = json.load(switchboard_corpus_file)

    return switchboard_data

def run_train(model, optimizer, loader, args):
    running_loss = RunningMetric()
    ppl = MetricLambda(math.exp, running_loss)

    model.train()
    for i, batch in tqdm(enumerate(loader)):

        sents, y = batch

        loss, _ = model(sents, y)

        running_loss.add(float(loss))

        loss.backward()
        if args.max_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)

        if i % 100 == 0:
            print("Running loss: ", running_loss.get())
        if i % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()


    print(f"Epoch loss: {running_loss.get()}")
    print(f"Epoch PPL: {ppl.get()}")

def run_eval(model, loader, vocab, args):
    model.eval()

    running_nll = RunningLambdaMetric(CrossEntropyLoss())

    ppl = MetricLambda(math.exp, running_nll)

    all_preds = []
    all_labels = []
    all_sents = []
    with torch.no_grad():
        for i, batch in tqdm(enumerate(loader)):
            sents, labels = batch

            _, logits = model(sents)
            predictions = logits.argmax(dim=-1)
            all_sents += sents
            all_preds += predictions.tolist()
            all_labels += labels.tolist()
            running_nll.add(logits.cpu(), labels)

    print("Validation:")
    print(f"NLL Loss: {running_nll.get()}")
    print(f"Perplexity: {ppl.get()}")
    print(f"Classification report")

    labels = [vocab.itos[i] for i in range(1, len(vocab))]
    preds = [vocab.itos[pred] for pred in all_preds]
    labs = [vocab.itos[lab] for lab in all_labels]

    print(classification_report(labs, preds, labels=labels))

def train_loop(model, optimizer, loaders, vocab, args):
    train_loader, valid_loader = loaders
    for i in range(args.n_epochs):
        print(f"Epoch {i + 1}")
        run_train(model, optimizer, train_loader, args)
        run_eval(model, valid_loader, vocab, args)

        torch.save(model.state_dict(), f'taggers/checkpoints/{args.model}_clf_{i + 1}.pt')

def train_infersent_crf_model(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_conversations, dev_conversations = load_swbd_data(args)

    train, valid = SwdaDataset(train_conversations), SwdaDataset(dev_conversations)
    V = 2
    params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                    'pool_type': 'max', 'dpout_model': 0.0, 'version': V}

    model = InferSentCRFTagger()



    model.to(args.device)

    train_loader = DataLoader(train, batch_size=args.batch_size,
                              collate_fn=prepare_batches,
                              shuffle=(not args.distributed))
    valid_loader = DataLoader(valid, batch_size=args.batch_size,
                              collate_fn=prepare_batches,
                              shuffle=False)

    optimizer = Adam(model.parameters(), lr=args.lr)

    with open('taggers/checkpoints/infersent_crf_config.pkl', 'wb') as infersent_training_config_file:
        pickle.dump({
            "params": params_model,
            "vocab": train.vocab
        }, infersent_training_config_file)

    train_loop(model, optimizer, (train_loader, valid_loader), train.vocab, args)

if __name__ == '__main__':
    pass