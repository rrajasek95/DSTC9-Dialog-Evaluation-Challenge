import argparse
import math
import os
import json
import pickle

import pandas as pd
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from transformers import AdamW

from taggers.models import InferSentClassifier, GPT2Classifier, PPLMGPT2Classifier
from taggers.dataset import AthenaDaDataset
from sklearn.model_selection import train_test_split
from torch.utils.data.dataloader import DataLoader
from torch.optim import Adam

from train_util.metrics import RunningMetric, MetricLambda, RunningLambdaMetric


def load_json_data(filename, skip_labels=[]):
    file_path = os.path.join(filename)

    with open(file_path, 'r') as all_json_file:
        label_utt_dict = json.load(all_json_file)

        dataset = []

        for label, utterances in label_utt_dict.items():
            print("Label {} Count {}".format(label, len(utterances)))
            if label in skip_labels:
                print(f"Label {label} skipped!")
                continue

            if len(utterances) < 10:  # Skip sparse labels
                print("Skipped label ", label)
                continue
            for utt in utterances:
                dataset.append((utt, label))


    unified_df = pd.DataFrame(dataset, columns=['text', 'label'])
    return unified_df

def prepare_batch(x):
    sents, targets = zip(*x)

    return sents, torch.LongTensor(targets)

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


def train_infersent_model(args):
    skip_labels = [
        "device",
        "nonsense",
        "interjection",
        "abandon",
        "rq",
        "invalid-command",
        "pause",
        "request-repeat",
        "request-options",
        "stop-intent",
        "hold"
    ]

    V = 2
    params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                    'pool_type': 'max', 'dpout_model': 0.0, 'version': V}

    df = load_json_data(filename='all_augmented.json', skip_labels=skip_labels)
    train_df, valid_df = train_test_split(df, test_size=0.3, stratify=df['label'])

    train, valid = AthenaDaDataset(train_df), AthenaDaDataset(valid_df)

    model = InferSentClassifier(train.num_labels(), args.infersent_model_path, args.infersent_w2v_path, params_model, args.device, args.joint_train, args.verbose)

    model.to(args.device)

    train_loader = DataLoader(train, batch_size=args.batch_size, collate_fn=prepare_batch)
    valid_loader = DataLoader(valid, batch_size=args.batch_size, collate_fn=prepare_batch, shuffle=False)
    optimizer = Adam(model.parameters(), lr=args.lr)

    with open('taggers/checkpoints/infersent_config.pkl', 'wb') as infersent_training_config_file:
        pickle.dump({
            "params": params_model,
            "vocab": train.vocab,
        }, infersent_training_config_file)

    train_loop(model, optimizer, (train_loader, valid_loader), train.vocab, args)

def train_gpt2model(args):
    train, valid = load_athena_dataset(args)

    model = GPT2Classifier(train.num_labels(), args.model_checkpoint, joint_train=args.joint_train, device=args.device)

    model.to(args.device)

    train_loader = DataLoader(train, batch_size=args.batch_size, collate_fn=prepare_batch)
    valid_loader = DataLoader(valid, batch_size=args.batch_size, collate_fn=prepare_batch, shuffle=False)
    optimizer = AdamW(model.parameters(), lr=args.lr)

    train_loop(model, optimizer, (train_loader, valid_loader), train.vocab, args)

def train_pplm_tagger(args):
    train, valid = load_athena_dataset(args)

    model = PPLMGPT2Classifier(num_labels=train.num_labels(), pretrained_model=args.model_checkpoint, device=args.device)
    train_loader = DataLoader(train, batch_size=args.batch_size, collate_fn=prepare_batch)
    valid_loader = DataLoader(valid, batch_size=args.batch_size, collate_fn=prepare_batch, shuffle=False)
    optimizer = AdamW(model.parameters(), lr=args.lr)

    with open('taggers/checkpoints/pplm_config.pkl', 'wb') as pplm_training_config_file:
        pickle.dump({
            "vocab": train.vocab,
        }, pplm_training_config_file)

    train_loop(model, optimizer, (train_loader, valid_loader), train.vocab, args)


def load_athena_dataset(args):
    skip_labels = [
        "device",
        "nonsense",
        "interjection",
        "abandon",
        "rq",
        "invalid-command",
        "pause",
        "request-repeat",
        "request-options",
        "stop-intent",
        "hold"
    ]
    df = load_json_data(filename=os.path.join(args.data_path, 'all_augmented.json'), skip_labels=skip_labels)
    train_df, valid_df = train_test_split(df, test_size=0.3, stratify=df['label'])
    train, valid = AthenaDaDataset(train_df), AthenaDaDataset(valid_df)
    return train, valid

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="infersent",
                        choices=['infersent', 'gpt2', 'pplm'])
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
    parser.add_argument('--n_epochs', default=2, type=int)
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")

    parser.add_argument('--joint_train', default=False, type=bool)
    parser.add_argument('--verbose', action="store_true")
    parser.add_argument('--infersent_model_path', default='taggers/encoder/infersent2.pkl')
    parser.add_argument('--infersent_w2v_path', default='taggers/fastText/crawl-300d-2M.vec')

    parser.add_argument('--model_checkpoint', default="microsoft/DialoGPT-medium")
    parser.add_argument('--data_path', default='taggers/data',
                        help='Base path for the stored Athena dialog act data')

    args = parser.parse_args()

    if args.model == "gpt2":
        train_gpt2model(args)
    elif args.model == "pplm":
        train_pplm_tagger(args)
    else:
        train_infersent_model(args)