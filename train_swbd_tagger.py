
"""
This new tagger for the switchboard corpus relies on pretrained
sentence embeddings along with capturing label dependencies. My hypothesis
is that a richer sentence representation coupled with label dependencies
can help improve the model accuracy.

The key question is whether InferSent which is trained on translation corpora
can still be used to capture representations for spoken conversation.
- Rishi
"""
import argparse
import json
import math
import os
import pickle
import random
from collections import defaultdict
from itertools import chain

import torch
from sklearn.metrics import classification_report
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from taggers.dataset import SwdaDataset
from taggers.models import InferSentCRFTagger
from train_util.metrics import RunningLambdaMetric, MetricLambda, RunningMetric


def load_swbd_data(args):
    with open(args.switchboard_data_path, 'r') as switchboard_corpus_file:
        switchboard_data = json.load(switchboard_corpus_file)

    split = defaultdict(list)

    for conv in switchboard_data["conversations"]:
        split[conv["partition_name"]].append(conv)

    return split["train"], split["dev"]

def run_train(model, optimizer, loader, args):
    running_loss = RunningMetric()
    ppl = MetricLambda(math.exp, running_loss)

    model.train()
    for i, batch in tqdm(enumerate(loader)):
        sents, tag_seqs, mask = batch
        tag_seqs = tag_seqs.to(args.device)
        mask = mask.to(args.device)

        loss, _ = model(sents, tag_seqs, mask)

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
    with torch.no_grad():
        model.eval()

        for i, batch in tqdm(enumerate(loader)):
            sents, tag_seqs, mask = batch
            tag_seqs = tag_seqs.to(args.device)
            mask = mask.to(args.device)

            outputs = model.decode(sents, mask)
            # Outputs will be a list [batch_size, var(seq_length)]
            # To compute NLL we need it as a tensor
            flat_outputs = list(chain(outputs))
            op_tensor = torch.LongTensor(flat_outputs)

            lab_tensor = tag_seqs.view(-1).cpu()
            flat_labels = lab_tensor.nonzero().tolist()

            all_preds += flat_outputs
            all_labels += flat_labels

            running_nll.add(op_tensor, lab_tensor)

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

        torch.save(model.state_dict(),
                   os.path.join(args.checkpoint_directory, f'{args.model}_clf_{i + 1}.pt'))

def prepare_batches(batch):
    dialogs, tag_seqs, lengths = zip(*batch)

    tensorized_tags = [torch.LongTensor(tag_seq) for tag_seq in tag_seqs]
    padded_tag_seqs = torch.nn.utils.rnn.pad_sequence(tensorized_tags, batch_first=True, padding_value=0)
    mask = padded_tag_seqs > 0  # EZ way to make a length mask
    return dialogs, padded_tag_seqs, mask

def train_infersent_crf_model(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_conversations, dev_conversations = load_swbd_data(args)

    train, valid = SwdaDataset(train_conversations), SwdaDataset(dev_conversations)
    V = 2
    params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                    'pool_type': 'max', 'dpout_model': 0.0, 'version': V}

    model = InferSentCRFTagger(train.num_labels(),
                               args.infersent_model_path,
                               args.infersent_w2v_path,
                               params_model,
                               300,
                               args.device,
                               join_train=args.joint_train)

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
    parser = argparse.ArgumentParser()
    parser.add_argument('-seed', type=int, default=42, help="Seed for training")
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
    parser.add_argument('--n_epochs', default=2, type=int)
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument('--joint_train', action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    parser.add_argument('--infersent_model_path', default='taggers/encoder/infersent2.pkl')
    parser.add_argument('--infersent_w2v_path', default='taggers/fastText/crawl-300d-2M.vec')
    parser.add_argument('--switchboard_data_path', type=str, default="taggers/ready_data/v1/swda-corpus-V1.json",
                        help="Switchboard data directory")
    parser.add_argument('--checkpoint_directory', default='taggers/checkpoints',
                        help='Path to save model checkpoint')
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank for distributed training (-1: not distributed)")


    args = parser.parse_args()

    args.distributed = (args.local_rank != -1)
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    train_infersent_crf_model(args)