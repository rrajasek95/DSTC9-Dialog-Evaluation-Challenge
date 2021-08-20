import argparse
import json
import os
import pickle

import torch.cuda
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from transformers import GPT2Tokenizer, AdamW, GPT2DoubleHeadsModel

from utils.dataloader import collate_double_heads_data
from utils.trainer import GPT2DoubleHeadLMTrainer
from datasets.pd_nrg import PdNrgDataset


def load_model_and_tokenizer(model_checkpoint_path, tokenizer_path):
    gpt2_model = GPT2DoubleHeadsModel.from_pretrained(model_checkpoint_path)
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)

    gpt2_model.resize_token_embeddings(new_num_tokens=len(gpt2_tokenizer))

    return gpt2_model, gpt2_tokenizer


def load_data(train_data_path):
    with open(train_data_path, 'rb') as train_data_file:
        training_data = pickle.load(train_data_file)
        return PdNrgDataset(training_data)


def train_model(model, tokenizer, device, training_dataset, validation_dataset, training_configuration):
    optimizer = AdamW(model.parameters(), lr=training_configuration["lr"], correct_bias=True)

    data_collate_fn = lambda batch: collate_double_heads_data(batch, 0)

    train_loader = DataLoader(training_dataset, batch_size=training_configuration["train_batch_size"],
                              collate_fn=data_collate_fn, shuffle=True)
    valid_loader = DataLoader(validation_dataset, batch_size=training_configuration["valid_batch_size"],
                              collate_fn=data_collate_fn, shuffle=True)

    current_experiment_path = os.path.join(training_configuration["experiments_path"],
                                           training_configuration["experiment_name"])

    summary_writer = SummaryWriter(current_experiment_path)
    trainer = GPT2DoubleHeadLMTrainer(model, optimizer, device,
                                      gradient_accumulation_steps=training_configuration['gradient_accumulation_steps'],
                                      log_every_n=training_configuration["log_every_n"],
                                      save_every_n=training_configuration["save_every_n"],
                                      eval_every_n=training_configuration["eval_every_n"],
                                      model_checkpoint_directory=training_configuration["model_checkpoint_directory"],
                                      writer=summary_writer)
    print("Model summary:")
    print(model)
    print("Training configuration:")
    for key, item in training_configuration.items():
        print(key, ":", item)
    print("optimizer:", optimizer)

    with open(os.path.join(current_experiment_path, "training_parameters.json"), "w") as training_parameters_file:
        json.dump(training_configuration, training_parameters_file)

    trainer.train(train_loader, valid_loader, training_configuration["num_epochs"])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_data_path', type=str, default='data/processed/swbd_pd_nrg/training/training.pkl')
    parser.add_argument('--validation_data_path', type=str, default='data/processed/swbd_pd_nrg/training/validation.pkl')

    parser.add_argument("--model_checkpoint_path",
                        type=str,
                        default="gpt2-medium",
                        help="Name or path to initial model weights/checkpoint (for GPT-2 based models)")
    parser.add_argument('--tokenizer_path',
                        default="data/processed/swbd_pd_nrg/tokenizer",
                        help="Path to tokenizer to use for training")

    parser.add_argument('--device', default="cuda" if torch.cuda.is_available() else "cpu")

    experiment_args = parser.add_argument_group('Experiment Arguments', 'Arguments for setting up the experiment')
    experiment_args.add_argument('--experiment_name', type=str, required=True, help="Name of the experiment ran")
    experiment_args.add_argument('--model_checkpoint_directory', default='models/swbd_pd_nrg/v1/')
    experiment_args.add_argument('--experiments_path', default='experiments/')

    training_args = parser.add_argument_group('Training Parameters',
                                              'Lists the parameters for training the model')
    training_args.add_argument('--lr', type=float, default=6.25e-5, help="Base learning rate for training")
    training_args.add_argument('--num_epochs', type=int, default=3, help="Number of epochs to run training for")
    training_args.add_argument('--train_batch_size', type=int, default=4, help="Batch size for train step")
    training_args.add_argument('--valid_batch_size', type=int, default=8, help="Batch size for validation step")
    training_args.add_argument('--log_every_n', type=int, default=500,
                               help="Frequency of logging (in number of train steps)")
    training_args.add_argument('--save_every_n', type=int, default=1800,
                               help="Frequency of checkpoint save (in number of train steps)")
    training_args.add_argument('--eval_every_n', type=int, default=3600,
                               help="Frequency of model evaluation")

    training_args.add_argument('--gradient_accumulation_steps', type=int, default=8)
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.model_checkpoint_path, args.tokenizer_path)
    training_examples = load_data(args.training_data_path)
    validation_examples = load_data(args.validation_data_path)

    training_configuration = {
        "experiments_path": args.experiments_path,
        "experiment_name": args.experiment_name,
        "lr": args.lr,
        "num_epochs": args.num_epochs,
        "train_batch_size": args.train_batch_size,
        "valid_batch_size": args.valid_batch_size,
        "log_every_n": args.log_every_n,
        "save_every_n": args.save_every_n,
        "eval_every_n": args.eval_every_n,
        "model_checkpoint_directory": args.model_checkpoint_directory,
        "gradient_accumulation_steps": args.gradient_accumulation_steps
    }

    train_model(model, tokenizer, args.device, training_examples, validation_examples, training_configuration)
