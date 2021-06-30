import argparse

import torch.cuda

from utils.trainer import GPT2DoubleHeadLMTrainer


def train_model(args):
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', type=str, required=True, help="Name of the experiment ran")

    parser.add_argument('--data_path', type=str)
    parser.add_argument("--initial_checkpoint",
                        type=str,
                        default="gpt2-medium",
                        help="Name or path to initial model weights/checkpoint (for GPT-2 based models)")


    training_parameters = parser.add_argument_group('Training Parameters',
                                                    'Lists the parameters for training the model')
    training_parameters.add_argument('--lr', type=float, default=6.25e-5, help="Base learning rate for training")
    training_parameters.add_argument('--num_epochs', type=int, default=3, help="Number of epochs to run training for")
    training_parameters.add_argument('--train_batch_size', type=int, default=4, help="Batch size for train step")
    training_parameters.add_argument('--valid_batch_size', type=int, default=8, help="Batch size for validation step")
    training_parameters.add_argument('--log_every_n', type=int, default=500, help="Frequency of logging (in number of train steps)")
    training_parameters.add_argument('--save_every_n', type=int, default=1800, help="Frequency of checkpoint save (in number of train steps)")
    training_parameters.add_argument('--device', default="cuda" if torch.cuda.is_available() else "cpu")


    args = parser.parse_args()

    train_model(args)