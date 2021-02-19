#! /usr/bin/env python

"""
Run experiment for permuted MNIST task in an online manner.
Train and test models for each task at the same time.
"""
import argparse
import os
import sys

import numpy as np
import time
import datetime
import random
import torch 

from models import Vanilla_NN
from models import MFVI_NN

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--hidden_size', type=int, default=100, help='Specify the hidden embedding size in neural nets')
parser.add_argument('--batch_size', type=int, default=256, help='Specify the batch size')
parser.add_argument('--no_epochs', type=int, default=100, help='Specify the number of training epochs')
parser.add_argument('--num_tasks', type=int, default=5, help='Specify the number of tasks to perform in continual learning')
parser.add_argument('--coreset_size', type=int, default=0, help='Specify the coreset size for episodic memory')
parser.add_argument('--coreset_size', type=int, default=1, help='Seed value for reproducibility')

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

# Set device
def get_default_device():
    if torch.cuda.is_available():
        print("Got CUDA!")
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def main(args):

    # Set the seed value all over the place to make this reproducible.
    seed_val = args.seed
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    # Choose device
    device = get_default_device()

    """
    1) Generate train and test data for each task
    2) Train Vanilla_NN on first task
    3) Loop through all tasks and incrementally train MFVI_NN on each task and also evaluate on all test data (so far)
    for each task
    4) Generate accuracy plot for how it drops with each task
    """


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)