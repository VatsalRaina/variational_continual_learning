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
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from transformers import AdamW, get_linear_schedule_with_warmup

from models import Vanilla_NN
from models import MFVI_NN

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--hidden_size', type=int, default=100, help='Specify the hidden embedding size in neural nets')
parser.add_argument('--batch_size', type=int, default=256, help='Specify the batch size')
parser.add_argument('--no_epochs', type=int, default=100, help='Specify the number of training epochs')
parser.add_argument('--num_tasks', type=int, default=5, help='Specify the number of tasks to perform in continual learning')
parser.add_argument('--coreset_size', type=int, default=0, help='Specify the coreset size for episodic memory')
parser.add_argument('--coreset_size', type=int, default=1, help='Seed value for reproducibility')
parser.add_argument('--adam_epsilon', type=float, default=e-8, help='Epislon value in Adam optimizer')
parser.add_argument('--learning_rate', type=int, default=2e-3, help='Learning rate during training')

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

    # Get all data for each task here
    # Note, the images must be converted to vectors to be used as inputs to deep neural networks
    #TODO
    # Adian must define the list of tensors X_train, Y_train


    ################## Train Vanilla_NN using data for first task ######################
    x_train, y_train = X_train[0], Y_train[0]
    train_data = TensorDataset(x_train, y_train)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size)

    model = Vanilla_NN(in_dim=x_train.size()[1], hidden_dim=args.hidden_size, out_dim=10).to(device)

    optimizer = AdamW(model.parameters(),
                    lr = args.learning_rate,
                    eps = args.adam_epsilon
                    )
    loss_values = []
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(args.n_epochs):
        # Perform one full pass over the training set.
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch + 1, args.n_epochs))
        print('Training...')
        # Measure how long the training epoch takes.
        t0 = time.time()
        # Reset the total loss for this epoch.
        total_loss = 0
        model.train()
        model.zero_grad()
        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):
            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)
                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
            b_x = batch[0].to(device)
            b_y = batch[1].to(device)
            model.zero_grad()
            logits = model(b_x)
            loss = criterion(logits, b_y)
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(train_dataloader)

        # Store the loss value for plotting the learning curve.
        loss_values.append(avg_train_loss)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epoch took: {:}".format(format_time(time.time() - t0)))

    # Now we are at a stage where we can extract the weights from the above trained model and call them the means
    mf_weights = model.parameters()
    mf_variances = None
    vanilla_weights = [mf_weights, mf_variances]

    ################## Train MFVI NN #######################

    model = MFVI_NN(in_dim=x_train.size()[1], hidden_dim=args.hidden_size, out_dim=10, num_tasks=args.num_tasks, prev_weights=vanilla_weights).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    for task_id in range(args.num_tasks):
        # Extract task specific data
        x_train, y_train = X_train[task_id], Y_train[task_id]
        train_data = TensorDataset(x_train, y_train)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size)
        # Set optimizer to be equal to all the shared parameters and the task specific head's parameters
        parameters = []
        parameters.extend(model.inputLayer.parameters())
        parameters.extend(model.hiddenLayer.parameters())
        parameters.extend(model.outputHeads[task_id].parameters())
        optimizer = AdamW(parameters, lr = args.learning_rate, eps = args.adam_epsilon)
        loss_values = []
        model.train()
        for epoch in range(args.n_epochs):
            for step, batch in enumerate(train_dataloader):
                b_x = batch[0].to(device)
                b_y = batch[1].to(device)
                model.zero_grad()
                prediction_logits = model(b_x, task_id)
                fit_loss = criterion(prediction_logits, b_y)
                # This is an inbuilt function for the imported BNN
                # However, this KL term is finding the KL divergence between the setting of parameters in the current and previous mini-batch
                # We are actually interested in finding the KL divergence between the setting of the parameters in the current mini-batch 
                # and the the final setting of the parameters from the previous TASK
                # So we will need to write our own KL divergence function which finds KL only for the shared parameters
                complexity_loss = model.nn_kl_divergence()  
                loss = fit_loss + complexity_loss
                total_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            avg_train_loss = total_loss / len(train_dataloader)
            loss_values.append(avg_train_loss)
            print("")
            print("  Average training loss: {0:.2f}".format(avg_train_loss))

        # Now perform evaluation on the test data
        x_test, y_test = X_test[task_id], Y_test[task_id]
        #TODO 

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)