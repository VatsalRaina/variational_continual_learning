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

from models import Vanilla_NN, MFVI_NN, VCL_discriminative

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--hidden_size', type=int, default=100, help='Specify the hidden embedding size in neural nets')
parser.add_argument('--batch_size', type=int, default=256, help='Specify the batch size')
parser.add_argument('--no_epochs', type=int, default=100, help='Specify the number of training epochs')
parser.add_argument('--num_tasks', type=int, default=5, help='Specify the number of tasks to perform in continual learning')
parser.add_argument('--coreset_size', type=int, default=0, help='Specify the coreset size for episodic memory')
parser.add_argument('--coreset_size', type=int, default=1, help='Seed value for reproducibility')
parser.add_argument('--adam_epsilon', type=float, default=1e-8, help='Epislon value in Adam optimizer')
parser.add_argument('--learning_rate', type=int, default=2e-3, help='Learning rate during training')
parser.add_argument('--use_from_scratch_model', action='store_true', help='Which model to use')


import gzip
import pickle 
from copy import deepcopy
from PIL import Image

class PermutedMnistGenerator():
    def __init__(self, num_tasks=10):
        #Unzipping and reading Compressex MNIST DATA
        f = gzip.open('mnist.pkl.gz', 'rb')
        u = pickle._Unpickler( f )
        u.encoding = 'latin1'
        train_set, valid_set, test_set = u.load()
        f.close()

        self.X_train = np.vstack((train_set[0], valid_set[0]))
        self.Y_train = np.hstack((train_set[1], valid_set[1]))
        self.X_test = test_set[0]
        self.Y_test = test_set[1]

    def create_tasks(self, num_tasks=10):
        np.random.seed(0)

        X_train, Y_train, X_test, Y_test = [], [], [] ,[]

        for i in range(num_tasks):
            x_train, y_train, x_test, y_test = self.generate_new_task()
            X_train.append(x_train)
            Y_train.append(y_train)
            X_test.append(x_test)
            Y_test.append(y_test)

        return (X_train, Y_train, X_test, Y_test)

    def print_example(self, examples=[0]):
        for example in examples:
            array = self.X_train[example]
            array_2D = np.reshape(array, (28, 28))
            img = Image.fromarray(np.uint8(array_2D * 255) , 'L')
            img.show()

    def generate_new_task(self):
        perm_inds = list(range(self.X_train.shape[1]))
        np.random.shuffle(perm_inds)

        # Retrieve train data
        x_train = deepcopy(self.X_train)
        x_train = x_train[:,perm_inds]
        # y_train = np.eye(10)[self.Y_train]   #One hot encodes labels
        y_train = self.Y_train

        # Retrieve test data
        x_test = deepcopy(self.X_test)
        x_test = x_test[:,perm_inds]
        # y_test = np.eye(10)[self.Y_test]
        y_test = self.Y_test

        return x_train, y_train, x_test, y_test

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

    use_from_scratch_model = args.use_from_scratch_model
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

    ################## Train Vanilla_NN using data for first task ######################
    data_processor = PermutedMnistGenerator(args.num_tasks)
    X_train, Y_train, X_test, Y_test = data_processor.create_tasks(args.num_tasks)
    x_train, y_train = torch.tensor(X_train[0]).to(device), torch.tensor(Y_train[0]).long().to(device)
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

    if use_from_scratch_model:
        model = VCL_discriminative(input_dim = x_train.size()[1], shared_layer_dim=args.hidden_size, output_dim=10, n_heads=args.num_tasks, prev_weights=vanilla_weights).to(device)
        optimizer = AdamW(model.parameters(), lr = args.learning_rate, eps = args.adam_epsilon)
    
    else:
        model = MFVI_NN(in_dim=x_train.size()[1], hidden_dim=args.hidden_size, out_dim=10, num_tasks=args.num_tasks, prev_weights=vanilla_weights).to(device)
        criterion = torch.nn.CrossEntropyLoss()
    
    
    for task_id in range(args.num_tasks):
        # Extract task specific data
        x_train, y_train = torch.tensor(X_train[task_id]).to(device), torch.tensor(Y_train[task_id]).long().to(device)
        train_data = TensorDataset(x_train, y_train)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size)
        # Set optimizer to be equal to all the shared parameters and the task specific head's parameters
        if not use_from_scratch_model:
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
                optimizer.zero_grad()


                if not use_from_scratch_model:
                    prediction_logits = model(b_x, task_id)
                    fit_loss = criterion(prediction_logits, b_y)
                    # This is an inbuilt function for the imported BNN
                    # However, this KL term is finding the KL divergence between the setting of parameters in the current and previous mini-batch
                    # We are actually interested in finding the KL divergence between the setting of the parameters in the current mini-batch 
                    # and the the final setting of the parameters from the previous TASK
                    # So we will need to write our own KL divergence function which finds KL only for the shared parameters
                    complexity_loss = model.nn_kl_divergence()  
                    loss = fit_loss + complexity_loss
                
                if use_from_scratch_model:
                    loss = model.vcl_loss(b_x, b_y, task_id, len(task_data))
                
                total_loss += loss.item()
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
