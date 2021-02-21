#! /usr/bin/env python

import torch
import torchvision.models as models

class Vanilla_NN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):

        super(Vanilla_NN, self).__init__()

        self.relu = torch.nn.RELU()
        self.inputLayer = torch.nn.Linear(in_dim, hidden_dim)
        self.hiddenLayer = torch.nn.Linear(hidden_dim, hidden_dim)
        self.outputLayer = torch.nn.Linear(hidden_dim, out_dim)

    def forward(self, x):

        h1 = self.relu(self.inputLayer(x))
        h2 = self.relu(self.hiddenLayer(h1))
        prediction_logits = self.outputLayer(h2)
        
        return prediction_logits

        


class MFVI_NN(torch.nn.Module):
    def __init__(self):

        super(MFVI_NN, self).__init__()

        # Model components' definitions here
        pass

    def forward(self):

        # Model architecture here
        pass