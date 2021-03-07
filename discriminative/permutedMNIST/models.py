#! /usr/bin/env python

import torch
import torchvision.models as models

from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator
from layer.bayesian import VCL_layer

class Vanilla_NN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_hidden_layers=5):

        super(Vanilla_NN, self).__init__()

        self.relu = torch.nn.RELU()
        self.inputLayer = torch.nn.Linear(in_dim, hidden_dim)
        self.hiddenLayers = []
        for i in range(num_hidden_layers):
            self.hiddenLayers.append(torch.nn.Linear(hidden_dim, hidden_dim))
        self.outputLayer = torch.nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        h1 = self.relu(self.inputLayer(x))
        for hidden_layer in self.hiddenLayers:
            h2 = self.relu(hidden_layer(h1))
            h1=h2
        prediction_logits = self.outputLayer(h2)
        
        return prediction_logits        



# See explanation at:
# https://towardsdatascience.com/blitz-a-bayesian-neural-network-library-for-pytorch-82f9998916c7
# Original code at:
# https://github.com/piEsposito/blitz-bayesian-deep-learning 
@variational_estimator
class MFVI_NN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_tasks, prev_weights):

        super(MFVI_NN, self).__init__()

        self.relu = torch.nn.RELU()
        self.inputLayer = BayesianLinear(in_dim, hidden_dim)
        self.hiddenLayer = BayesianLinear(hidden_dim, hidden_dim)
        self.outputHeads = []
        for i in range(num_tasks):
            self.outputHeads.append(BayesianLinear(hidden_dim, out_dim))

        # Initialise using the Vanilla neural network weights when the model is first initialised
        self.init_weights(prev_weights)

    def init_weights(self, prev_weights):
        """
        Initialise using Vanilla neural netwrok parameters for the means and a pre-decided variance
        """
        pass

    def forward(self, x, task):

        h1 = self.relu(self.inputLayer(x))
        h2 = self.relu(self.hiddenLayer(h1))
        prediction_logits = self.outputHeads[task](h2)
        
        return prediction_logits



class VCL_discriminative(torch.nn.module):
    def __init__(self, input_dim, shared_layer_dim, output_dim, n_shared_layers, n_heads, init_variance):
        super.__init__()
        self.input_dim = shared_layer_dim
        self.shared_layer_dim = layer_size
        self.output_dim = output_dim
        self.n_shared_layers = n_shared_layers
        self.n_heads = n_heads
        self.init_variance = init_variance

        layer_sizes = [input_dim] + [shared_layer_dim]*n_shared_layers
        self.shared_layers = torch.nn.ModuleList([VCL_layer(layer_sizes[i], layer_sizes[i+1], init_variance) for i in range(len(layer_sizes)-1)])

        self.heads = torch.nn.ModuleList([VCL_layer(shared_layer_dim[-1], output_dim, init_variance) for _ in range(n_heads)])

        self.softmax = torch.nn.Softmax(dim=1)
        return

    def forward(self, x, head:int):
        for layer in self.layers[:-1]:
            x = torch.nn.functional.relu(layer(x))
        x = self.layers[-1](x)
        x = self.softmax(x)
        return x

    def vcl_loss(self, head:int, x, y):
        return self.KL_div(head) - torch.nn.NLLLoss()(self(x, head), y)

    def kl_divergence(self, head:int):
        div = torch.zeros(1,0)
        for layer in self.shared_layers:
            div = torch.add(div, layer.kl_divergence())
        div = torch.add(div, self.heads[head].kl_divergence())
        return div

    def update_prior_posterior(self, head:int):
        for layer in self.shared_layers:
            layer.update_prior_posterior()
        self.heads[head].update_prior_posterior()
        return

    def prediction(self, x, head:int):
        return torch.argmax(self(x, head))







