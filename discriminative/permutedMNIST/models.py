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
        """Bayesian multi-task neural network working through variational inference"""
        super(MFVI_NN, self).__init__(n_heads: int)


        self.posterior = None
        self.posterior_head = None

        self.init_weights()

        # Model components' definitions here
        pass

    def forward(self, x, task:int):
        """Forward pass of the multi-head bayesian nn. Does not include the last Softmax layer used for classification!"""

        W_mean, W_logvar, b_mean, b_logvar = self.posterior
        W_mean_head, W_logvar_head, b_mean_head, b_logvar_head = self.posterior_head[task]

        sampled_layers = self.sample_layers(W_mean, W_logvar, b_mean, b_logvar)
        for W_sample, b_sample in sampled_layers:
            x = W_sample @ x + b_sample
        return x


    def sample_parameters(self, W_mean: torch.Tensor, W_logvar: torch.Tensor, b_mean: torch.Tensor, b_logvar: torch.Tensor) -> list:
        """Samples parameters using the local reparameterization trick"""
        W_sample, b_sample = [], []
        for layer in range(len(W_mean)):
            epsilon_W = torch.randn_like(W_mean[layer])
            epsilon_b = torch.randn_like(b_mean[layer])

            W_sample.append(W_mean[layer] + espilon_W * torch.exp(0.5 * W_logvar[layer_n])) # Element-wise multiplication of epsilon with variance
            b_sample.append(b_mean[layer] + espilon_W * torch.exp(0.5 * b_logvar[layer_n])) 
        return zip(W_sample, b_sample)


    def mc_vcl_loss(self, x, y, n_samples):
        """
        Returns one basic summation element of the Monte Carlo version of the VCL loss.
        """
        return  self.log_likelihood(x, y) - kl_divergence()


    def log_likelihood(self, x, y, task, n_samples):
        list_predicted_y = []
        for sample in range(n_samples):
            list_predicted_y.append(self.forward(x, task))
        return -nn.CrossEntropyLoss()(torch.cat(list_predicted_y), y.repeat((n_samples,1)))

    def kl_divergence(self):
        pass

    def init_weights(self):
        pass

