#! /usr/bin/env python

import torch
import torchvision.models as models

from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator
from layer.bayesian import VCL_layer

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

    def init_weights():
        """
        Initialise using Vanilla neural netwrok parameters for the means and a pre-decided variance
        """
        pass

    def forward(self, x, task):

        h1 = self.relu(self.inputLayer(x))
        h2 = self.relu(self.hiddenLayer(h1))
        prediction_logits = self.outputHeads[task](h2)
        
        return prediction_logits



class VCL_discriminative_1(torch.nn.Module):
    def __init__(self):
        """Bayesian multi-task neural network working through variational inference, implemented from scratch."""
        super(MFVI_NN, self).__init__(n_heads: int, input_size:int, layer_size:int, n_layers:int, output_size:int, n_heads: int, initial_variance:float)

        self.prior_body, self.prior_heads = None, None
        self.posterior_body, self.posterior_heads = None, None

        self.input_size = input_size
        self.layer_size = layer_size
        self.n_layers = n_layers
        self.output_size = output_size
        self.n_heads = n_heads
        self.initial_variance = initial_variance

        self.init_weights()

        # Model components' definitions here
        pass

    def forward(self, x, task:int):
        """Forward pass of the multi-head bayesian nn."""

        # Go through the body layers
        W_mean, W_logvar, b_mean, b_logvar = self.posterior_body
        sampled_layers = self.sample_parameters(W_mean, W_logvar, b_mean, b_logvar)
        for W_sample, b_sample in sampled_layers:
            x = torch.nn.relu(W_sample @ x + b_sample)

        # Go through the head of the selected task. 
        W_mean_head, W_logvar_head, b_mean_head, b_logvar_head = self.posterior_heads[task]
        W_sample_head, b_sample_head = self.sample_parameters(W_mean_head, W_Logvar_head, b_mean_head, b_logvar_head)
        x = W_sample_head @ x + b_sample_head # Activation of the final layer will be done via softmax

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
            list_predicted_y.append(self(x, task))
        return -nn.CrossEntropyLoss()(torch.cat(list_predicted_y), y.repeat((n_samples,1)))

    def kl_divergence(self):
        pass

    def init_weights(self):
        W_mean_prior_body = [torch.zeros(self.input_size, self.layer_size)] + [torch.zeros(self.layer_size, self.layer_size)]*(self.n_layers - 1)
        W_logvar_prior_body = [torch.zeros(self.input_size, self.layer_size)] + [torch.zeros(self.layer_size, self.layer_size)]*(self.n_layers - 1)
        b_mean_prior_body = [torch.zeros(self.layer_size)]*self.n_layers
        b_logvar_prior_body = [torch.zeros(self.layer_size)]*self.n_layers
        self.prior_body = (W_mean_prior_body, W_logvar_prior_body, b_mean_prior_body, b_logvar_prior_body)

        W_mean_prior_heads = [torch.zeros(self.layer_size, self.output_size)]*self.n_heads
        W_logvar_prior_heads = [torch.zeros(self.layer_size, self.output_size)]*self.n_heads
        b_mean_prior_heads = [torch.zeros(self.output_size)]*self.n_heads
        b_logvar_prior_heads = [torch.zeros(self.output_size)]*self.n_heads
        self.prior_heads = zip(W_mean_prior_heads, W_logvar_prio_heads, b_mean_prior_heads, b_logvar_prior_heads)
        

        W_mean_posterior_body = [torch.empty(self.input_size, self.layer_size)] + [torch.empty(self.layer_size, self.layer_size)]*(self.n_layers - 1)
        W_logvar_posterior_body = [torch.empty(self.input_size, self.layer_size)] + [torch.empty(self.layer_size, self.layer_size)]*(self.n_layers - 1)
        b_mean_posterior_body = [torch.empty(self.layer_size)]*self.n_layers
        b_logvar_posterior_body = [torch.empty(self.layer_size)]*self.n_layers
        self.posterior_body = (W_mean_posterior_body, W_logvar_posterior_body, b_mean_posterior_body, b_logvar_posterior_body)

        W_mean_posterior_heads = [torch.empty(self.layer_size, self.output_size)]*self.n_heads
        W_logvar_posterior_heads = [torch.empty(self.layer_size, self.output_size)]*self.n_heads
        b_mean_posterior_heads = [torch.empty(self.layer_size)]*self.n_heads
        b_logvar_posterior_heads = [torch.empty(self.layer_size)]*self.n_heads

        #Initialize posterior variances with constant parameter
        for tensor in W_logvar_posterior_body + b_logvar_posterior_body + W_logvar_posterior_heads + b_logvar_posterior_heads:
            torch.nn.init.constant(tensor, math.log(self.initial_variance))

        self.posterior_heads = zip(W_mean_posterior_heads, W_logvar_prior_heads, b_mean_posterior_heads, b_logvar_posterior_heads)
        
        # We will register the prior as a buffer, since it will not get updated by the optimizer
        for layer in range(n_layers):
            self.register_buffer('W_mean_prior_body, layer '+str(layer), W_mean_prior_body[layer])
            self.register_buffer('W_logvar_prior_body, layer '+str(layer), W_mean_prior_body[layer])
            self.register_buffer('b_mean_prior_body, layer '+str(layer), b_mean_prior_body[layer])
            self.register_buffer('b_mean_prior_body, layer '+str(layer), b_mean_prior_body[layer])

        for head in range(n_heads):
            self.register_buffer('W_mean_prior_head, head '+str(head), W_mean_prior_head[head])
            self.register_buffer('W_logvar_prior_head, head '+str(head), W_mean_prior_head[head])
            self.register_buffer('b_mean_prior_head, head '+str(head), b_mean_prior_head[head])
            self.register_buffer('b_mean_prior_head, head '+str(head), b_mean_prior_head[head])

        # The posterior, on the other hand, has to be updated, so it will be registered as a parameter
        for layer in range(n_layers):
            self.register_parameter('W_mean_posterior_body, layer '+str(layer), W_mean_posterior_body[layer])
            self.register_parameter('W_logvar_posterior_body, layer '+str(layer), W_mean_posterior_body[layer])
            self.register_parameter('b_mean_posterior_body, layer '+str(layer), b_mean_posterior_body[layer])
            self.register_parameter('b_mean_posterior_body, layer '+str(layer), b_mean_posterior_body[layer])

        for head in range(n_heads):
            self.register_parameter('W_mean_posterior_head, head '+str(head), W_mean_posterior_head[head])
            self.register_parameter('W_logvar_posterior_head, head '+str(head), W_mean_posterior_head[head])
            self.register_parameter('b_mean_posterior_head, head '+str(head), b_mean_posterior_head[head])
            self.register_parameter('b_mean_posterior_head, head '+str(head), b_mean_posterior_head[head])


class VCL_discriminative(torch.nn.module):
    def __init__(self, input_size, shared_layer_size, output_size, n_shared_layers, n_heads, init_variance):
        super.__init__()
        self.input_size = input_size
        self.shared_layer_size = layer_size
        self.output_size = output_size
        self.n_shared_layers = n_layers
        self.n_heads = n_heads
        self.init_variance = init_variance

        layer_sizes = [input_size] + [shared_layer_size]*n_shared_layers
        self.shared_layers = torch.nn.ModuleList([VCL_layer(layer_sizes[i], layer_sizes[i+1], init_variance) for i in range(len(layer_sizes)-1)])

        self.heads = torch.nn.ModuleList([VCL_layer(shared_layer_size[-1], output_size, init_variance) for _ in range(n_heads)])

        self.softmax = nn.Softmax(dim=1)
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

    def update_prior_posterior(self. head:int):
        for layer in self.shared_layers:
            layer.update_prior_posterior()
        self.heads[head].update_prior_posterior()
        return

    def prediction(self, x, head:int):
        return torch.argmax(self(x, head))







