import torch
import math

class VCL_layer(torch.nn.Module):
    def __init__(self, input_size: int, output_size: int, init_variance: float = 1e-5):
        super.__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.init_variance = init_variance

        prior_W_mean = torch.zeros(self.output_size,self.input_size) # Reversed dimensions for torch.nn.functional.linear
        prior_W_logvar = torch.zeros(self.output_size,self.input_size)
        prior_b_mean = torch.zeros(self.output_size)
        prior_b_logvar = torch.zeros(self.output_size)
        self.register_buffer('prior_W_mean', prior_W_mean)
        self.register_buffer('prior_W_logvar', prior_W_logvar)
        self.register_buffer('prior_b_mean', prior_b_mean)
        self.register_buffer('prior_b_logvar', prior_b_logvar)

        posterior_W_mean = torch.empty(self.output_size,self.input_size)
        posterior_W_logvar = torch.nn.init.constant(torch.empty(self.output_size,self.input_size), math.log(self.init_variance))
        posterior_b_mean = torch.empty(self.output_size)
        posterior_b_logvar = torch.nn.init.constant(torch.empty(self.output_size), math.log(self.init_variance))
        self.register_parameter('posterior_W_mean', posterior_W_mean)
        self.register_parameter('posterior_W_logvar', posterior_W_logvar)
        self.register_parameter('posterior_b_mean', posterior_b_mean)
        self.register_parameter('posterior_b_logvar', posterior_b_logvar)

    def sample_parameters(self):
        epsilon_W = torch.randn_like(W_mean[layer])
        epsilon_b = torch.randn_like(b_mean[layer])

        W_sample.append(posterior_W_mean[layer] + espilon_W * torch.exp(0.5 * posterior_W_logvar[layer_n])) # Element-wise multiplication of epsilon with variance
        b_sample.append(posterior_b_mean[layer] + espilon_W * torch.exp(0.5 * posterior_b_logvar[layer_n])) 
        return W_sample, b_sample

    def forward(self, x):
        W, b = self.sample_parameters()
        return torch.nn.functional.linear(x, W, b) # No activation function here, will be managed in main model

    def kl_divergence(self):
        #TODO: redo the demonstration of this
        prior_means = torch.autograd.Variable(torch.cat(
            (torch.reshape(self.prior_W_mean, (-1,)),
             torch.reshape(self.prior_b_mean, (-1,)))),
            requires_grad=False
        )
        prior_logvars = torch.autograd.Variable(torch.cat(
            (torch.reshape(self.prior_W_logvar, (-1,)),
             torch.reshape(self.prior_b_logvar, (-1,)))),
            requires_grad=False
        )
        prior_vars = torch.exp(prior_log_vars)

        posterior_means = torch.cat(
            (torch.reshape(self.posterior_W_mean, (-1,)),
             torch.reshape(self.posterior_b_mean, (-1,))),
        )
        posterior_logvars = torch.cat(
            (torch.reshape(self.posterior_W_logvar, (-1,)),
             torch.reshape(self.posterior_b_logvar, (-1,))),
        )
        posterior_vars = torch.exp(posterior_logvars)

        # compute kl divergence (this computation is valid for multivariate diagonal Gaussians)
        kl_elementwise = posterior_vars / (prior_vars + self.epsilon) + \
                         torch.pow(prior_means - posterior_means, 2) / (prior_vars + self.epsilon) - \
                         1 + prior_logvars - posterior_logvars

        return 0.5 * kl_elementwise.sum()
    
    def update_prior_posterior(self):
        """The previous posterior becomes the new prior"""
        self._buffers['prior_W_mean'].data.copy_(self.posterior_W_mean.data)
        self._buffers['prior_W_logvar'].data.copy_(self.posterior_W_logvar.data)
        self._buffers['prior_b_mean'].data.copy_(self.posterior_b_mean.data)
        self._buffers['prior_b_logvar'].data.copy_(self.posterior_b_logvar.data)
