import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal



class GaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes=(256, 256), activation=F.relu, log_std_min=-20, log_std_max=2):
        super(GaussianPolicy, self).__init__()
        self.activation = activation
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        layers = []
        input_dim = state_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_size))
            input_dim = hidden_size
        self.shared_net = nn.ModuleList(layers)

        self.mean_layer = nn.Linear(input_dim, action_dim)
        self.log_std_layer = nn.Linear(input_dim, action_dim)


    def forward(self, state):
        x = state
        for layer in self.shared_net:
            x = self.activation(layer(x))
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std


    def sample(self, state):
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        normal = Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)
        # To compute log_prob, account for the tanh transformation
        # See Appendix C of https://arxiv.org/pdf/1801.01290.pdf for details
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-7)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob, mean, std


    def get_log_prob(self, state, action):
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        normal = Normal(mean, std)
        action = torch.clamp(action, -0.999999, 0.999999)
        z = 0.5 * torch.log((1 + action) / (1 - action))
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-7)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return log_prob


