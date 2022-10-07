from cs285.policies.MLP_policy import MLPPolicy
import torch
import numpy as np
from cs285.infrastructure import sac_utils
from cs285.infrastructure import pytorch_util as ptu
from torch import nn
from torch import optim
import itertools

class MLPPolicySAC(MLPPolicy):
    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 discrete=False,
                 learning_rate=3e-4,
                 training=True,
                 log_std_bounds=[-20,2],
                 action_range=[-1,1],
                 init_temperature=1.0,
                 **kwargs
                 ):
        super(MLPPolicySAC, self).__init__(ac_dim, ob_dim, n_layers, size, discrete, learning_rate, training, **kwargs)
        self.log_std_bounds = log_std_bounds
        self.action_range = action_range
        self.init_temperature = init_temperature
        self.learning_rate = learning_rate

        self.log_alpha = torch.tensor(np.log(self.init_temperature)).to(ptu.device)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.learning_rate)

        self.target_entropy = -ac_dim

    @property
    def alpha(self):
        # TODO: Formulate entropy term
        entropy = torch.exp(self.log_alpha)
        return entropy

    def get_action(self, obs: np.ndarray, sample = True) -> np.ndarray:
        # TODO: return sample from distribution if sampling
        # if not sampling return the mean of the distribution 
        if len(obs.shape) > 1: 
            obs_n = obs
        else: 
            obs_n = obs[None]

        # Get obs as tensor 
        obs_t = ptu.from_numpy(obs_n)

    
        # Call .forward()
        action_dist = self.forward(obs_t)

        if sample: 
            actions = action_dist.rsample().detach()
            actions = torch.tanh(actions)

        else: 
            actions = action_dist.mean.detach()


        # Calculate entropies
        log_probs = action_dist.log_prob(actions)
        # log_probs = action_dist.log_prob(actions) - torch.log(1 - actions.pow(2) + self.eps)
        entropies = -log_probs.sum(dim = 1, keepdim = True)

        return actions, entropies

    def forward(self, observation: torch.FloatTensor):
        # TODO: Implement pass through network, computing logprobs and apply correction for Tanh squashing
        # HINT: 
        # You will need to clip log values
        # You will need SquashedNormal from sac_utils file 


        # Get mean
        means = self.mean_net(observation)

        # Get stds
        log_stds = self.logstd.tanh()

        # Clip values
        log_std_lb = self.log_std_bounds[0]
        log_std_ub = self.log_std_bounds[1]
        log_stds = torch.clamp(log_stds, min = log_std_lb, max = log_std_ub) 

        # Squashed Normal
        log_stds = log_stds.exp()
        # dist = torch.distributions.Normal(means, log_stds)
        squashed_dist = sac_utils.SquashedNormal(means, log_stds)

        return squashed_dist

    def update(self, obs, critic):

        # TODO Update actor network and entropy regularizer
        # return losses and alpha value

        # Calculate clipped double Q
        action, entropy = self.get_action(obs, sample = True)
        q1, q2 = critic.forward(ptu.from_numpy(obs), action)
        q = torch.min(q1, q2)

        # Calculate actor loss
        actor_loss = torch.mean( -q - self.alpha * entropy )

        # Calculate alpha loss 
        alpha_loss = -torch.mean(self.log_alpha * (self.target_entropy - entropy).detach())

        # Actor -- backward
        self.optimizer.zero_grad()
        actor_loss.backward()
        self.optimizer.step()

        # Entropy -- backward 
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()


        return actor_loss, alpha_loss, self.alpha