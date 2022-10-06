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

    def get_action(self, obs: np.ndarray, sample=True) -> np.ndarray:
        # TODO: return sample from distribution if sampling
        # if not sampling return the mean of the distribution 
        if len(obs.shape) > 1: 
            obs_n = obs
        else: 
            obs_n = obs[None]

        obs_t = ptu.from_numpy(obs_n)

        # Get action dist
        if sample: 
            dist = self.forward(obs_t, sample = True)
            action_t = dist.rsample()

        else: 
            action_t = dist.mean()

        action_log_p = dist.log_prob(action_t).sum(dim = 1, keepdim = True)

        return action_t, action_log_p

    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    def forward(self, observation: torch.FloatTensor):
        # TODO: Implement pass through network, computing logprobs and apply correction for Tanh squashing

        # HINT: 
        # You will need to clip log values
        # You will need SquashedNormal from sac_utils file 

        mean_val = self.mean_net(observation)
        log_std_tanh = self.logstd.tanh()
        clipped_log_std = self.log_std_bounds[0] + (self.log_std_bounds[1] - self.log_std_bounds[0]) * (log_std_tanh + 1) / 2
        std_val = clipped_log_std.exp()

        # now lets use the squashed normal 
        action_distribution = sac_utils.SquashedNormal(mean_val, std_val)

        return action_distribution

    def update(self, obs, critic):
        # TODO Update actor network and entropy regularizer
        # return losses and alpha value

        obs_t = ptu.from_numpy(obs)
        
        #call forward to get te action
        action_t, action_log_p = self.get_action(obs_t, sample = True)
        qval1, qval2 = critic.forward(obs_t, action_t)
        min_qval = torch.min(qval1, qval2)
        
        # Calculate Actor Loss 
        actor_loss = (self.alpha.detach() * action_log_p - min_qval).mean()

        # Step
        self.optimizer.zero_grad()
        actor_loss.backward()
        self.optimizer.step()

        # Alpha Loss 
        alpha_loss = (self.alpha * (-1*action_log_p - self.target_entropy).detach()).mean()

        # Step Alpha Loss
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        return actor_loss, alpha_loss, self.alpha