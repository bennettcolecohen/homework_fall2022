from cs285.policies.MLP_policy import MLPPolicy
import torch
import numpy as np
from cs285.infrastructure import sac_utils
from cs285.infrastructure import pytorch_util as ptu
from torch import nn, unbind
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

    # def get_action(self, obs: np.ndarray, sample = True) -> np.ndarray:
    #     # TODO: return sample from distribution if sampling
    #     # if not sampling return the mean of the distribution 
    #     if len(obs.shape) > 1: 
    #         obs_n = obs
    #     else: 
    #         obs_n = obs[None]

    #     # Get obs as tensor 
    #     obs_t = ptu.from_numpy(obs_n)
    
    #     # Call .forward()
    #     action_dist = self.forward(obs_t)

    #     if sample: 
    #         actions = action_dist.sample().detach()
    #     else: 
    #         actions = action_dist.mean.detach()

    #     # Calculate entropy
    #     # log_probs = action_dist.log_prob(actions)
    #     # entropies = -log_probs.sum(dim = 1, keepdim = True)


    #     return actions

    def get_action(self, obs: np.ndarray, sample = True) -> np.ndarray:

        if len(obs.shape) == 1: 
            obs = obs[None, ...]
        
        with torch.no_grad(): 
            action_dist = self.forward(ptu.from_numpy(obs))
            if sample: 
                action = action_dist.sample()
            else: 
                action = action_dist.mean

        # Clip Action
        action = action.clamp(min = self.action_range[0], max = self.action_range[1])
        action = ptu.to_numpy(action)

        return action
            
    def forward(self, observation: torch.FloatTensor): 

        if self.discrete: 

            logits = self.logits_na(observation)
            action_dist = torch.distributions.Categorical(logits)
        else: 
            
            # Get Mean + Std 
            batch_mean = self.mean_net(observation)
            batch_std = self.logstd.clip(min = self.log_std_bounds[0], max = self.log_std_bounds[1]).exp()

            # Squashed Normal
            action_dist = sac_utils.SquashedNormal(loc = batch_mean, scale = batch_std)

        return action_dist

    def update(self, obs, critic): 

        if isinstance(obs, np.ndarray): 
            obs = ptu.from_numpy(obs)

        # Get Action
        action_dist = self.forward(obs)
        action = action_dist.sample()

        # Get Log Probs
        log_prob = action_dist.log_prob(action)
        log_prob = log_prob.sum(dim = 1, keepdim = True)

        # Get q from critic
        q_val = critic.forward(obs, action)

        # Actor Loss
        actor_loss = (self.alpha.detach() * log_prob - q_val).mean()
        self.optimizer.zero_grad()
        actor_loss.backward()
        self.optimizer.step()

        # Alpha Loss
        alpha_loss = ( self.alpha * (-log_prob - self.target_entropy).detach() ).mean()
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()
        
        return actor_loss.item(), alpha_loss.item(), self.alpha


    # def forward(self, observation: torch.FloatTensor):
    #     # TODO: Implement pass through network, computing logprobs and apply correction for Tanh squashing
    #     # HINT: 
    #     # You will need to clip log values
    #     # You will need SquashedNormal from sac_utils file 

    #     # Get mean
    #     means = self.mean_net(observation)

    #     # Get stds
    #     log_stds = self.logstd
    #     log_stds = torch.tanh(log_stds)

    #     # Clip values
    #     lb = self.log_std_bounds[0]
    #     ub = self.log_std_bounds[1]
    #     clipped_log_stds = torch.clamp(log_stds, min = lb, max = ub) 
        

    #     # Squashed Normal
    #     squashed_dist = sac_utils.SquashedNormal(means, clipped_log_stds.exp())

    #     return squashed_dist

    # def update(self, obs, critic):

        # TODO Update actor network and entropy regularizer
        # return losses and alpha value

        obs_t = ptu.from_numpy(obs)

        # # Get action
        # action = self.get_action(obs, sample = True)

        # # Calculate entropy
        # action_dist = self.forward(obs_t)
        # log_probs = action_dist.log_prob(action)
        # sum_log_probs = log_probs.sum(dim = 1, keepdim = True)

        # All in one
        action_dist = self.forward(obs_t)
        actions = action_dist.rsample() 
        actions = torch.clamp(actions, min = self.action_range[0], max = self.action_range[1])

        log_probs = action_dist.log_prob(actions)
        sum_log_probs = torch.sum(log_probs, dim = 1)
        

        # Calculate clipped double Q
        q1, q2 = critic.forward(ptu.from_numpy(obs), actions)
        q = torch.min(q1, q2)

        # Calculate actor loss
        actor_loss = torch.mean( self.alpha * sum_log_probs - q)

    
        # Calculate alpha loss 
        alpha_loss = -torch.mean(self.log_alpha * (sum_log_probs + self.target_entropy ).detach())


        # Actor -- backward
        self.optimizer.zero_grad()
        actor_loss.backward()
        self.optimizer.step()

        # Entropy -- backward 
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()


        return actor_loss.item(), alpha_loss.item(), self.alpha