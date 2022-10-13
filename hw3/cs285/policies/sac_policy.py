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

    def forward(self, observation):

        if self.discrete: 
            logits = self.logits_na(observation)
            action_dist = torch.distributions.Categorical(logits = logits)
            return action_dist
        else: 

            # Get mean 
            batch_means = self.mean_net(observation)

            # Get std 
            log_std = self.logstd

            # Clip std useing log_std_bounds
            min_std, max_std = self.log_std_bounds 
            clipped_log_std = torch.clamp(log_std, min = min_std, max = max_std)
            clipped_log_std = torch.exp(clipped_log_std)

            # Create squashed normal
            action_dist = sac_utils.SquashedNormal(
                loc = batch_means, 
                scale = clipped_log_std)

            return action_dist
        
    def get_action(self, obs: np.ndarray, sample = True) -> np.ndarray: 

        if len(obs.shape) > 1: 
            obs = obs
        else: 
            obs = obs[None]

        # Convert to tensor 
        if not isinstance(obs, torch.Tensor): 
            obs = ptu.from_numpy(obs)

        # Run policy
        action_dist = self(obs)

        # Get action
        if sample: 
            # Sample action from action_dist
            action = action_dist.rsample() 
        else: 
            # Use mean
            action = action_dist.mean

        # Use action ranges to clip action
        min_ac, max_ac = self.action_range
        action = torch.clip(action, min = min_ac, max = max_ac)

        # Calculate log probs
        log_probs = action_dist.log_prob(action)
        log_probs = log_probs.sum(dim = -1, keepdim = True)
        log_probs = log_probs.view(-1)

        # Convert back to np
        action = ptu.to_numpy(action)


        return action, log_probs

    def update(self, obs, critic):  

        # Convert obs to tensor
        obs = ptu.from_numpy(obs)

        # Get the action and log probs
        action, log_probs = self.get_action(obs, sample = True)
        
        # Convert action to numpy
        action_t = ptu.from_numpy(action)
        
        # Calculate qvals
        q1, q2 = critic(obs, action_t)
        q = torch.min(q1,q2).view(-1)

        # Actor Loss 
        actor_loss = torch.mean(self.alpha * log_probs - q)
        self.optimizer.zero_grad()
        actor_loss.backward()
        self.optimizer.step()

        # Alpha Loss 
        alpha_loss = -(self.alpha * (log_probs + self.target_entropy).detach()).mean()
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        return actor_loss, alpha_loss, self.alpha