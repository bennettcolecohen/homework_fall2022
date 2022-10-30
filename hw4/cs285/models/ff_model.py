from torch import nn, normal
import torch
from torch import optim
from cs285.models.base_model import BaseModel
from cs285.infrastructure.utils import normalize, unnormalize
from cs285.infrastructure import pytorch_util as ptu
from typing import Dict, Tuple
import numpy as np

class FFModel(nn.Module, BaseModel):

    def __init__(self, ac_dim, ob_dim, n_layers, size, learning_rate=0.001):
        super(FFModel, self).__init__()

        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.size = size
        self.learning_rate = learning_rate
        self.delta_network = ptu.build_mlp(
            input_size=self.ob_dim + self.ac_dim,
            output_size=self.ob_dim,
            n_layers=self.n_layers,
            size=self.size,
        )
        self.delta_network.to(ptu.device)
        self.optimizer = optim.Adam(
            self.delta_network.parameters(),
            self.learning_rate,
        )
        self.loss = nn.MSELoss()
        self.obs_mean = None
        self.obs_std = None
        self.acs_mean = None
        self.acs_std = None
        self.delta_mean = None
        self.delta_std = None

    def update_statistics(
            self,
            obs_mean: torch.Tensor,
            obs_std: torch.Tensor,
            acs_mean: torch.Tensor,
            acs_std: torch.Tensor,
            delta_mean: torch.Tensor,
            delta_std: torch.Tensor,
    ) -> None:
        self.obs_mean = obs_mean
        self.obs_std = obs_std
        self.acs_mean = acs_mean
        self.acs_std = acs_std
        self.delta_mean = delta_mean
        self.delta_std = delta_std

    def forward(
            self,
            obs_unnormalized: torch.Tensor,
            acs_unnormalized: torch.Tensor,
            obs_mean: torch.Tensor,
            obs_std: torch.Tensor,
            acs_mean: torch.Tensor,
            acs_std: torch.Tensor,
            delta_mean: torch.Tensor,
            delta_std: torch.Tensor,
    ):
        """
        :param obs_unnormalized: Unnormalized observations
        :param acs_unnormalized: Unnormalized actions
        :param obs_mean: Mean of observations
        :param obs_std: Standard deviation of observations
        :param acs_mean: Mean of actions
        :param acs_std: Standard deviation of actions
        :param delta_mean: Mean of state difference `s_t+1 - s_t`.
        :param delta_std: Standard deviation of state difference `s_t+1 - s_t`.
        :return: tuple `(next_obs_pred, delta_pred_normalized)`
        This forward function should return a tuple of two items
            1. `next_obs_pred` which is the predicted `s_t+1`
            2. `delta_pred_normalized` which is the normalized (i.e. not
                unnormalized) output of the delta network. This is needed
        """
        
        # Normalize obs/acs
        obs_norm = normalize(obs_unnormalized, obs_mean, obs_std)
        acs_norm = normalize(acs_unnormalized, acs_mean, acs_std)

        # Combine obs/acs
        obs_ac = torch.cat([obs_norm, acs_norm], dim=1)

        # Calculate next obs_pred
        delta_pred_normalized = self.delta_network(obs_ac)
        delta_pred_unnormalized = unnormalize(delta_pred_normalized, delta_mean, delta_std)
        next_obs_pred = obs_unnormalized + delta_pred_unnormalized
        
        return next_obs_pred, delta_pred_normalized

    def get_prediction(self, observations, actions, data_statistics):
        """
        :param obs: numpy array of observations (s_t)
        :param acs: numpy array of actions (a_t)
        :param data_statistics: A dictionary with the following keys (each with
        a numpy array as the value):
             - 'obs_mean'
             - 'obs_std'
             - 'acs_mean'
             - 'acs_std'
             - 'delta_mean'
             - 'delta_std'
        :return: a numpy array of the predicted next-states (s_t+1)
        """
        
        # Convert obs/ac to tensor
        obs_unnorm = ptu.from_numpy(observations)
        acs_unnorm = ptu.from_numpy(actions)

        # Convert to tensors
        data_statistics = {k:ptu.from_numpy(v) for k,v in data_statistics.items()}
        
        # Get prediction and convert to numpy
        prediction, _ = self(obs_unnorm, acs_unnorm, **data_statistics)
        prediction = ptu.to_numpy(prediction)

        return prediction

    def update(self, observations, actions, next_observations, data_statistics):
        """
        :param obs: numpy array of observations
        :param acs: numpy array of actions
        :param next_obs: numpy array of next observations
        :param data_statistics: A dictionary with the following keys (each with
        a numpy array as the value):
             - 'obs_mean'
             - 'obs_std'
             - 'acs_mean'
             - 'acs_std'
             - 'delta_mean'
             - 'delta_std'
        :return: training loss
        """

        # Compute target 
        delta_target = next_observations - observations
        delta_target_norm = normalize(delta_target, data_statistics["delta_mean"], data_statistics["delta_std"])
        delta_target_norm = ptu.from_numpy(delta_target_norm)

        # Unnormalize
        obs_unnormalized=ptu.from_numpy(observations)
        acs_unnormalized=ptu.from_numpy(actions)

        # Convert to tensors and update statistics
        data_statistics = {k:ptu.from_numpy(v) for k,v in data_statistics.items()}
        self.update_statistics(**data_statistics)
        
        # Get prediction
        _, delta_pred_norm = self(obs_unnormalized, acs_unnormalized, **data_statistics)
        
        # Loss + step
        loss = self.loss(delta_pred_norm, delta_target_norm)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            'Training Loss': ptu.to_numpy(loss),
        }