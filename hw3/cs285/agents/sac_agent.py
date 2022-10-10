from collections import OrderedDict

from cs285.critics.bootstrapped_continuous_critic import \
    BootstrappedContinuousCritic
from cs285.infrastructure.replay_buffer import ReplayBuffer
from cs285.infrastructure.sac_utils import soft_update_params
from cs285.infrastructure.utils import *
from cs285.policies.MLP_policy import MLPPolicyAC
from .base_agent import BaseAgent
import gym
from cs285.policies.sac_policy import MLPPolicySAC
from cs285.critics.sac_critic import SACCritic
import cs285.infrastructure.pytorch_util as ptu
import torch

class SACAgent(BaseAgent):
    def __init__(self, env: gym.Env, agent_params):
        super(SACAgent, self).__init__()

        self.env = env
        self.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]
        self.agent_params = agent_params
        self.gamma = self.agent_params['gamma']
        self.critic_tau = 0.005
        self.learning_rate = self.agent_params['learning_rate']

        self.actor = MLPPolicySAC(
            self.agent_params['ac_dim'],
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            self.agent_params['discrete'],
            self.agent_params['learning_rate'],
            action_range=self.action_range,
            init_temperature=self.agent_params['init_temperature']
        )
        self.actor_update_frequency = self.agent_params['actor_update_frequency']
        self.critic_target_update_frequency = self.agent_params['critic_target_update_frequency']

        self.critic = SACCritic(self.agent_params)
        self.critic_target = copy.deepcopy(self.critic).to(ptu.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.training_step = 0
        self.replay_buffer = ReplayBuffer(max_size=100000)

    # def update_critic(self, ob_no, ac_na, next_ob_no, re_n, terminal_n):
    #     # TODO: 
    #     # 1. Compute the target Q value. 
    #     # HINT: You need to use the entropy term (alpha)
    #     # 2. Get current Q estimates and calculate critic loss
    #     # 3. Optimize the critic  

    #     # Grab everything as a tensor
    #     obs_t = ptu.from_numpy(ob_no)
    #     ac_t = ptu.from_numpy(ac_na)
    #     next_obs_t = ptu.from_numpy(next_ob_no)
    #     re_t = ptu.from_numpy(re_n)
    #     terminal_t = ptu.from_numpy(terminal_n)

    #     # Get next action 
    #     next_actions_t = self.actor.get_action(next_ob_no)

    #     # Get next step log probs 
    #     next_actions_dist = self.actor.forward(next_obs_t)
    #     next_log_probs = next_actions_dist.log_prob(next_actions_t)
    #     next_sum_log_probs = next_log_probs.sum(dim = 1, keepdim = True)
        
    #     # Calculate Target Q
    #     next_q1, next_q2 = self.critic_target(next_obs_t, next_actions_t)
    #     min_q = torch.min(next_q1, next_q2)
    #     target_q = torch.unsqueeze(re_t + self.gamma, dim = -1) * (min_q - self.actor.alpha * next_sum_log_probs)
    #     target_q = torch.squeeze(target_q)

    #     # Get current Q estimates
    #     curr_q1, curr_q2 = self.critic.forward(obs_t, ac_t)
    #     curr_q1 = torch.squeeze(curr_q1)
    #     curr_q2 = torch.squeeze(curr_q2)

    #     # Calculate critic loss 
    #     curr_q1_loss = self.critic.loss(curr_q1, target_q)
    #     curr_q2_loss = self.critic.loss(curr_q2, target_q)
    #     critic_loss = curr_q1_loss + curr_q2_loss
        
    #     # Optimize the critic 
    #     self.critic.optimizer.zero_grad()
    #     critic_loss.backward()
    #     self.critic.optimizer.step()
    #     return critic_loss.item()

    # def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
    #     # TODO 
    #     # for agent_params['num_critic_updates_per_agent_update'] steps,
    #     #     update the critic

    #     # 2. Softly update the target every critic_target_update_frequency (HINT: look at sac_utils)

    #     # If you need to update actor
    #     # for agent_params['num_actor_updates_per_agent_update'] steps,
    #     #     update the actor
    #     # 4. gather losses for logging

    #     # Grab everything as a tensor
    #     obs_t = ptu.from_numpy(ob_no)
    #     ac_t = ptu.from_numpy(ac_na)
    #     next_obs_t = ptu.from_numpy(next_ob_no)
    #     re_t = ptu.from_numpy(re_n)
    #     terminal_t = ptu.from_numpy(terminal_n)

    #     for i in range(self.agent_params['num_critic_updates_per_agent_update']): 
    #         critic_loss = self.update_critic(ob_no, ac_na, next_ob_no, re_n, terminal_n)

    #     soft_update_params(self.critic, self.critic_target, self.critic_tau)

    #     ## still need a line for if we actually need to update not sure which params
    #     for j in range(self.agent_params['num_actor_updates_per_agent_update']): 
    #         actor_loss, alpha_loss, alpha = self.actor.update(ob_no, self.critic)

    #     qval1, qval2 = self.critic.forward(obs_t, ac_t)
    #     q = torch.min(qval1, qval2)

    #     loss = OrderedDict()
    #     loss['Critic_Loss'] = critic_loss
    #     loss['Actor_Loss'] = actor_loss
    #     loss['Alpha_Loss'] = alpha_loss
    #     loss['Temperature'] = q.mean().item()

    #     return loss

    def update_critic(self, ob_no, ac_na, next_ob_no, re_n, terminal_n): 

        # Get things as tensors 
        ob_no = ptu.from_numpy(ob_no)
        ac_na = ptu.from_numpy(ac_na).to(torch.long)
        next_ob_no = ptu.from_numpy(next_ob_no)
        re_n = ptu.from_numpy(re_n)
        terminal_n = ptu.from_numpy(terminal_n)

        # Get Next Action
        next_action_dist = self.actor(next_ob_no)
        next_action = next_action_dist.sample()

        # Get next q
        next_q = self.critic_target(next_ob_no, next_action)

        # Get entropy 
        next_log_probs = next_action_dist.log_prob(next_action)

        # Compute target 
        target_q = re_n + self.gamma * (1 - terminal_n) * (next_q - self.actor.alpha * next_log_probs)

        # Call Update method from sac critic
        critic_loss = self.critic.update(ob_no, ac_na, target_q)

        return critic_loss

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n): 
        
        # ob_no = ptu.from_numpy(ob_no),
        # ac_na = ptu.from_numpy(ac_na),
        # re_n = ptu.from_numpy(re_n),
        # next_ob_no = ptu.from_numpy(next_ob_no),
        # terminal_n = ptu.from_numpy(terminal_n)
      
        # Update critic
        for i in range(self.agent_params['num_critic_updates_per_agent_update']): 
            critic_loss = self.update_critic(ob_no, ac_na, next_ob_no, re_n, terminal_n)

        # Update Target Critic per schedule
        if self.training_step % self.critic_target_update_frequency == 0: 
            soft_update_params( self.critic.Q1, self.critic_target.Q1, self.critic_tau)

            soft_update_params( self.critic.Q2, self.critic_target.Q2, self.critic_tau)

        # Update Actor
        if self.training_step % self.actor_update_frequency == 0:
            for _ in range(self.agent_params['num_actor_updates_per_agent_update']):
                actor_loss, alpha_loss, temperature = self.actor.update(ob_no, self.critic)

        loss = OrderedDict()
        loss['Critic_Loss'] = critic_loss
        loss['Actor_Loss'] = actor_loss
        loss['Alpha_Loss'] = alpha_loss
        loss['Temperature'] = temperature

        return loss
        

    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)

    def sample(self, batch_size):
        return self.replay_buffer.sample_recent_data(batch_size)
