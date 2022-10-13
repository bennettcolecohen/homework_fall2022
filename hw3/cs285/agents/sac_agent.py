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

    def update_critic(self, ob_no, ac_na, next_ob_no, re_n, terminal_n):

        # Convert to tensors 
        ob_no = ptu.from_numpy(ob_no)
        ac_na = ptu.from_numpy(ac_na)
        next_ob_no = ptu.from_numpy(next_ob_no)
        re_n = ptu.from_numpy(re_n).unsqueeze(1)
        terminal_n = ptu.from_numpy(terminal_n).unsqueeze(1)

        with torch.no_grad(): 

            # Get next action
            next_action, next_log_probs = self.actor.get_action(ptu.to_numpy(next_ob_no), sample = True)

            if isinstance(next_action, np.ndarray):
                next_action = ptu.from_numpy(next_action)

            # Compute targets 
            next_q1, next_q2 = self.critic_target(next_ob_no, next_action)
            next_q = torch.minimum( next_q1, next_q2 )

            # Calculate target 
            target = re_n + self.gamma * (1 - terminal_n) * (next_q - self.actor.alpha * next_log_probs)

        # Get current q estimates 
        q1, q2 = self.critic(ob_no, ac_na)
        
        # Calculate loss for q1,q2
        q1_loss = self.critic.loss(q1, target)
        q2_loss = self.critic.loss(q2, target)
        critic_loss = q1_loss + q2_loss

        # Update Critic 
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()


        return critic_loss

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):

        # Update Critic 
        num_critic_updates = self.agent_params['num_critic_updates_per_agent_update']
        for _ in range(num_critic_updates): 
            critic_loss = self.update_critic(ob_no, ac_na, next_ob_no, re_n, terminal_n)

        if self.training_step % self.critic_target_update_frequency == 0:
            soft_update_params(self.critic, self.critic_target, self.critic_tau)


        # Update Actor 
        num_actor_updates = self.agent_params['num_actor_updates_per_agent_update']
        if self.training_step % self.actor_update_frequency == 0: 
            for _ in range(num_actor_updates): 
                actor_loss, alpha_loss, temperature = self.actor.update(ob_no, self.critic)



        # Gather Losses for logging
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
