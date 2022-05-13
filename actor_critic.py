import torch
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = 64
        self.tanh = torch.nn.Tanh()

        """
            Actor network
        """
        self.fc1_actor = torch.nn.Linear(state_space, self.hidden)
        self.fc2_actor = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_actor_mean = torch.nn.Linear(self.hidden, action_space)
        
        # Learned standard deviation for exploration at training time 
        self.sigma_activation = F.softplus
        init_sigma = 0.5
        self.sigma = torch.nn.Parameter(torch.zeros(self.action_space)+init_sigma)
        
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        """
            Actor
        """
        x_actor = self.tanh(self.fc1_actor(x))
        x_actor = self.tanh(self.fc2_actor(x_actor))
        action_mean = self.fc3_actor_mean(x_actor)

        sigma = self.sigma_activation(self.sigma)
        normal_dist = Normal(action_mean, sigma)

        return normal_dist

        

class Value_function(torch.nn.Module):
    def __init__(self, state_space):
      super().__init__()
      self.state_space = state_space
      self.hidden = 64
      # TODO 2.2.b: critic network for actor-critic algorithm
      """
          Critic network
      """

      self.fc1_critic = torch.nn.Linear(state_space, self.hidden)
      self.fc2_critic = torch.nn.Linear(self.hidden, self.hidden)
      self.fc3_critic = torch.nn.Linear(self.hidden, 1)
      
      self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)
                  
    def forward(self, x):
        x_critic = F.relu(self.fc1_critic(x))
        x_critic = F.relu(self.fc2_critic(x_critic))
        value = self.fc3_critic(x_critic)
        
        return value

class Agent(object):
    def __init__(self, learning_rate_vf, learning_rate_p, policy, value_function, device='cpu'):
        self.train_device = device
        self.policy = policy.to(self.train_device)
        self.value_funtion = value_function.to(self.train_device)
        self.policy_optimizer = torch.optim.Adam(policy.parameters(), learning_rate_p)
        self.vf_optimizer = torch.optim.Adam(value_function.parameters(), learning_rate_vf)

        self.gamma = 0.99
        self.states = []
        self.next_states = []
        self.action_log_probs = []
        self.rewards = []
        self.done = []

    def update_policy(self):
      
        action_log_probs = torch.stack(self.action_log_probs, dim=0).to(self.train_device).squeeze(-1)
        states = torch.stack(self.states, dim=0).to(self.train_device).squeeze(-1)
        next_states = torch.stack(self.next_states, dim=0).to(self.train_device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        done = torch.Tensor(self.done).to(self.train_device)

        actor_losses = [] 
        critic_losses = []       
        #
        # TODO 2.2.b:
        #             - compute boostrapped discounted return estimates
        #             - compute advantage terms
        for log_prob, state, next_state, reward, done in zip(self.action_log_probs, self.states, self.next_states, self.rewards, self.done):
              value = self.value_funtion(state)
              next_value = self.value_funtion(next_state)
              discounted_returns = reward + self.gamma * next_value*(1- int(done))
              advantage = discounted_returns - value
              advantage.detach()
              # calculate actor loss 
              actor_losses.append(-log_prob * advantage)
              # calculate critic loss
              critic_losses.append(advantage**2)
    

        #             - compute gradients and step the optimizer
        actor_loss = torch.stack(actor_losses).sum()
        self.policy_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.policy_optimizer.step()

        self.vf_optimizer.zero_grad()
        critic_loss = torch.stack(critic_losses).sum()
        critic_loss.backward(retain_graph = True)
        self.vf_optimizer.step()
        
        return
    
    def get_action(self, state, evaluation=False):
        x = torch.from_numpy(state).float().to(self.train_device)

        normal_dist = self.policy(x)

        if evaluation:  # Return mean
            return normal_dist.mean, None

        else:   # Sample from the distribution
            action = normal_dist.sample()

            # Compute Log probability of the action [ log(p(a[0] AND a[1] AND a[2])) = log(p(a[0])*p(a[1])*p(a[2])) = log(p(a[0])) + log(p(a[1])) + log(p(a[2])) ]
            action_log_prob = normal_dist.log_prob(action).sum()

            return action, action_log_prob
            
    def empty_outcome(self):
        self.states.clear()
        self.next_states.clear()
        self.action_log_probs.clear()
        self.rewards.clear()
        self.done.clear()
 

    def store_outcome(self, state, next_state, action_log_prob, reward, done):
        self.states.append(torch.from_numpy(state).float())
        self.next_states.append(torch.from_numpy(next_state).float())
        self.action_log_probs.append(action_log_prob)
        self.rewards.append(torch.Tensor([reward]))
        self.done.append(done)
