import numpy as np
import gym
import model as m
import torch
import torch.nn as nn
import torch.optim as optim
import new_gaddag as g

def pick_action(weights, state):
    possible_actions = None # generate_possible_moves
    best_action = None
    best_action_heuristic = np.inf

    for action in possible_actions:
        heuristic = m.objective_function(weights["alpha"], weights["beta"], weights["gamma"], weights["delta"], weights["epsilon"], state["board"], action)

        if heuristic < best_action_heuristic:
            best_action = action
            best_action_heuristic = heuristic

    return best_action
    
class Network(nn.module):
    def __init__(self, state_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 5)
        )
        
    def forward(self, x):
        return self.network(x)
    
def gradient_descent(env_name, epochs=100):
    env = gym.make(env_name)
    dim = env.observation_space.shape[0]

    policy = Network(dim)
    optimizer = optim.Adam(policy.parameters(), lr=0.01)

    current_best_weights = None
    current_best_reward = float('-inf')

    for epoch in range(epochs):
        state, _ = env.reset()
        done = False

        # do batch training? or is that too much considering its prob high runtime

        while not done:
            state_tensor = torch.FloatTensor(state)
            weights = policy(state_tensor)

            # add noise to weights here?

            action = pick_action(weights, state)

            state, reward, done, _ = env.step(action)

        episode_reward = reward
        
        if episode_reward > current_best_reward:
                current_best_reward = episode_reward
                current_best_weights = weights.detach().numpy()
        
        loss = -episode_reward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Return: {episode_reward}")
    
    return policy, current_best_weights

class FinalWeightEvaluation:
    def __init__(self, env):
        self.env = env

    def evaluate_weights(self, weights, num_episodes=10):
        total_rewards = []

        for _ in range(num_episodes):
            state, _ = self.env.reset()
            ep_reward = 0
            done = False

            while not done:
                action = pick_action(weights, state)
                
                state, reward, done, _ = self.env.step(action)
                
            total_rewards.append(reward)

        return np.mean(total_rewards)