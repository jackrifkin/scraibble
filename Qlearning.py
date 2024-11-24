import numpy as np
import pickle
import gym

class descent_training:
    def __init__(self, env, num_episodes=1000, learning_rate=0.01, discount_factor=0.95, epsilon=0.1):
        self.env = env
        self.lrate = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon

        