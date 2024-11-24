import numpy as np
import pickle
import gym
import model as m

class descent_training:
    def __init__(self, env, num_episodes=1000):
        self.env = env
        
    def evaluate_weights(self, ):
        total_punishments = [] # punishment as in opposite of reward, for minimizing

        for _ in range(self.num_episodes):
            state, _ = self.env.reset()
            episode_punishment = 0
            done = False

            while not done:
                