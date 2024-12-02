import numpy as np
import gym
import model as m
import new_gaddag as g
import util as u
from scrabble_gym import ScrabbleEnv

def pick_action(weights, state):
    board = state["board"]
    possible_actions = u.generate_possible_moves(board, state["letter_rack"], state["cross_sets"])
    possible_actions = [action for action in possible_actions if len(action) > 1]
    best_action = None
    best_action_heuristic = -1
    best_action_factors = np.zeros(5)

    # TODO: just for debugging, remove
    # rack_str = ''
    # for letter in state["letter_rack"]:
    #     rack_str += u.char_idx_to_char(letter)
    # print(rack_str)

    for action in possible_actions:
        try:
            points_scored_val = m.points_scored(board, action)
        except ValueError as e:
            # action is invalid, skip
            continue
        # TODO: uncomment heuristics
        # weighted_multipliers_val = m.weighted_multipliers(action)
        # action_use_val = m.action_use_value(action)
        # multiplier_distance_reduction_val = m.multiplier_distance_reduction(action)
        # opened_spaces_val = m.opened_spaces(board, action)

        # heuristic = m.objective_function(weights, points_scored_val, weighted_multipliers_val, action_use_val, multiplier_distance_reduction_val, opened_spaces_val)

        # if heuristic > best_action_heuristic:
        #     best_action = action
        #     best_action_heuristic = heuristic
        #     best_action_factors[0] = points_scored_val
        #     best_action_factors[1] = weighted_multipliers_val
        #     best_action_factors[2] = action_use_val
        #     best_action_factors[3] = multiplier_distance_reduction_val
        #     best_action_factors[4] = opened_spaces_val

        # TODO: REMOVE
        if points_scored_val > best_action_heuristic:
            best_action = action
            best_action_heuristic = points_scored_val

    return best_action, best_action_factors
    
def gradient_descent(epochs=1, decay_rate=0.9999, lr=0.001):
    env = ScrabbleEnv()

    best_weights1 = np.ones(5) / 5 # initialize weights evenly
    best_weights2 = np.ones(5) / 5
    epsilon = 1

    for _ in range(epochs):
        state = env.reset()
        done = False

        if np.random.rand() < epsilon:
            # use random weights
            weights1 = np.random.dirichlet(np.ones(5))
            weights2 = np.random.dirichlet(np.ones(5))
        else:
            # use optimal weights
            weights1 = best_weights1
            weights2 = best_weights2

        factor_sums1 = np.zeros(5)
        factor_sums2 = np.zeros(5)

        while not done:
            # choose action for player 1
            action1, action_factors1 = pick_action(weights1, state)
            
            # add to heuristic factor sum for player 1
            factor_sums1 += action_factors1
            
            # perform player 1 action
            state, score1, done, _ = env.step(action1)
            print(env.render()) # TODO: remove

            # choose action for player 2
            action2, action_factors2 = pick_action(weights2, state)
            
            # add to heuristic factor sum for player 2
            factor_sums2 += action_factors2
            
            # perform player 2 action
            state, score2, done, _ = env.step(action2)
            print(env.render()) # TODO: remove
            
        # decay epsilon
        epsilon *= decay_rate

        # calculate episode score diff for each player
        episode_score_diff1 = (score1 - score2) / (score1 + score2)
        episode_score_diff2 = -episode_score_diff1 / (score1 + score2)
        
        # Compute gradients with respect to each weight, update best_weights
        for i in range(len(factor_sums1)):
            grad_i = episode_score_diff1 * factor_sums1[i] * lr
            best_weights1[i] += grad_i
        # normalize weights
        best_weights1 = best_weights1 / np.sum(best_weights1)

        # Compute gradients with respect to each weight, update best_weights
        for i in range(len(factor_sums2)):
            grad_i = episode_score_diff2 * factor_sums2[i] * lr
            best_weights2[i] += grad_i
        # normalize weights
        best_weights2 = best_weights2 / np.sum(best_weights2)

        print(env.render())
    
    return best_weights1, best_weights2

gradient_descent()

class FinalWeightEvaluation:
    def __init__(self, env):
        self.env = env

    def evaluate_weights(self, weights, num_episodes=100):
        total_scores = []

        for _ in range(num_episodes):
            state, _ = self.env.reset()
            done = False

            while not done:
                action, _, _, _, _, _ = pick_action(weights, state)
                
                state, score, done, _ = self.env.step(action)
                
            total_scores.append(score)

        return np.mean(total_scores)