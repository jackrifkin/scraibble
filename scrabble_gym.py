import gym
from gym import spaces
import numpy as np


class ScrabbleEnv(gym.Env):
    def __init__(self):
        super(ScrabbleEnv, self).__init__()

        # Board dimensions
        self.rows = 15
        self.cols = 15

        # Initialize the game board and multiplier reference
        self.board = np.full((self.rows, self.cols), -1)  # -1 for empty cells
        self.multipliers = np.zeros((self.rows, self.cols))

        # Define a dictionary for multiplier cells
        self.multiplier_positions = {(0, 0): 3, (7, 7): 2}  # example positions
        for (x, y), multiplier in self.multiplier_positions.items():
            self.multipliers[x, y] = multiplier

        # Letter bag setup with 100 tiles in total
        self.letter_bag = {i: 0 for i in range(27)}  # 0 to 25 for A-Z, 26 for blanks
        # Fill in actual tile counts (for simplicity, equal distribution here)
        for i in range(26):
            self.letter_bag[i] = 4
        self.letter_bag[26] = 2  # Two blank tiles

        # Player's letter rack (7 letters in Scrabble)
        self.letter_rack = np.zeros(7, dtype=int) - 1  # -1 indicates an empty slot

        # Action space: row, col, letter
        self.action_space = spaces.MultiDiscrete([self.rows, self.cols, 27])

        # Observation space: board state and letter rack
        self.observation_space = spaces.Dict({
            "board": spaces.Box(low=-1, high=26, shape=(self.rows, self.cols), dtype=int),
            "letter_rack": spaces.Box(low=-1, high=26, shape=(7,), dtype=int)
        })

        # Initial state
        self.reset()

    def reset(self):
        # Reset board and refill letter rack
        self.board.fill(-1)
        self.fill_letter_rack()
        return {"board": self.board, "letter_rack": self.letter_rack}

    def fill_letter_rack(self):
        # Fill the player's rack with random letters from the letter bag
        for i in range(7):
            if self.letter_rack[i] == -1:
                self.letter_rack[i] = np.random.choice([k for k, v in self.letter_bag.items() if v > 0])
                self.letter_bag[self.letter_rack[i]] -= 1

    def step(self, action):
        row, col, letter = action

        # Validate action
        if self.board[row, col] != -1 or self.letter_rack[letter] == -1:
            return {"board": self.board, "letter_rack": self.letter_rack}, -1, False, {}

        # Place the letter on the board
        self.board[row, col] = letter
        self.letter_rack[np.where(self.letter_rack == letter)[0][0]] = -1

        # Refill the rack
        self.fill_letter_rack()

        # Calculate reward (multiplier logic)
        reward = 1  # base score, expand as needed

        return {"board": self.board, "letter_rack": self.letter_rack}, reward, False, {}

    def render(self, mode="human"):
        print("Board:")
        print(self.board)
        print("Letter Rack:", self.letter_rack)


# Test environment
env = ScrabbleEnv()
obs = env.reset()
print("Initial Observation:", obs)
