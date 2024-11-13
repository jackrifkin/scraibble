import gym
from gym import spaces
import numpy as np
import random as rand
import gaddag as g

class ScrabbleEnv(gym.Env):
    def __init__(self):
        super(ScrabbleEnv, self).__init__()

        # Initialize the GADDAG (creating an instance)
        self.gaddag = g.Gaddag()
        ## I'm not sure how to load the json file into the GADDAG, i tried using
        # self.gaddag = g.Gaddag().load_from_json(SOWPODS.json)
        # maybe I converted the .txt file discord to a .json file wrong

        # Board dimensions
        self.rows = 15
        self.cols = 15

        # Initialize the game board and multiplier reference
        self.board = np.full((self.rows, self.cols), -1)  # -1 for empty cells
        self.multipliers = np.zeros((self.rows, self.cols))

        # Define a dictionary for multiplier cells
        self.multiplier_positions = {
            # Triple Word Score (3x)
            (0, 0): 3, (0, 7): 3, (0, 14): 3,
            (7, 0): 3, (7, 14): 3,
            (14, 0): 3, (14, 7): 3, (14, 14): 3,

            # Double Word Score (2x)
            (1, 1): 2, (2, 2): 2, (3, 3): 2, (4, 4): 2,
            (1, 13): 2, (2, 12): 2, (3, 11): 2, (4, 10): 2,
            (10, 4): 2, (11, 3): 2, (12, 2): 2, (13, 1): 2,
            (10, 10): 2, (11, 11): 2, (12, 12): 2, (13, 13): 2,

            # Triple Letter Score (3x)
            (1, 5): 3, (1, 9): 3, (5, 1): 3, (5, 13): 3,
            (9, 1): 3, (9, 13): 3, (13, 5): 3, (13, 9): 3,

            # Double Letter Score (2x)
            (0, 3): 2, (0, 11): 2, (2, 6): 2, (2, 8): 2,
            (3, 0): 2, (3, 7): 2, (3, 14): 2,
            (6, 2): 2, (6, 6): 2, (6, 8): 2, (6, 12): 2,
            (7, 3): 2, (7, 11): 2,
            (8, 2): 2, (8, 6): 2, (8, 8): 2, (8, 12): 2,
            (11, 0): 2, (11, 7): 2, (11, 14): 2,
            (12, 6): 2, (12, 8): 2, (14, 3): 2, (14, 11): 2
        }

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
                letter = rand.choice([k for k, v in self.letter_bag.items() if v > 0])
                self.letter_rack[i] = letter
                self.letter_bag[letter] -= 1

    def step(self, action):
        row, col, letter = action
        if self.board[row, col] != -1 or self.letter_rack[letter] == -1:
            return {"board": self.board, "letter_rack": self.letter_rack}, -1, False, {}

        # Place the letter on the board
        self.board[row, col] = letter
        self.letter_rack[np.where(self.letter_rack == letter)[0][0]] = -1

        # Refill the rack
        self.fill_letter_rack()

        # Calculate reward (basic for now, we can expand this)
        reward = 1  # Base reward
        return {"board": self.board, "letter_rack": self.letter_rack}, reward, False, {}

    def is_valid_word(self, word):
        """Use the GADDAG structure to check if a word is valid."""
        return self.gaddag.word_defined(word)

    def place_word(self, row, col, direction, word, rack_letter_indices):
        # Validate word using Gaddag
        if not self.is_valid_word(word):
            print(f"'{word}' is not a valid word.")
            return False

        # Convert rack_letter_indices to actual letters
        rack_letters = [self.letter_rack[i] for i in rack_letter_indices]

        # Check if the letters are in the player's rack
        for letter in word:
            if letter not in rack_letters:
                print(f"You don't have the letter '{letter}' in your rack.")
                return False

        # Place the word on the board
        for i, letter in enumerate(word):
            if direction == "right":
                target_row, target_col = row, col + i
            elif direction == "down":
                target_row, target_col = row + i, col

            # Ensure the placement is within bounds and there is no conflict
            if target_row < 0 or target_row >= self.rows or target_col < 0 or target_col >= self.cols:
                print(f"Placement of '{word}' goes out of bounds.")
                return False
            if self.board[target_row, target_col] != -1 and self.board[target_row, target_col] != ord(letter) - 65:
                print(f"Conflict at position ({target_row}, {target_col}) with an existing letter.")
                return False

            # Place the letter (or allow it to intersect)
            letter_index = ord(letter) - 65
            action = (target_row, target_col, letter_index)
            _, reward, done, _ = self.step(action)  # Call step for each letter

        return True

    def render(self, mode="human"):
        # Convert board values to displayable characters
        def cell_to_char(i, j):
            cell_value = self.board[i, j]
            if cell_value != -1:  # letter is placed
                return "_" if cell_value == 26 else chr(ord("A") + cell_value)
            elif self.multipliers[i, j] != 0:  # multiplier cell
                return str(int(self.multipliers[i, j]))
            else:
                return " "  # empty cell

        # Print top border
        print("   " + "  ".join([f"{i:2}" for i in range(self.cols)]))
        print("  +" + "---+" * self.cols)

        # Print board rows with grid lines
        for i in range(self.rows):
            row_display = [cell_to_char(i, j) for j in range(self.cols)]
            print(f"{i:2}| " + " | ".join(row_display) + " |")
            print("  +" + "---+" * self.cols)  # row separator

        # Display the letter rack with character conversion
        letter_rack_display = [chr(ord("A") + l) if 0 <= l <= 25 else '_' for l in self.letter_rack]
        print("Letter Rack:", " ".join(letter_rack_display))


