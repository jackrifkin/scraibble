import gym
from gym import spaces
import numpy as np
import random as rand
import util

class ScrabbleEnv(gym.Env):
    def __init__(self):
        super(ScrabbleEnv, self).__init__()

        self.current_player = 0 # 0 for p1, 1 for p2
        self.p1_score = 0
        self.p2_score = 0

        # Initialize the game board and multiplier reference
        self.board = np.full((util.BOARD_DIM, util.BOARD_DIM), -1)  # -1 for empty cells

        # Letter bag setup with 100 tiles in total
        self.letter_bag = {i: 0 for i in range(27)}  # 0 to 25 for A-Z, 26 for blanks
        
        # Fill letter_bag
        for i in range(27):
            self.letter_bag[i] = util.TILE_COUNTS[i]

        # Players' letter racks (7 letters in Scrabble)
        self.p1_letter_rack = np.zeros(7, dtype=int) - 1  # -1 indicates an empty slot
        self.p2_letter_rack = np.zeros(7, dtype=int) - 1 

        # Action space: [{"row": 0, "col": 0, "tile": 0}, {"row": 0, "col": 1, "tile": 26}, ...] (set of tile placements to form word)
        self.action_space = spaces.Sequence(
            spaces.Dict({
                "row": spaces.Discrete(15),
                "col": spaces.Discrete(15),
                "tile": spaces.Discrete(27)
            })
        )

        # Observation space: board state and letter rack
        self.observation_space = spaces.Dict({
            "board": spaces.Box(low=-1, high=26, shape=(util.BOARD_DIM, util.BOARD_DIM), dtype=int),
            "letter_rack": spaces.Box(low=-1, high=26, shape=(7,), dtype=int),
            "current_player": spaces.Discrete(2)
        })

        # Initial state
        self.reset()

    def reset(self):
        # Reset board and refill letter rack
        self.board.fill(-1)
        self.fill_letter_racks()
        return

    def draw_letter(self):
        available_letters = [k for k, v in self.letter_bag.items() if v > 0]
        if not available_letters:
            return None

        letter = rand.choice(available_letters)
        self.letter_bag[letter] -= 1
        return letter

    def fill_letter_racks(self):
        # Fill the players letter racks with random letters from the letter bag
        for i in range(7):
            if self.p1_letter_rack[i] == -1:
                letter = self.draw_letter()
                if letter is None:
                    return
                self.p1_letter_rack[i] = letter
            if self.p2_letter_rack[i] == -1:
                letter = self.draw_letter()
                if letter is None:
                    return
                self.p2_letter_rack[i] = letter

    def get_observation(self):
        return {
            "board": self.board,
            "letter_rack": self.p1_letter_rack if self.current_player == 0 else self.p2_letter_rack,
            "current_player": self.current_player
        }

    def step(self, action):
        if len(action) == 0:
            raise ValueError('Action must have at least one tile placement')
        
        # validate each tile is in player's letter_rack
        for tile_placement in action:
            tile = tile_placement["tile"]
            if not self.current_letter_rack_has_letter(tile):
                raise ValueError('Player does not have tile: ' + tile)
        
        # validate tile placements are all valid
        if not util.is_action_placement_valid(self.board, action):
            raise ValueError('Invalid action, tiles in action are not continuous')
        
        total_score = util.calculate_score_for_action(self.board, action)
        
        # add score to current player's total score
        if self.current_player == 0:
            self.p1_score += total_score
        else:
            self.p2_score += total_score

        for tile_placement in action:
            row = tile_placement["row"]
            col = tile_placement["col"]
            tile = tile_placement["tile"]

            # place tile on board
            self.board[row, col] = tile

            # remove tiles from letter_rack
            if self.current_player == 0:
                self.p1_letter_rack[np.where(self.p1_letter_rack == tile)[0][0]] = -1
            else:
                self.p2_letter_rack[np.where(self.p2_letter_rack == tile)[0][0]] = -1
        
        # refill rack
        self.fill_letter_racks()

        isDone = self.is_game_over()

        if isDone:
            if np.all(self.p1_letter_rack == -1):
                # subtract value of remaining letters from p2's letter rack
                self.p2_score -= np.sum(self.p2_letter_rack[self.p2_letter_rack != -1])
            else:
                # subtract value of remaining letters from p1's letter rack
                self.p1_score -= np.sum(self.p1_letter_rack[self.p1_letter_rack != -1])

        # rotate player
        self.current_player = 1 - self.current_player
        
        return self.get_observation(), total_score, isDone, {}

    def current_letter_rack_has_letter(self, letter):
        return np.isin(letter, self.p1_letter_rack if self.current_player == 0 else self.p2_letter_rack)

    def is_game_over(self):
        has_letters = self.draw_letter() is None
        is_p1_done = np.all(self.p1_letter_rack == -1)
        is_p2_done = np.all(self.p2_letter_rack == -1)
        return not has_letters and (is_p1_done or is_p2_done)

    def render(self, mode="human"):
        # Convert board values to displayable characters
        def cell_to_char(i, j):
            cell_value = self.board[i, j]
            letter_multiplier = util.LETTER_MULTIPLIER_POSITIONS.get((i, j))
            word_multiplier = util.WORD_MULTIPLIER_POSITIONS.get((i, j))
            if cell_value != -1:  # letter is placed
                return "_" if cell_value == 26 else chr(ord("A") + cell_value)
            elif letter_multiplier is not None and letter_multiplier != 0:  # letter multiplier cell
                return "DL" if letter_multiplier == 2 else "TL"
            elif word_multiplier is not None and word_multiplier != 0:  # word multiplier cell
                return "DW" if word_multiplier == 2 else "TW"
            else:
                return " "  # empty cell

        # Print top border
        print("   " + "  ".join([f"{i:2}" for i in range(util.BOARD_DIM)]))
        print("  +" + "---+" * util.BOARD_DIM)

        # Print board rows with grid lines
        for i in range(util.BOARD_DIM):
            row_display = [cell_to_char(i, j) for j in range(util.BOARD_DIM)]
            print(f"{i:2}| " + " | ".join(row_display) + " |")
            print("  +" + "---+" * util.BOARD_DIM)  # row separator

        # Display the letter rack with character conversion
        current_letter_rack = self.p1_letter_rack if self.current_player == 0 else self.p2_letter_rack
        letter_rack_display = [chr(ord("A") + l) if 0 <= l <= 25 else '_' for l in current_letter_rack]
        print("Letter Rack:", " ".join(letter_rack_display))


