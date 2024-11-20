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

        self.current_player = 0 # 0 for p1, 1 for p2
        self.p1_score = 0
        self.p2_score = 0

        # Initialize the game board and multiplier reference
        self.board = np.full((self.rows, self.cols), -1)  # -1 for empty cells

        self.word_multiplier_positions = {
            # Triple Word Score (3x)
            (0, 0): 3, (0, 7): 3, (0, 14): 3,
            (7, 0): 3, (7, 14): 3,
            (14, 0): 3, (14, 7): 3, (14, 14): 3,

            # Double Word Score (2x)
            (1, 1): 2, (2, 2): 2, (3, 3): 2, (4, 4): 2,
            (1, 13): 2, (2, 12): 2, (3, 11): 2, (4, 10): 2,
            (10, 4): 2, (11, 3): 2, (12, 2): 2, (13, 1): 2,
            (10, 10): 2, (11, 11): 2, (12, 12): 2, (13, 13): 2
        }

        self.letter_multiplier_positions = {
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

        # Letter bag setup with 100 tiles in total
        self.letter_bag = {i: 0 for i in range(27)}  # 0 to 25 for A-Z, 26 for blanks
        
        # actual # of tiles per letter in Scrabble:
        tile_counts = [9, 2, 2, 4, 12, 2, 3, 2, 9, 1, 1, 4, 2, 6, 8, 2, 1, 6, 4, 6, 4, 2, 2, 1, 2, 1, 2]
        # Fill letter_bag
        for i in range(27):
            self.letter_bag[i] = tile_counts[i]

        self.tile_values = [1, 3, 3, 2, 1, 4, 2, 4, 1, 8, 5, 1, 3, 1, 1, 3, 10, 1, 1, 1, 1, 4, 4, 8, 4, 10, 0]

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
            "board": spaces.Box(low=-1, high=26, shape=(self.rows, self.cols), dtype=int),
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
        
        for tile_placement in action:
            row = tile_placement["row"]
            col = tile_placement["col"]
            tile = tile_placement["tile"]
            # validate placement of tile (adjacent to some other tile (or on middle space))
            if not self.is_tile_placement_valid(row, col, action):
                raise ValueError('Invalid tile placement')
            # validate tile is in player's letter_rack
            if not self.current_letter_rack_has_letter(tile):
                raise ValueError('Player does not have tile: ' + tile)
        
        # validate tiles are all in same row or same col
        if not self.is_action_continuous(action):
            raise ValueError('Invalid action, tiles in action are not continuous')
        
        # get all words made by new tiles (any sequence of tiles in the same row or col as each tile that is adjacent to each tile)
        # proposed_words = {'word': [[x,y,4,1], [x,y,6,1], [x,y,6,0], ...], 'word2': [[x,y,6,0], [x,y,4,1], [x,y,19,1], ...]}
        # [row, col, tile, (0 if existing tile, 1 if placed)]
        proposed_words = self.get_words_made_by_action(action)

        # validate each word
        for word in proposed_words.keys():
            if not self.is_valid_word(word):
                raise ValueError('Invalid word: ' + word)

        # calculate score
        total_score = 0
        for _, letter_placements in proposed_words.items():
            total_score += self.calculate_score_for_word(letter_placements)

        # bingo (all 7 tiles from rack) = +50 points
        if len(action) == 7:
            total_score += 50
        
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

    def is_valid_word(self, word):
        """Use the GADDAG structure to check if a word is valid."""
        return self.gaddag.word_defined(word)
    
    # checks if the given row and col are adjacent to an existing tile or a different tile from the action
    def is_tile_placement_valid(self, row, col, action):
        result = False
        if (row >= 0 and row < self.rows) and (col >= 0 and col < self.cols):
            adjacent_positions = [
                (row - 1, col),
                (row + 1, col),
                (row, col - 1),
                (row, col + 1)
            ]
            
            for adj_row, adj_col in adjacent_positions:
                if 0 <= adj_row < self.rows and 0 <= adj_col < self.cols:
                    if self.board[adj_row, adj_col] != -1:
                        result = True # there is an adjacent tile
                        break
            
            # if there are no existing adjacent tiles, check other tiles from the current action
            if not result:
                for tile_placement in action:
                    other_row = tile_placement["row"]
                    other_col = tile_placement["col"]
                    for adj_row, adj_col in adjacent_positions:
                        if adj_row == other_row and adj_col == other_col:
                            result = True # one of the tiles from the actions is adjacent
                            break
                    if result:
                        break
        return result

    def current_letter_rack_has_letter(self, letter):
        return np.isin(letter, self.p1_letter_rack if self.current_player == 0 else self.p2_letter_rack)
    
    # checks if all tile placements in the action are either all in the same row or all in the same col
    # and if they form an unbroken continuous word
    def is_action_continuous(self, action):
        rows = np.array([tile["row"] for tile in action])
        cols = np.array([tile["col"] for tile in action])
        
        in_same_row = np.all(rows == rows[0])
        in_same_col = np.all(cols == cols[0])
        tiles_inline = in_same_row or in_same_col

        continuous = True
        if in_same_row:
            first_col = np.min(cols)
            last_col = np.max(cols)
            row = rows[0]
            # starting at first_col, first empty tile to the right must have col > last_col
            # (if not, that means there is an empty space between placed tiles which is invalid)
            current_col = first_col
            while current_col < last_col:
                next_col = current_col + 1
                if next_col > 14:
                    break # reached the end of the board, tiles are continuous
                # if next_col is empty and won't be filled by a letter in the action, tiles are not continuous
                if self.board[row][next_col] == -1 and not np.isin(next_col, cols):
                    continuous = False
                    break
                current_col += 1
        elif in_same_col:
            first_row = np.min(rows)
            last_row = np.max(rows)
            col = cols[0]
            # starting at first_row, first empty tile below must have row > last_row
            # (if not, that means there is an empty space between placed tiles which is invalid)
            current_row = first_row
            while current_row < last_row:
                next_row = current_row + 1
                if next_row > 14:
                    break # reached the end of the board, tiles are continuous
                # if next_row is empty and won't be filled by a letter in the action, tiles are not continuous
                if self.board[next_row][col] == -1 and not np.isin(next_row, rows):
                    continuous = False
                    break
                current_col += 1

        return tiles_inline and continuous
    
    def get_tile_val_from_action(self, action, row, col):
        for tile_placement in action:
            if tile_placement["row"] == row and tile_placement["col"] == col:
                return tile_placement["tile"]
        return None
    
    def get_continuous_word(self, action, first_placed_letter_idx, last_placed_letter_idx, axis_idx, is_vertical):
        rows = np.array([tile["row"] for tile in action])
        cols = np.array([tile["col"] for tile in action])
        # find farthest left/up tile
        leftmost_upmost = first_placed_letter_idx
        while leftmost_upmost > 0:
            next_cell = self.board[axis_idx][leftmost_upmost] if is_vertical else self.board[leftmost_upmost][axis_idx]
            if next_cell == -1:
                # if next cell (up or left) is empty and not one of the placed letters, 
                # then the previous cell was the farthest left/up
                if not np.isin(leftmost_upmost, rows if is_vertical else cols):
                    leftmost_upmost = leftmost_upmost + 1
                    break
            leftmost_upmost -= 1
        # find farthest right/down tile
        rightmost_downmost = last_placed_letter_idx
        while rightmost_downmost < (self.rows - 1):
            next_cell = self.board[axis_idx][rightmost_downmost] if is_vertical else self.board[rightmost_downmost][axis_idx]
            if next_cell == -1:
                # if next cell (down or right) is empty and not one of the placed letters, 
                # then the previous cell was the farthest right/down
                if not np.isin(rightmost_downmost, rows if is_vertical else cols):
                    rightmost_downmost = rightmost_downmost - 1
                    break
            rightmost_downmost += 1
        # get word made by tiles
        word_key = ""
        # store the [row, col, letter_idx] for each letter in the word
        letter_entries = []
        for i in range(rightmost_downmost - leftmost_upmost + 1):
            cell_value = self.board[leftmost_upmost + i][axis_idx] if is_vertical else self.board[axis_idx][leftmost_upmost + i]
            was_placed = 0
            if cell_value == -1:
                was_placed = 1
                if is_vertical:
                    cell_value = self.get_tile_val_from_action(action, leftmost_upmost + i, axis_idx)
                else:
                    cell_value = self.get_tile_val_from_action(action, axis_idx, leftmost_upmost + i)
            word_key += ("_" if cell_value == 26 else chr(ord("A") + cell_value))
            if is_vertical:
                letter_entries.append([leftmost_upmost + i, axis_idx, cell_value, was_placed])
            else:
                letter_entries.append([leftmost_upmost + i, axis_idx, cell_value, was_placed])

        return word_key, letter_entries

    
    # gets all words made by the placed tiles, assuming tiles are continuous (verified by above function)
    def get_words_made_by_action(self, action):
        rows = np.array([tile["row"] for tile in action])
        cols = np.array([tile["col"] for tile in action])
        in_same_row = np.all(rows == rows[0])
        in_same_col = np.all(cols = cols[0])
        words = {}

        if in_same_row:
            # find horizontal word made by row of letters
            horizontal_word, horizontal_letters = self.get_continuous_word(action, np.min(cols), np.max(cols), rows[0], False)
            words[horizontal_word] = horizontal_letters

            # find any vertical words made by each letter in the row
            for tile_placement in action:
                vertical_word, vertical_letters = self.get_continuous_word(action, rows[0], rows[0], tile_placement["col"], True)
                # if the vertical word only contains the letter, the word/letter should not be counted twice
                if len(vertical_word) > 1:
                    words[vertical_word] = vertical_letters
        elif in_same_col:
            # find vertical word made by col of letters
            vertical_word, vertical_letters = self.get_continuous_word(action, np.min(rows), np.max(rows), cols[0], True)
            words[vertical_word] = vertical_letters

            # find any horizontal words made by each letter in the col
            for tile_placement in action:
                horizontal_word, horizontal_letters = self.get_continuous_word(action, cols[0], cols[0], tile_placement["row"], False)
                # if the vertical word only contains the letter, the word/letter should not be counted twice
                if len(horizontal_word) > 1:
                    words[horizontal_word] = horizontal_letters
        return words
    
    def calculate_score_for_word(self, word):
        total_score = 0
        total_word_multiplier = 1

        for letter in word:
            row = letter[0]
            col = letter[1]
            tile = letter[2]
            is_new = bool(letter[3])

            if is_new:
                # multiply letter score by letter multiplier (if any)
                letter_multiplier = self.letter_multiplier_positions.get((row, col), 1)
                total_score += self.tile_values[tile] * letter_multiplier

                # include word multiplier in total_word_multiplier (if any)
                total_word_multiplier *= self.word_multiplier_positions.get((row, col), 1)
            else:
                total_score += self.tile_values[tile]
        return total_score * total_word_multiplier

    def is_game_over(self):
        has_letters = self.draw_letter() is None
        is_p1_done = np.all(self.p1_letter_rack == -1)
        is_p2_done = np.all(self.p2_letter_rack == -1)
        return not has_letters and (is_p1_done or is_p2_done)

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
        current_letter_rack = self.p1_letter_rack if self.current_player == 0 else self.p2_letter_rack
        letter_rack_display = [chr(ord("A") + l) if 0 <= l <= 25 else '_' for l in current_letter_rack]
        print("Letter Rack:", " ".join(letter_rack_display))


