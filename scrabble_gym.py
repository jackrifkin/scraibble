import gym
from gym import spaces
from gaddag import DELIM, END_WORD_DELIM
import numpy as np
import random as rand
import util
from util import DIRECTION, GADDAG

class ScrabbleEnv(gym.Env):
    def __init__(self):
        super(ScrabbleEnv, self).__init__()

        self.current_player = 0 # 0 for p1, 1 for p2
        self.p1_score = 0
        self.p2_score = 0

        # Initialize the game board
        self.board = np.full((util.BOARD_DIM, util.BOARD_DIM), -1)  # -1 for empty cells
        
        # down crosssets of tile at (7,7): crosssets[7,7]["down"]
        # across crosssets of tile at (7,7): crosssets[7,7]["across"]
        self.cross_sets = np.tile(np.arange(26, dtype=int), (util.BOARD_DIM, util.BOARD_DIM, 2, 1))

        # Letter bag setup with 100 tiles in total
        self.letter_bag = {i: 0 for i in range(27)}  # 0 to 25 for A-Z, 26 for blanks
        
        # Fill letter_bag
        for i in range(27):
            self.letter_bag[i] = util.TILE_COUNTS[i]

        # Players' letter racks (7 letters in Scrabble)
        self.p1_letter_rack = np.zeros(7, dtype=int) - 1  # -1 indicates an empty slot
        self.p2_letter_rack = np.zeros(7, dtype=int) - 1

        # fill racks
        self.fill_letter_racks()

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
            # cross_sets in the observation space are an array of max 26 elements representing the letters that can form
            # valid crosswords
            # example: cross_sets[7, 7, 0] will give the 'down' cross_set of the tile at pos: (7, 7)
            "cross_sets": spaces.Box(low=0, high=26, shape=(util.BOARD_DIM, util.BOARD_DIM, 2, 26), dtype=int),
            "letter_rack": spaces.Box(low=-1, high=26, shape=(7,), dtype=int),
            "current_player": spaces.Discrete(2)
        })

        print('\nInitialized ScrabbleEnv\n')

    def reset(self):
        self.__init__()
        return self.get_observation()

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

    def fill_cross_set_array(self, cross_set):
        padded_values = np.pad(cross_set, (0, max(0, 26 - len(cross_set))), constant_values=-1)
        return padded_values[:26]

    def get_observation(self):
        cross_sets = np.full((util.BOARD_DIM, util.BOARD_DIM, 2, 26), -1, dtype=int)
        
        for row in range(util.BOARD_DIM):
            for col in range(util.BOARD_DIM):
                cross_set = self.cross_sets[row, col]
                cross_sets[row, col, 0] = self.fill_cross_set_array(cross_set[DIRECTION.DOWN.value])
                cross_sets[row, col, 1] = self.fill_cross_set_array(cross_set[DIRECTION.ACROSS.value])

        return {
            "board": self.board,
            "cross_sets": cross_sets,
            "letter_rack": self.p1_letter_rack if self.current_player == 0 else self.p2_letter_rack,
            "current_player": self.current_player
        }
    
    def pass_move(self):
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
        
        return self.get_observation(), 0, isDone, {}

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
        
        # update cross_sets
        self.update_all_crosssets_affected_by_move(action)

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
        
    
    ## can get rightmost (1) or leftmost (-1) letter (depending on step)
    def get_last_letter(self, start_coordinate, direction, step):
        curr = start_coordinate
        next = util.offset(start_coordinate, direction, step)
        row, col = next[0], next[1]
        while util.pos_in_bounds((row, col)) and self.board[row, col] != -1:
            curr = next
            next = util.offset(curr, direction, step)
            row, col = next[0], next[1]
        return curr
    
    def update_all_crosssets_affected_by_move(self, action):
        # determine direction and start coord of action
        direction = DIRECTION.ACROSS if all(letter['row'] == action[0]['row'] for letter in action) else DIRECTION.DOWN
        start_coordinate = self.get_last_letter((action[0]['row'], action[0]['col']), direction, -1)
        end_coordinate = self.get_last_letter((action[0]['row'], action[0]['col']), direction, 1)

        # update cross_sets on ends of action
        self.update_cross_set(start_coordinate, direction)

        other_direction = DIRECTION.ACROSS if direction == DIRECTION.DOWN else DIRECTION.DOWN
        word_length = (end_coordinate[0] - start_coordinate[0] if direction == DIRECTION.DOWN else end_coordinate[1] - start_coordinate[1]) + 1
        # update cross_sets perpendicular to action
        for i in range(word_length):
            delta = (1, 0) if direction == DIRECTION.DOWN else (0, 1)
            curr_coord = (start_coordinate[0] + delta[0] * i, start_coordinate[1] + delta[1] * i)
            self.update_cross_set(curr_coord, other_direction)


    def update_cross_set(self, start_coordinate, direction):
        ## helpers:
        ## returns boolean
        def check_candidate(coord, candidate, direction, step):
            last_arc = candidate
            state = candidate.destination
            row, col = util.offset(coord, direction, step)
            while util.pos_in_bounds((row, col)) and self.board[row, col] != -1:
                coord = (row, col)
                tile = util.char_idx_to_char(self.board[coord])
                last_arc = state.arcs[tile] if tile in state.arcs else None
                if not last_arc:
                    return False
                state = last_arc.destination
                row, col = util.offset(coord, direction, step)
            return tile in last_arc.letter_set
        
        # clears existing crossets next to the word being made
        def clear_existing_crosssets(coord, direction):
            rightmost_coord = self.get_last_letter(coord, direction, 1)
            right_empty = util.offset(rightmost_coord, direction, 1)
            if (self.board[right_empty[0], right_empty[1]]) == -1:
                self.cross_sets[right_empty[0], right_empty[1]][direction.value] = np.zeros(26) - 1

            leftmost_coord = self.get_last_letter(coord, direction, -1)
            left_empty = util.offset(leftmost_coord, direction, -1)
            if (self.board[left_empty[0], left_empty[1]]) == -1:
                self.cross_sets[left_empty[0], left_empty[1]][direction.value] = np.zeros(26) - 1
        
        if not util.pos_in_bounds(start_coordinate) or self.board[start_coordinate] == -1:
            return # do not do anything
        end_coordinate = self.get_last_letter(start_coordinate, direction, 1)

        curr_coord = end_coordinate
        curr_char = util.char_idx_to_char(self.board[curr_coord])
        last_state = GADDAG.root
        state = last_state.get_next(curr_char)
        next_coord = util.offset(curr_coord, direction, -1)
        while util.pos_in_bounds(next_coord) and self.board[next_coord] != -1:
            curr_coord = next_coord
            curr_char = util.char_idx_to_char(self.board[curr_coord])
            last_state = state
            state = last_state.get_next(curr_char)
            if not state and not last_state.__contains__(END_WORD_DELIM): # there is an invalid word on the board
                clear_existing_crosssets(start_coordinate, direction)
                return
            next_coord = util.offset(curr_coord, direction, -1)
        state = last_state
        
        # the start and end of the word
        right_square = util.offset(end_coordinate, direction, 1)
        left_square = util.offset(curr_coord, direction, -1)

        # special edge case where there is a empty square with tiles on both sides (APE case)
        left_of_left = util.offset(left_square, direction, -1)
        right_of_right = util.offset(right_square, direction, 1)

        curr_char = util.char_idx_to_char(self.board[curr_coord])
        if util.pos_in_bounds(left_of_left) and self.board[left_of_left] != -1:
            candidates = (arc for arc in state if arc.character != DELIM)
            cross_set_characters = [
                candidate.character 
                for candidate in candidates 
                if check_candidate(left_square, candidate, direction, -1)
            ]
            cross_set_indices = map(util.char_to_char_idx, cross_set_characters)
            cross_set = list(cross_set_indices)
            self.cross_sets[left_square][direction.value] = util.create_cross_set_np_array(cross_set)
        elif util.pos_in_bounds(left_square):
            letter_set = last_state.get_arc(curr_char).letter_set if last_state.get_arc(curr_char) else []
            next_state = last_state.get_next(curr_char)
            cross_set = filter(lambda letter: 
                               next_state.__contains__(letter)
                               and next_state.get_arc(letter)
                               and END_WORD_DELIM in next_state.get_arc(letter).letter_set, 
                               letter_set)
            cross_set = list(map(util.char_to_char_idx, cross_set))
            self.cross_sets[left_square][direction.value] = util.create_cross_set_np_array(cross_set)
        
        ## right side
        if util.pos_in_bounds(right_of_right) and self.board[right_of_right] != -1:
            end_state = state.get_next(DELIM)
            candidates = (arc for arc in end_state if arc != DELIM) if end_state else {}
            cross_set_characters = [
                candidate.character 
                for candidate in candidates 
                if check_candidate(right_square, candidate, direction, 1)
            ]
            cross_set_indices = map(util.char_to_char_idx, cross_set_characters)
            cross_set = list(cross_set_indices)
            self.cross_sets[right_square][direction.value] = util.create_cross_set_np_array(cross_set)
        elif util.pos_in_bounds(right_square):
            letter_set = state.get_arc(curr_char).letter_set if state.get_arc(curr_char) else []
            next_state = state.get_next(curr_char)
            cross_set = []
            if DELIM in letter_set:
                delim_state = next_state.get_next(DELIM)
                letter_set = delim_state.letter_set
                cross_set = filter(lambda letter:
                                    delim_state.get_arc(letter)
                                    and END_WORD_DELIM in delim_state.get_arc(letter).letter_set, 
                                    letter_set)
                cross_set = list(map(util.char_to_char_idx, cross_set))
            self.cross_sets[right_square][direction.value] = util.create_cross_set_np_array(cross_set)

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
                return f"\033[91m {util.char_idx_to_char(cell_value)} \033[0m"
            elif letter_multiplier is not None and letter_multiplier != 0:  # letter multiplier cell
                return "DL " if letter_multiplier == 2 else "TL "
            elif word_multiplier is not None and word_multiplier != 0:  # word multiplier cell
                return "DW " if word_multiplier == 2 else "TW "
            else:
                return "   "

        # Print top border
        print("    " + "    ".join([f"{i:2}" for i in range(util.BOARD_DIM)]))
        print("  +" + "-----+" * util.BOARD_DIM)

        # Print board rows with grid lines
        for i in range(util.BOARD_DIM):
            row_display = [cell_to_char(i, j) for j in range(util.BOARD_DIM)]
            print(f"{i:2}| " + " | ".join(row_display) + " |")
            print("  +" + "-----+" * util.BOARD_DIM)  # row separator

        # Display the letter rack with character conversion
        current_letter_rack = self.p1_letter_rack if self.current_player == 0 else self.p2_letter_rack
        letter_rack_display = [util.char_idx_to_char(l) for l in current_letter_rack]
        print("Letter Rack:", " ".join(letter_rack_display))



    ## my stuff Daniel
    def generate_moves_p1(self):
        return util.generate_possible_moves(self.board, self.p1_letter_rack, self.cross_sets)


    def generate_moves_p2(self):
        return util.generate_possible_moves(self.board, self.p2_letter_rack, self.cross_sets)