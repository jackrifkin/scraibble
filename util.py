from enum import Enum
import numpy as np
import new_gaddag as g
import string

# CONSTANTS
WORD_MULTIPLIER_POSITIONS = {
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

LETTER_MULTIPLIER_POSITIONS = {
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

TILE_VALUES = [1, 3, 3, 2, 1, 4, 2, 4, 1, 8, 5, 1, 3, 1, 1, 3, 10, 1, 1, 1, 1, 4, 4, 8, 4, 10, 0]

# actual # of tiles per letter in Scrabble:
TILE_COUNTS = [9, 2, 2, 4, 12, 2, 3, 2, 9, 1, 1, 4, 2, 6, 8, 2, 1, 6, 4, 6, 4, 2, 2, 1, 2, 1, 0] # TODO: add blanks once supported (currently set to 0)

BOARD_DIM = 15

# directions for cross sets
class DIRECTION(Enum):
  ACROSS = 1
  DOWN = 0

GADDAG = g.Gaddag()

# Initialize the GADDAG (creating an instance)
def init_gaddag():
  GADDAG.construct_from_txt("SOWPODS.txt")

# END CONSTANTS

def is_valid_word(word):
  return GADDAG.is_word_in_gaddag(word)

def is_action_placement_valid(board, action):
  tile_placements_valid = True
  for tile_placement in action:
    row = tile_placement["row"]
    col = tile_placement["col"]
    # validate placement of tile (adjacent to some other tile (or on middle space))
    if not is_tile_placement_valid(board, row, col, action):
      tile_placements_valid = False

  return is_action_continuous(board, action) and tile_placements_valid

# checks if the given row and col are adjacent to an existing tile or a different tile from the action
def is_tile_placement_valid(board, row, col, action):
  result = False
  if (row >= 0 and row < BOARD_DIM) and (col >= 0 and col < BOARD_DIM):
    # if there is already a tile on the board, invalid
    if board[row, col] != -1:
      return False

    adjacent_positions = [
      (row - 1, col),
      (row + 1, col),
      (row, col - 1),
      (row, col + 1)
    ]
    
    for adj_row, adj_col in adjacent_positions:
      if 0 <= adj_row < BOARD_DIM and 0 <= adj_col < BOARD_DIM:
        if board[adj_row, adj_col] != -1:
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

# checks if all tile placements in the action are either all in the same row or all in the same col
# and if they form an unbroken continuous word
def is_action_continuous(board, action):
  rows = np.array([tile["row"] for tile in action])
  cols = np.array([tile["col"] for tile in action])
  
  in_same_row = np.all(rows == rows[0])
  in_same_col = np.all(cols == cols[0])
  tiles_inline = in_same_row or in_same_col

  # place tiles on board copy for testing continuity
  new_board = board.copy()
  for tile_placement in action:
    row = tile_placement["row"]
    col = tile_placement["col"]
    tile = tile_placement["tile"]
    new_board[row, col] = tile

  continuous = True
  if in_same_row:
    first_col = np.min(cols)
    last_col = np.max(cols)

    board_rows, board_cols = new_board.shape

    # mask board for only tiles between first_col and last_col
    tiles_in_action = np.zeros((board_rows, board_cols), dtype=bool)
    tiles_in_action[rows[0], first_col:last_col + 1] = True
    
    # all tiles in between must be non-empty
    continuous = np.all(new_board[tiles_in_action] != -1)
  elif in_same_col:
    first_row = np.min(rows)
    last_row = np.max(rows)
    
    board_rows, board_cols = new_board.shape

    # mask board for only tiles between first_row and last_row
    tiles_in_action = np.zeros((board_rows, board_cols), dtype=bool)
    tiles_in_action[first_row:last_row + 1, cols[0]] = True
    
    # all tiles in between must be non-empty
    continuous = np.all(new_board[tiles_in_action] != -1)

  return tiles_inline and continuous

def calculate_score_for_action(board, action):
  # get all words made by new tiles (any sequence of tiles in the same row or col as each tile that is adjacent to each tile)
  # proposed_words = {'word': [[x,y,4,1], [x,y,6,1], [x,y,6,0], ...], 'word2': [[x,y,6,0], [x,y,4,1], [x,y,19,1], ...]}
  # [row, col, tile, (0 if existing tile, 1 if placed)]
  proposed_words = get_words_made_by_action(board, action)

  # validate each word
  for word in proposed_words.keys():
    if not is_valid_word(word):
      raise ValueError('Invalid word: ' + word)

  # calculate score
  total_score = 0
  for _, letter_placements in proposed_words.items():
    total_score += calculate_score_for_word(letter_placements)

  # bingo (all 7 tiles from rack) = +50 points
  if len(action) == 7:
    total_score += 50

  return total_score

def get_tile_val_from_action(action, row, col):
  for tile_placement in action:
    if tile_placement["row"] == row and tile_placement["col"] == col:
      return tile_placement["tile"]
  return None

def get_continuous_word(board, action, first_placed_letter_idx, last_placed_letter_idx, is_vertical, row=None, col=None):
  rows = np.array([tile["row"] for tile in action])
  cols = np.array([tile["col"] for tile in action])
  row = row if row != None else rows[0]
  col = col if col != None else cols[0]

  new_board = board.copy()
  for tile_placement in action:
    row_to_place = tile_placement["row"]
    col_to_place = tile_placement["col"]
    tile_to_place = tile_placement["tile"]
    new_board[row_to_place, col_to_place] = tile_to_place
  
  board_rows, board_cols = new_board.shape
  tiles_in_word = np.zeros((board_rows, board_cols), dtype=bool)

  if is_vertical:
    # create mask of tiles between given range
    tiles_in_word[first_placed_letter_idx:last_placed_letter_idx + 1, col] = True

    current_row_idx = first_placed_letter_idx
    # while board[row--, col] is not empty, add that tile to tiles_in_word mask
    while (current_row_idx >= 0 and new_board[current_row_idx, col] != -1):
      tiles_in_word[current_row_idx, col] = True
      current_row_idx -= 1

    current_row_idx = last_placed_letter_idx
    # while board[row++, col] is not empty, add that tile to tiles_in_word mask
    while (current_row_idx <= BOARD_DIM - 1 and new_board[current_row_idx, col] != -1):
      tiles_in_word[current_row_idx, col] = True
      current_row_idx += 1
  else:
    # create mask of tiles between given range
    tiles_in_word[row, first_placed_letter_idx:last_placed_letter_idx + 1] = True

    current_col_idx = first_placed_letter_idx
    # while board[row, col--] is not empty, add that tile to tiles_in_word mask
    while (current_col_idx >= 0 and new_board[row, current_col_idx] != -1):
      tiles_in_word[row, current_col_idx] = True
      current_col_idx -= 1
        
    current_col_idx = last_placed_letter_idx
    # while board[row, col++] is not empty, add that tile to tiles_in_word mask
    while (current_col_idx <= BOARD_DIM - 1 and new_board[row, current_col_idx] != -1):
      tiles_in_word[row, current_col_idx] = True
      current_col_idx += 1

  # get word made by tiles
  word_key = ""
  # store the [row, col, letter_idx, was_placed] for each letter in the word
  letter_entries = []

  for row in range(tiles_in_word.shape[0]):
    for col in range(tiles_in_word.shape[1]):
      if tiles_in_word[row, col]:
        was_placed = 0
        cell_value = board[row, col]
        if cell_value == -1: # tile was placed during this action
          was_placed = 1
          cell_value = new_board[row, col]
        word_key += (char_idx_to_char(cell_value))
        letter_entries.append([row, col, cell_value, was_placed])

  return word_key, letter_entries


# gets all words made by the placed tiles, assuming tiles are continuous (verified by above function)
def get_words_made_by_action(board, action):
  rows = np.array([tile["row"] for tile in action])
  cols = np.array([tile["col"] for tile in action])
  in_same_row = np.all(rows == rows[0])
  in_same_col = np.all(cols == cols[0])
  words = {}

  if in_same_row:
    # find horizontal word made by row of letters
    horizontal_word, horizontal_letters = get_continuous_word(board, action, np.min(cols), np.max(cols), False)
    words[horizontal_word] = horizontal_letters

    # find any vertical words made by each letter in the row
    for i in range(len(action)):
      vertical_word, vertical_letters = get_continuous_word(board, action, rows[i], rows[i], True, row=rows[0], col=cols[i])
      # if the vertical word only contains the letter, the word/letter should not be counted twice
      if len(vertical_word) > 1:
        words[vertical_word] = vertical_letters
  elif in_same_col:
    # find vertical word made by col of letters
    vertical_word, vertical_letters = get_continuous_word(board, action, np.min(rows), np.max(rows), True)
    words[vertical_word] = vertical_letters

    # find any horizontal words made by each letter in the col
    for i in range(len(action)):
      horizontal_word, horizontal_letters = get_continuous_word(board, action, cols[i], cols[i], False, row=rows[i], col=cols[0])
      # if the vertical word only contains the letter, the word/letter should not be counted twice
      if len(horizontal_word) > 1:
        words[horizontal_word] = horizontal_letters
  return words

def calculate_score_for_word(word):
  total_score = 0
  total_word_multiplier = 1

  for letter in word:
    row = letter[0]
    col = letter[1]
    tile = letter[2]
    is_new = bool(letter[3])

    if is_new:
      # multiply letter score by letter multiplier (if any)
      letter_multiplier = LETTER_MULTIPLIER_POSITIONS.get((row, col), 1)
      total_score += TILE_VALUES[tile] * letter_multiplier

      # include word multiplier in total_word_multiplier (if any)
      total_word_multiplier *= WORD_MULTIPLIER_POSITIONS.get((row, col), 1)
    else:
      total_score += TILE_VALUES[tile]
  return total_score * total_word_multiplier

def offset(coord, direction, offset):
    res = None
    if direction == DIRECTION.ACROSS:
        res = coord[0], coord[1] + offset # offset the column
    elif direction == DIRECTION.DOWN:
        res = coord[0] + offset, coord[1] # offset the row
    else:
        return TypeError("INVALID DIRECTION SPECIFIED")
    return res

def char_idx_to_char(char_idx):
  return "_" if char_idx == 26 else chr(ord("A") + char_idx)

def char_to_char_idx(char):
  if char == '_':
    return 26
  return ord(char) - 65 

def pos_in_bounds(pos):
    row, col = pos
    return 0 <= row < BOARD_DIM and 0 <= col < BOARD_DIM

def generate_possible_moves(board, rack, cross_sets):
  """
  Generates all possible unique actions (valid word placements) given a board state and a rack of letters.
  Inspired by anchor-based recursive move generation.
  
  Logic derived from Steven A. Gordon's "A Faster Scrabble Move Generation Algorithm"
  https://ericsink.com/downloads/faster-scrabble-gordon.pdf 

  :param board: A 2D numpy array representing the Scrabble board state. -1 indicates empty tiles.
  :param rack: A numpy array of integers representing the rack (-1 = emptpy, 0 = A, ..., 25 = Z, 26 = blank).
  :param cross_sets: A 4D array representing the across and down cross_sets for each tile on the board.
  :return: A list of unique actions, where each action is a list of dictionaries describing tile placements.
            Each dictionary contains 'row', 'col', and 'tile' keys.
  """
  actions = []
  anchors_used = set()

  def gen(pos, word, rack, arc: g.Arc, new_tiles, blanks, anchor, direction):
    rack = rack.copy()
    current_position = offset(anchor, direction, pos)
    tile = board[current_position]

    if tile != -1:
      go_on(pos, char_idx_to_char(tile), word, rack, arc.get_next(char_idx_to_char(tile)), arc, new_tiles, blanks, anchor, direction)
    elif np.any(rack != -1):
      other_direction = DIRECTION.ACROSS if direction == DIRECTION.DOWN else DIRECTION.DOWN
      cross_set = cross_sets[current_position][other_direction.value]
      for letter in (x for x in set(rack) if x in cross_set or np.all(cross_set == -1)):
        tmp_rack = rack.copy()
        idx_to_remove = np.where(tmp_rack == letter)[0]
        if idx_to_remove.size > 0:
          np.delete(tmp_rack, idx_to_remove[0])
        else:
          raise RuntimeError('Letter rack does not contain letter option')
        tmp_new_tiles = new_tiles.copy()
        tmp_new_tiles.append(current_position)
        letter_char = char_idx_to_char(letter)
        go_on(pos, letter_char, word, tmp_rack, arc.get_next(letter_char), arc, tmp_new_tiles, blanks, anchor, direction)
      if 26 in rack:
        for letter in (x for x in set(string.ascii_uppercase) if x in cross_set or np.all(cross_set == -1)):
          tmp_rack = rack.copy()
          idx_to_remove = np.where(tmp_rack == letter)[0]
          if idx_to_remove.size > 0:
            np.delete(tmp_rack, idx_to_remove[0])
          else:
            raise RuntimeError('Letter rack does not contain letter option')
          tmp_new_tiles = new_tiles.copy()
          tmp_new_tiles.append(current_position)
          tmp_blanks = blanks.copy()
          tmp_blanks.append(current_position)
          letter_char = char_idx_to_char(letter)
          go_on(pos, letter_char, word, tmp_rack, arc.get_next(letter_char), arc, tmp_new_tiles, tmp_blanks, anchor, direction)

  def go_on(pos, char, word, rack, new_arc: g.Arc, old_arc: g.Arc, new_tiles, blanks, anchor, direction):
    directly_left_pos = offset(anchor, direction, pos - 1)
    directly_right_pos = offset(anchor, direction, pos + 1)
    right_of_anchor = offset(anchor, direction, 1)

    left_unoccupied = not pos_in_bounds(directly_left_pos) or board[directly_left_pos] == -1
    right_unoccupied = not pos_in_bounds(directly_right_pos) or board[directly_right_pos] == -1
    far_right_unoccupied = not pos_in_bounds(right_of_anchor) or board[right_of_anchor] == -1

    if pos <= 0: # moving left
      word = char + word
      if char in old_arc.letter_set and left_unoccupied and far_right_unoccupied and new_tiles:
        if is_valid_word(word):
          action = build_action(pos, anchor, direction, word, new_tiles, blanks)
          # add action to possible actions
          actions.append(action)
      if new_arc:
        if pos_in_bounds(directly_left_pos) and directly_left_pos not in anchors_used:
          gen(pos - 1, word, rack, new_arc, new_tiles, blanks, anchor, direction)
        new_arc = new_arc.get_next(g.DELIM)
        if new_arc and left_unoccupied and pos_in_bounds(right_of_anchor):
          gen(1, word, rack, new_arc, new_tiles, blanks, anchor, direction) # add delimeter, switch directions
    else: # moving right
      word = word + char
      if char in old_arc.letter_set and right_unoccupied and new_tiles:
        if is_valid_word(word):
          action = build_action(pos, anchor, direction, word, new_tiles, blanks)
          # add action to possible actions
          actions.append(action)
      if new_arc and pos_in_bounds(directly_right_pos):
        gen(pos + 1, word, rack, new_arc, new_tiles, blanks, anchor, direction)

  def build_action(pos, anchor, direction, word, new_tiles, blanks):
    current_tile = offset(anchor, direction, pos)
    action = []
    for i in range(len(word)):
      if current_tile in new_tiles:
        action.append({'row': current_tile[0], 'col': current_tile[1], 'tile': char_to_char_idx(word[i])})
      elif current_tile in blanks:
        action.append({'row': current_tile[0], 'col': current_tile[1], 'tile': 26})
      current_tile = offset(current_tile, direction, 1)

    return action
  
  anchors = get_anchors(board)

  for anchor in anchors:
    for direction in DIRECTION:
      initial_arc = g.Arc("", GADDAG.root())
      gen(0, "", rack.copy(), initial_arc, [], [], anchor, direction)

  return actions

def get_anchors(board):
  anchors = []
  rows, cols = BOARD_DIM, BOARD_DIM

  mid_square_idx = 7
  # if center square is open, center is the only anchor
  if board[mid_square_idx, mid_square_idx] == -1:
    return [(mid_square_idx, mid_square_idx)]
  for row in range(rows):
    for col in range(cols):
      if (board[row, col] == -1): # checking if current square is empty
        neighbors = [
          (row - 1, col),
          (row + 1, col),
          (row, col - 1),
          (row, col + 1)
        ] 
        for r, c in neighbors: # checking around the empty square to look for non empties
          if 0 <= r < rows and 0 <= c < cols and board[r, c] != -1:
            anchors.append((row, col))

  return anchors
