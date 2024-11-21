# TODO: MANUAL TESTING FILE, REMOVE BEFORE SUBMISSION

import numpy as np

board = np.full((10, 10), -1)  # -1 for empty cells
board[2, 1] = 0
board[2, 2] = 1
board[2, 3] = 2
board[3, 1] = 3
board[6, 2] = 7
board[4, 3] = 8
board[3, 5] = 12
board[4, 4] = 13

vertical_action = [
    {"row": 3, "col": 2, "tile": 4},
    {"row": 4, "col": 2, "tile": 5},
    {"row": 5, "col": 2, "tile": 6},
]

horizontal_action = [
    {"row": 3, "col": 2, "tile": 9},
    {"row": 3, "col": 3, "tile": 10},
    {"row": 3, "col": 4, "tile": 11},
]

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
    num_rows = 10
    num_cols = 10

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
        # while self.board[row--, col] is not empty, add that tile to tiles_in_word mask
        while (current_row_idx >= 0 and new_board[current_row_idx, col] != -1):
            tiles_in_word[current_row_idx, col] = True
            current_row_idx -= 1

        current_row_idx = last_placed_letter_idx
        # while self.board[row++, col] is not empty, add that tile to tiles_in_word mask
        while (current_row_idx <= num_rows - 1 and new_board[current_row_idx, col] != -1):
            tiles_in_word[current_row_idx, col] = True
            current_row_idx += 1
    else:
        # create mask of tiles between given range
        tiles_in_word[row, first_placed_letter_idx:last_placed_letter_idx + 1] = True

        current_col_idx = last_placed_letter_idx
        # while self.board[row, col--] is not empty, add that tile to tiles_in_word mask
        while (current_col_idx >= 0 and new_board[row, current_col_idx] != -1):
            tiles_in_word[row, current_col_idx] = True
            current_col_idx -= 1
            
        current_col_idx = last_placed_letter_idx
        # while self.board[row, col++] is not empty, add that tile to tiles_in_word mask
        while (current_col_idx <= num_cols - 1 and new_board[row, current_col_idx] != -1):
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
                word_key += ("_" if cell_value == 26 else chr(ord("A") + cell_value))
                letter_entries.append([row, col, int(cell_value), was_placed])

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

# is_action_continuous(board, action)
# print(get_words_made_by_action(board, vertical_action)) # should produce: 'BEFGH', 'DE', 'FIN'
# print(get_words_made_by_action(board, horizontal_action)) # should produce: 'DJKLM', 'BJ', 'CKI', 'LN'