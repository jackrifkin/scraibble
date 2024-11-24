# TODO: MANUAL TESTING FILE, REMOVE BEFORE SUBMISSION

import numpy as np
import util

board0 = np.full((10, 10), -1)  # -1 for empty cells
board0[2, 1] = 0
board0[2, 2] = 1
board0[2, 3] = 2
board0[3, 1] = 3
board0[6, 2] = 7
board0[4, 3] = 8
board0[3, 5] = 12
board0[4, 4] = 13

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

'''
_ _ _ _ _ _
_ _ _ _ _ _
_ _ _ _ _ _
_ _ A _ E _
_ _ _ _ _ _
_ _ _ _ _ _
'''
board1 = np.full((util.BOARD_DIM, util.BOARD_DIM), -1)
board1[3, 2] = 0
board1[3, 4] = 4

'''
A _ _ _
_ _ _ _
A T _ _
_ _ _ _
'''
board2 = np.full((util.BOARD_DIM, util.BOARD_DIM), -1)
# board2[0, 0] = 0
# board2[2, 0] = 0
# board2[2, 1] = 19
cross_sets2 = np.zeros((util.BOARD_DIM, util.BOARD_DIM, 2, 26), dtype=int) - 1
cross_sets2[7, 7, util.DIRECTION.ACROSS.value] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
# cross_sets2[7, 6, util.DIRECTION.ACROSS.value, 0] = 7
# cross_sets2[7, 6, util.DIRECTION.DOWN.value, 0] = 7
# board2[1, 1] = 0
# board2[7, 7] = 0
# board2[7, 8] = 19

'''
_ _ A P _ _ E _
_ _ _ _ _ _ _ _
_ _ _ _ _ _ _ _
_ _ _ _ _ _ _ _
_ _ _ _ _ _ _ _
_ _ _ _ _ _ _ _
_ _ _ _ _ _ _ _
_ _ _ _ _ _ _ _
'''
board3 = np.full((util.BOARD_DIM, util.BOARD_DIM), -1)
board3[0, 2] = 0
board3[0, 3] = 15
board3[0, 6] = 4

rack1 = np.array([2, 0, 12, 15, 11, 17, 0])
rack2 = np.array([18, 19, 0, 15, 11, 4, 3])
rack3 = np.array([19, 14])
empty_rack = np.array([-1, -1, -1, -1, -1, -1, -1])

def action_to_word(action):
  word = ''
  for letter in action:
    word += util.char_idx_to_char(letter["tile"])
  return word

def action_to_grid(board, action):
  for letter in action:
    row = letter["row"]
    col = letter["col"]
    tile = letter["tile"]
    board[row, col] = tile
  return board

def print_board(board):
  for row in range(util.BOARD_DIM):
    row_str = ''
    for col in range(util.BOARD_DIM):
      row_str += util.char_idx_to_char(board[row, col])
      row_str += ' '
    print(row_str)

def actions_to_boards(board, actions):
  for action in actions:
    new_board = board.copy()
    new_board = action_to_grid(new_board, action)
    print(action_to_word(action))
    print_board(new_board)
    print('\n\n')
 

def testing_gaddag():
  # util.init_gaddag()
  print(util.GADDAG.root().letter_set)
  
testing_gaddag()
# print(util.is_action_continuous(board0, vertical_action))
# print(util.get_words_made_by_action(board0, vertical_action)) # should produce: 'BEFGH', 'DE', 'FIN'
# print(util.get_words_made_by_action(board0, horizontal_action)) # should produce: 'DJKLM', 'BJ', 'CKI', 'LN'
# actions_to_boards(board2, util.generate_possible_moves(board2, rack2, cross_sets2))