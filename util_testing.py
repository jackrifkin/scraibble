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
rack2 = np.array([18, 19, 0, 15, 11, 4, 3]) # STAPLED
rack3 = np.array([19, 14]) # TO
rack4 = np.array([22, 14, 17, 3]) # WORD
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
    if (len(action) > 1):
      new_board = board.copy()
      new_board = action_to_grid(new_board, action)
      print(f"{action_to_word(action)} {len(action)}")
      print_board(new_board)
      print('\n\n')

def word_validation_results(tests):
  all_passed = True
  for word, expected_valid in tests:
    if util.GADDAG.is_word_in_gaddag(word) != expected_valid:
      all_passed = False
      print(f"validation for word: '{word}' failed.")
  if all_passed:
    print("All word validation tests passed!")

word_validation_tests = []
# print(util.is_action_continuous(board0, vertical_action))
# print(util.get_words_made_by_action(board0, vertical_action)) # should produce: 'BEFGH', 'DE', 'FIN'
# print(util.get_words_made_by_action(board0, horizontal_action)) # should produce: 'DJKLM', 'BJ', 'CKI', 'LN'
# actions_to_boards(board2, util.generate_possible_moves(board2, rack4, cross_sets2))
word_validation_tests.append(("hello", True))
word_validation_tests.append(("olleh", False))
word_validation_tests.append(("olle", False))
word_validation_tests.append(("hell", False))
word_validation_tests.append(("hel", False))
word_validation_tests.append(("he", False))
word_validation_tests.append(("h", False))
word_validation_tests.append(("world", True))
word_validation_tests.append(("word", True))
word_validation_tests.append(("worl", False))
word_validation_tests.append(("wor", False))
word_validation_tests.append(("wo", False))
word_validation_tests.append(("w", False))
word_validation_results(word_validation_tests)
