import util

def action_to_word(action):
  word = ''
  if not action:
    return word
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
word_validation_tests.append(("hello", True))
word_validation_tests.append(("olleh", False))
word_validation_tests.append(("olle", False))
word_validation_tests.append(("hel", False))
word_validation_tests.append(("h", False))
word_validation_tests.append(("world", True))
word_validation_tests.append(("word", True))
word_validation_tests.append(("worl", False))
word_validation_tests.append(("wor", False))
word_validation_tests.append(("w", False))
word_validation_results(word_validation_tests)