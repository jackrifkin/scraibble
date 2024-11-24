# TODO: MANUAL TESTING FILE, REMOVE BEFORE SUBMISSION

import numpy as np
import util

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

print(util.is_action_continuous(board, vertical_action))
print(util.get_words_made_by_action(board, vertical_action)) # should produce: 'BEFGH', 'DE', 'FIN'
print(util.get_words_made_by_action(board, horizontal_action)) # should produce: 'DJKLM', 'BJ', 'CKI', 'LN'