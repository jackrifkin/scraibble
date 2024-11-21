# TODO: MANUAL TESTING FILE, REMOVE BEFORE SUBMISSION

import model as m
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

less_open_spaces_action = [
    {"row": 3, "col": 2, "tile": 4},
    {"row": 4, "col": 2, "tile": 5},
    {"row": 5, "col": 2, "tile": 6},
]

more_open_spaces_action = [
    {"row": 7, "col": 2, "tile": 9},
]

print(m.opened_spaces(board, less_open_spaces_action)) # should be -2
print(m.opened_spaces(board, more_open_spaces_action)) # should be +2