import numpy as np
from scipy.ndimage import convolve

def objective_function(alpha, beta, gamma, delta, epsilon, points_scored, weighted_multipliers_used, rack_value_lost, multiplier_distance_reduction, new_rows_opened):
    return 1 / (alpha * points_scored + beta * weighted_multipliers_used - gamma * rack_value_lost + delta * multiplier_distance_reduction + epsilon * new_rows_opened)

def points_scored(board, action):
  # return points_scored(board, action) from utility functions
  return
  
def weighted_multipliers(board, action):
  weighted_sum = 0

  for tile_placement in action:
    row = tile_placement["row"]
    col = tile_placement["col"]

    '''
    if (row,col is multiplier (needs adjustment to gym code))
      weighted_sum += multiplier value at row, col
    '''

  return weighted_sum

def rack_value_lost(board, action):
  letter_heuristic_values = {
    'A': 2, 'B': 4, 'C': 4, 'D': 4, 'E': 2,
    'F': 6, 'G': 4, 'H': 6, 'I': 2, 'J': 12,
    'K': 9, 'L': 2, 'M': 4, 'N': 2, 'O': 2,
    'P': 4, 'Q': 12, 'R': 2, 'S': 2, 'T': 2,
    'U': 2, 'V': 6, 'W': 6, 'X': 10, 'Y': 6, 'Z': 10,
    '_': 0
  }

  value_lost = 0

  for tile_placement in action:
    letter = tile_placement["tile"]

    value_lost += letter_heuristic_values[letter]

  return value_lost

def multiplier_distance_reduction(board, action):
  return

def opened_spaces(board, action):
  initial_empty_adjacent_tiles = calculate_empty_adjacent_tiles(board)

  # place tiles
  for tile_placement in action:
    row = tile_placement["row"]
    col = tile_placement["col"]
    tile = tile_placement["tile"]

    board[row, col] = tile
  
  new_empty_adjacent_tiles = calculate_empty_adjacent_tiles(board)

  return new_empty_adjacent_tiles - initial_empty_adjacent_tiles

def calculate_empty_adjacent_tiles(board):
  # Adjacency kernel
  kernel = np.array([[0, 1, 0],
                    [1, 0, 1],
                    [0, 1, 0]])
  
  occupied_tiles = board != -1
  empty_tiles = board == -1
  
  # finds all tiles that are adjacent to the occupied tiles
  adjacent_tiles = convolve(occupied_tiles.astype(int), kernel, mode='constant', cval=0)
  
  # tiles that are empty and adjacent to occupied tiles
  empty_and_adjacent = (adjacent_tiles > 0) & empty_tiles
  
  return np.sum(empty_and_adjacent.astype(int))