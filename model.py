import numpy as np
from scipy.ndimage import convolve
from util import WORD_MULTIPLIER_POSITIONS, LETTER_MULTIPLIER_POSITIONS, calculate_score_for_action

def objective_function(weights, points_scored_val, weighted_multipliers_val, action_use_val, multiplier_distance_reduction_val, opened_spaces_val):
  return weights[0] * points_scored_val + weights[1] * weighted_multipliers_val + weights[2] * action_use_val + weights[3] * multiplier_distance_reduction_val + weights[4] * opened_spaces_val

# The literal in game points scored by action
def points_scored(board, action):
  return calculate_score_for_action(board, action)
  
# Sum of multipliers used, weighted by their multiplier and if they are for words or letters, 
# for the heuristic value of taking them away from the opponent
def weighted_multipliers(action):
  weighted_sum = 0

  for tile_placement in action:
    row = tile_placement["row"]
    col = tile_placement["col"]

    if (row, col) in WORD_MULTIPLIER_POSITIONS:
      weighted_sum += 3 * WORD_MULTIPLIER_POSITIONS[row, col] # word multipliers are more impactful, so weighted 3x as high
    elif (row, col) in LETTER_MULTIPLIER_POSITIONS:
      weighted_sum += LETTER_MULTIPLIER_POSITIONS[row, col]
    
  return weighted_sum

# A higher rack value means the move is more likely to be played
def action_use_value(action):
  letter_heuristic_values = {
    'A': 2, 'B': 4, 'C': 4, 'D': 4, 'E': 2,
    'F': 6, 'G': 4, 'H': 6, 'I': 2, 'J': 12,
    'K': 9, 'L': 2, 'M': 4, 'N': 2, 'O': 2,
    'P': 4, 'Q': 12, 'R': 2, 'S': 1, 'T': 2,
    'U': 2, 'V': 6, 'W': 6, 'X': 10, 'Y': 6, 'Z': 10,
    '_': 0
  }

  total_use_value = 0

  for tile_placement in action:
    letter = tile_placement["tile"]

    total_use_value += letter_heuristic_values[letter]

  return total_use_value

# TODO Update to consider number of moves it would take to reach a mulitplier instead of number of tiles
# We assume that the opponent can definitely reach 3 spots outwards from any move that we play
# Should we make this a dynamic measure later on?
def multiplier_distance_reduction(action, opponent_range=3):

  ## HELPER FUNCTIONS ##
  # Euclidean distance between tile and multiplier
  # TODO is euclidean distance the right metric? 4 tiles away in one direction is 1 move away, 
  # but that same distance in euclidean distance (or even shorter) is 2 moves away
  def calculate_distance(placement, multiplier_pos):
    return np.sqrt((placement["row"] - multiplier_pos[0]) ** 2 + (placement["col"] - multiplier_pos[1]) ** 2)

  # Decay function - encourages playing towards multipliers 
  def proximity_score(placement, multiplier_pos):
    distance = calculate_distance(placement, multiplier_pos)
    return 1 / (distance + 1)
  
  # Return true if building this move would expose the multiplier to the opponent
  def is_multiplier_exposed(placement, multiplier_pos, opponent_range):
    distance = calculate_distance(placement, multiplier_pos)
    return distance <= opponent_range

  score = 0

  ## Actual iteration logic
  for tile_placement in action:
    for pos, _ in LETTER_MULTIPLIER_POSITIONS:
      score += proximity_score(tile_placement, pos)

      if is_multiplier_exposed(tile_placement, pos, opponent_range):
        score -= 6
      elif calculate_distance(tile_placement, pos) > opponent_range:
        score += 3
    
    for pos, _ in WORD_MULTIPLIER_POSITIONS:
      score += proximity_score(tile_placement, pos)

      if is_multiplier_exposed(tile_placement, pos, opponent_range):
        score -= 10
      elif calculate_distance(tile_placement, pos) > opponent_range:
        score += 5

  return score

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