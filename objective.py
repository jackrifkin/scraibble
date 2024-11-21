def objective_function(alpha, beta, gamma, delta, epsilon, points_scored, weighted_multipliers_used, rack_value_lost, multiplier_distance_reduction, new_rows_opened):
    return 1 / (alpha * points_scored + beta * weighted_multipliers_used - gamma * rack_value_lost + delta * multiplier_distance_reduction + epsilon * new_rows_opened)

