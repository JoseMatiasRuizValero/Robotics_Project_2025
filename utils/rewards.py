import math

def get_distance(pos1, pos2):
    return math.sqrt((pos1[0]-pos2[0])**2 + (pos1[1]-pos2[1])**2)

def calculate_reward(robot_pos, goal_pos, sensors, action, prev_distance=None):
    dist = get_distance(robot_pos, goal_pos)

    # reached goal - MASSIVE reward (increased threshold for easier success)
    if dist < 0.12:
        return 1000, True

    # collision - increased penalty for better risk awareness
    if max(sensors) > 500:
        return -200, True

    # ZERO step cost - encourage exploration instead of early termination
    reward = 0.0

    # discourage stopping - but mildly
    if action == 0:
        reward -= 0.5

    # reward for getting closer - MODERATE rewards to prevent exploitation
    if prev_distance is not None:
        distance_change = prev_distance - dist
        if distance_change > 0:
            # moving closer - good reward to encourage goal-seeking
            reward += distance_change * 300
        else:
            # moving away - penalty to discourage bad moves
            reward += distance_change * 50
    
    # proximity bonus - MINIMAL staged rewards to guide robot
    # Reduced significantly to prevent reward exploitation
    if dist < 0.5:
        reward += 5    # getting closer
    if dist < 0.3:
        reward += 10   # approaching goal
    if dist < 0.2:
        reward += 20   # very close
    if dist < 0.15:
        reward += 30   # almost there! (total: 65)

    return reward, False
