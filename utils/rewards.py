import math

def get_distance(pos1, pos2):
    return math.sqrt((pos1[0]-pos2[0])**2 + (pos1[1]-pos2[1])**2)

def calculate_reward(robot_pos, goal_pos, sensors, action, prev_distance=None):
    dist = get_distance(robot_pos, goal_pos)

    # reached goal
    if dist < 0.1:
        return 100, True

    # collision
    if max(sensors) > 500:
        return -200, True

    # each step costs -0.5
    reward = -0.5

    # discourage stopping
    if action == 0:
        reward -= 5

    # reward for getting closer to goal
    if prev_distance is not None:
        distance_change = prev_distance - dist
        if distance_change > 0:
            # moving closer - large reward
            reward += distance_change * 20
        else:
            # moving away - penalty
            reward += distance_change * 10

    return reward, False
