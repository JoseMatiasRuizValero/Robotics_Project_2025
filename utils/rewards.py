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
        return -50, True

    # each step costs -1
    reward = -1

    # discourage stopping
    if action == 4:
        reward -= 2

    # reward for getting closer to goal
    # this helps speed up learning
    if prev_distance is not None:
        distance_change = prev_distance - dist
        if distance_change > 0:
            # moving closer - small reward
            reward += distance_change * 5
        else:
            # moving away - small penalty
            reward += distance_change * 3

    return reward, False