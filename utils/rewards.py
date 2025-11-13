import math

def get_distance(pos1, pos2):
    return math.sqrt((pos1[0]-pos2[0])**2 + (pos1[1]-pos2[1])**2)

def calculate_reward(robot_pos, goal_pos, sensors, action):
    dist = get_distance(robot_pos, goal_pos)

    # reached goal
    if dist < 0.1:
        return 100, True

    # collision
    if max(sensors) > 500:
        return -50, True

    # each step costs -1
    reward = -1

    # discourage stopping - robot was stopping too much in tests
    if action == 4:
        reward -= 2  # tried -1 first but still stopped a lot

    return reward, False