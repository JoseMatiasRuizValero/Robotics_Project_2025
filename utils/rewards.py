import math

def get_distance(pos1, pos2):
    return math.sqrt((pos1[0]-pos2[0])**2 + (pos1[1]-pos2[1])**2)

def calculate_reward(robot_pos, goal_pos, sensors, action, prev_distance=None):
    dist = get_distance(robot_pos, goal_pos)

    SUCCESS_RADIUS = 0.8
    CLOSE_RADIUS = 1.2
    STEP_PENALTY = -0.005
    STOP_PENALTY = -0.3
    COLLISION_PENALTY = -80.0
    COLLISION_SENSOR_THRESHOLD = 500

    PROGRESS_SCALE = 40.0
    DISTANCE_SCALE = 0.5

    if dist < SUCCESS_RADIUS:
        return 300.0, True

    if max(sensors) > COLLISION_SENSOR_THRESHOLD:
        return COLLISION_PENALTY, True

    reward = STEP_PENALTY

    if action == 0:
        reward += STOP_PENALTY

    if prev_distance is not None:
        distance_change = prev_distance - dist
        reward += PROGRESS_SCALE * distance_change

    reward -= DISTANCE_SCALE * dist

    if dist < CLOSE_RADIUS:
        reward += 2.0

    return reward, False