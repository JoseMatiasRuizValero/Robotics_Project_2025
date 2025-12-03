import math

DEFAULT_SUCCESS_RADIUS = 0.56
SUCCESS_REWARD = 400
INNER_RADIUS_MULTIPLIER = 0.5
INNER_RADIUS_BONUS = 40
COLLISION_THRESHOLD = 500
COLLISION_PENALTY = -100
WARNING_THRESHOLD = 400
DANGER_THRESHOLD = 450
STEP_COST = -0.5
STOP_EXTRA_COST = -1.5
DISTANCE_GAIN = 3.0


def get_distance(pos1, pos2):
    return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)


def calculate_reward(robot_pos, goal_pos, sensors, action, prev_distance=None, success_radius=None):
    # reward function for robot navigation
    radius = success_radius or DEFAULT_SUCCESS_RADIUS
    dist = get_distance(robot_pos, goal_pos)
    max_sensor = max(sensors)

    # check if reached goal
    if dist < radius:
        return SUCCESS_REWARD, True

    # check collision
    if max_sensor > COLLISION_THRESHOLD:
        return COLLISION_PENALTY, True

    reward = STEP_COST

    # penalize stopping
    if action == 0:
        reward += STOP_EXTRA_COST

    # distance-based shaping
    if prev_distance is not None:
        distance_change = prev_distance - dist
        reward += distance_change * DISTANCE_GAIN

    # proximity bonus
    if dist < 0.45:
        reward += 5
    if dist < 0.3:
        reward += 10
    if dist < 0.2:
        reward += 15
    if dist < 0.1:
        reward += 25

    if dist < radius * INNER_RADIUS_MULTIPLIER:
        reward += INNER_RADIUS_BONUS

    # sensor penalties
    if max_sensor > WARNING_THRESHOLD:
        reward -= 5
    if max_sensor > DANGER_THRESHOLD:
        reward -= 10

    return reward, False
