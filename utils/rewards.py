import math

DEFAULT_SUCCESS_RADIUS = 0.4
COLLISION_SENSOR_THRESHOLD = 500.0
COLLISION_PENALTY = -80.0

STEP_PENALTY = -0.005
STOP_PENALTY = -0.3

PROGRESS_SCALE = 40.0      # reward for reducing distance
DISTANCE_SCALE = 0.5       # penalty for being far from goal
CLOSE_RADIUS = 1.2
CLOSE_BONUS = 2.0

WARNING_THRESHOLD = 400.0
WARNING_PENALTY = -3.0     
DANGER_THRESHOLD = 450.0
DANGER_PENALTY = -6.0

# heading reward weight
HEADING_REWARD_SCALE = 2.0


def get_distance(pos1, pos2):
    return math.sqrt((pos1[0]-pos2[0])**2 + (pos1[1]-pos2[1])**2)


def calculate_reward(
    robot_pos,
    goal_pos,
    sensors,
    action,
    prev_distance=None,
    success_radius=None,
    alignment_angle=None  # robot's relative angle toward the goal
):
    radius = success_radius if success_radius is not None else DEFAULT_SUCCESS_RADIUS
    dist = get_distance(robot_pos, goal_pos)
    max_sensor = max(sensors)

    # success
    if dist < radius:
        return 300.0, True, False

    # collision
    if max_sensor > COLLISION_SENSOR_THRESHOLD:
        return COLLISION_PENALTY, True, True

    reward = STEP_PENALTY

    # penalise stopping
    if action == 0:
        reward += STOP_PENALTY

    # progress shaping: reward reduction in distance
    if prev_distance is not None:
        distance_change = prev_distance - dist
        reward += PROGRESS_SCALE * distance_change

    # distance shaping: being far is costly
    reward -= DISTANCE_SCALE * dist

    # small bonus when roughly close to goal
    if dist < CLOSE_RADIUS:
        reward += CLOSE_BONUS

    # heading reward
    # alignment_angle = 0 → perfectly facing the goal
    # Smaller magnitude = better orientation
    if alignment_angle is not None:
        heading_quality = 1.0 - (abs(alignment_angle) / math.pi)
        reward += heading_quality * HEADING_REWARD_SCALE

    # mild sensor-based penalties (discourage getting too close)
    if max_sensor > WARNING_THRESHOLD:
        reward += WARNING_PENALTY
    if max_sensor > DANGER_THRESHOLD:
        reward += DANGER_PENALTY

    return reward, False, False
