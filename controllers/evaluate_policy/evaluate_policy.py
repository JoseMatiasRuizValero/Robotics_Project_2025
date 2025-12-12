from controller import Supervisor
import sys
import math
import os
import numpy as np

# add paths for imports
sys.path.append('..')
sys.path.append('../..')

TIMESTEP = 64
MAX_V = 6.28
COLLISION_SENSOR_THRESHOLD = 500.0

# ================= USER CONFIG =================
ALGORITHM = "sarsa"  
# Options: "qlearning", "sarsa"

MAP_TYPE = "test1"  
# Options: "original", "test1", "test2"

MAX_STEPS = 2000
SUCCESS_RADIUS = 0.4
# ===============================================

# ============= MAP CONFIGURATION ================
if MAP_TYPE == "test1":
    GOAL_POSITION = (0.4, 0.1)
    START_POSITION = (0.0, -0.8)
    print("Evaluation map: test1")

elif MAP_TYPE == "test2":
    GOAL_POSITION = (-0.4, -0.1)
    START_POSITION = (-0.2, 0.942588)
    print("Evaluation map: test2")

else:  # original
    GOAL_POSITION = (0.4, 0.4)
    START_POSITION = (-0.7, -0.7)
    print("Evaluation map: original")

START_Z = 0.0
START_ROTATION = [0, 1, 0, 0]
MAP_BOUNDARY_X = 0.95
MAP_BOUNDARY_Y = 0.95
# ===============================================

# sensor groups
PS_GROUP_FRONT = [0, 7]
PS_GROUP_LEFT = [5, 6]
PS_GROUP_RIGHT = [1, 2]

STATE_SIZE = 135
ACTION_SIZE = 4

# Angle thresholds for 5-level direction state (must match training code)
ANGLE_THRESHOLD_CENTER = 0.26  # approx 15 degrees
ANGLE_THRESHOLD_SIDE = 1.57    # approx 90 degrees


def discretize_sensor(value):
    """Discretize sensor value to 3 levels."""
    if value < 100:
        return 0
    elif value < 500:
        return 1
    else:
        return 2


def get_state(ps_values, gps, imu):
    """Get discrete state from sensor readings and position."""
    val_front = max(ps_values[i] for i in PS_GROUP_FRONT)
    val_left = max(ps_values[i] for i in PS_GROUP_LEFT)
    val_right = max(ps_values[i] for i in PS_GROUP_RIGHT)

    state_F = discretize_sensor(val_front)
    state_L = discretize_sensor(val_left)
    state_R = discretize_sensor(val_right)

    
    x, y, _ = gps.getValues()
    yaw = imu.getRollPitchYaw()[2]

    # goal direction (5 levels - must match training code)
    target_angle = math.atan2(GOAL_POSITION[1] - y, GOAL_POSITION[0] - x)
    relative_angle = target_angle - yaw

    if relative_angle > math.pi:
        relative_angle -= 2 * math.pi
    if relative_angle <= -math.pi:
        relative_angle += 2 * math.pi

    if abs(relative_angle) < ANGLE_THRESHOLD_CENTER:
        state_G = 0  # facing target
    elif -ANGLE_THRESHOLD_SIDE < relative_angle <= -ANGLE_THRESHOLD_CENTER:
        state_G = 1  # target front-left
    elif relative_angle <= -ANGLE_THRESHOLD_SIDE:
        state_G = 2  # target back-left
    elif ANGLE_THRESHOLD_CENTER <= relative_angle < ANGLE_THRESHOLD_SIDE:
        state_G = 3  # target front-right
    else:
        state_G = 4  # target back-right

    # calculate state ID (L*45 + F*15 + R*5 + G)
    state = (state_L * 45) + (state_F * 15) + (state_R * 5) + state_G
    return max(0, min(STATE_SIZE - 1, state))


def run_evaluation():
    robot = Supervisor()

    # sensors
    ps = [robot.getDevice(f'ps{i}') for i in range(8)]
    for s in ps:
        s.enable(TIMESTEP)

    gps = robot.getDevice('gps')
    gps.enable(TIMESTEP)

    imu = robot.getDevice('inertial unit')
    imu.enable(TIMESTEP)

    # motors
    lw = robot.getDevice('left wheel motor')
    rw = robot.getDevice('right wheel motor')
    lw.setPosition(float('inf'))
    rw.setPosition(float('inf'))
    lw.setVelocity(0.0)
    rw.setVelocity(0.0)

    # load Q-table (with map type in filename)
    project_root = os.path.join(os.path.dirname(__file__), '../..')
    results_dir = os.path.join(project_root, 'results')

    if ALGORITHM == "sarsa":
        q_table_path = os.path.join(results_dir, f'sarsa_q_table_{MAP_TYPE}.npy')
        print(f"Loading SARSA Q-table from: {q_table_path}")
    else:  # qlearning
        q_table_path = os.path.join(results_dir, f'q_table_{MAP_TYPE}.npy')
        print(f"Loading Q-learning Q-table from: {q_table_path}")

    if not os.path.exists(q_table_path):
        print(f"Error: Q-table file not found at {q_table_path}")
        print("Please train the agent first to generate the Q-table.")
        return

    q_table = np.load(q_table_path)
    print(f"Q-table loaded. Shape: {q_table.shape}")
    print(f"Starting evaluation with {ALGORITHM.upper()} policy (deterministic, epsilon=0)")

    # reset robot
    robot_node = robot.getSelf()
    tField = robot_node.getField('translation')
    rField = robot_node.getField('rotation')

    robot_node.resetPhysics()
    tField.setSFVec3f([START_POSITION[0], START_POSITION[1], START_Z])
    rField.setSFRotation(START_ROTATION)

    # wait for sensors to stabilize
    for _ in range(30):
        robot.step(TIMESTEP)

    print("Starting evaluation...")

    result = None
    min_dist = float('inf')

    for step in range(MAX_STEPS):
        ps_values = [s.getValue() for s in ps]
        x, y, _ = gps.getValues()

        # check for GPS anomaly
        if math.isnan(x) or math.isnan(y) or abs(x) > 10 or abs(y) > 10:
            print(f"Warning: GPS anomaly detected ({x}, {y})")
            result = "TIMEOUT"
            break

        # check map boundaries
        if abs(x) > MAP_BOUNDARY_X or abs(y) > MAP_BOUNDARY_Y:
            print(f"Warning: Robot out of bounds ({x:.2f}, {y:.2f})")
            result = "COLLISION"
            break

        # get state
        state = get_state(ps_values, gps, imu)

        # choose action deterministically (argmax, no exploration)
        action = int(np.argmax(q_table[state]))

        # execute action: 0=slow forward, 1=fast forward, 2=left, 3=right
        if action == 0:  # slow forward
            lw.setVelocity(0.3 * MAX_V)
            rw.setVelocity(0.3 * MAX_V)
        elif action == 1:  # fast forward
            lw.setVelocity(0.5 * MAX_V)
            rw.setVelocity(0.5 * MAX_V)
        elif action == 2:  # turn left
            lw.setVelocity(-0.15 * MAX_V)
            rw.setVelocity(0.15 * MAX_V)
        elif action == 3:  # turn right
            lw.setVelocity(0.15 * MAX_V)
            rw.setVelocity(-0.15 * MAX_V)
        else:
            lw.setVelocity(0.0)
            rw.setVelocity(0.0)

        robot.step(TIMESTEP)

        # check for collision
        ps_values_after = [s.getValue() for s in ps]
        max_sensor = max(max(ps_values_after[i] for i in PS_GROUP_FRONT),
                        max(ps_values_after[i] for i in PS_GROUP_LEFT),
                        max(ps_values_after[i] for i in PS_GROUP_RIGHT))
        
        if max_sensor > COLLISION_SENSOR_THRESHOLD:
            result = "COLLISION"
            break

        # check for success
        x, y, _ = gps.getValues()
        dist = math.sqrt((x - GOAL_POSITION[0])**2 + (y - GOAL_POSITION[1])**2)
        
        if dist < min_dist:
            min_dist = dist

        if dist < SUCCESS_RADIUS:
            result = "SUCCESS"
            break

    # determine final result if not set
    if result is None:
        result = "TIMEOUT"

    # print result
    print(f"\n{'='*50}")
    print(f"EVALUATION RESULT: {result}")
    print(f"Steps taken: {step + 1}")
    if result == "SUCCESS":
        print(f"Final distance to goal: {dist:.3f}m")
    else:
        print(f"Final distance to goal: {min_dist:.3f}m")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    run_evaluation()
