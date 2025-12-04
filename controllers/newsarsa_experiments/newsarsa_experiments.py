from controller import Robot, Supervisor
import sys
import math
import csv
import os
from datetime import datetime

# add paths for imports
sys.path.append('..')
sys.path.append('../..')

from sarsa_agent.sarsa_agent import SARSAAgent
from utils.newrewards import calculate_reward, COLLISION_PENALTY
from utils.metrics import print_training_summary, get_recent_performance

TIMESTEP = 64
MAX_V = 6.28

# --- low-level motion parameters ---
FORWARD_V = 0.5 * MAX_V  # normal forward speed
SLOW_FORWARD_V = 0.3 * MAX_V  # slow forward speed
TURN_V = 0.08 * MAX_V  # angular velocity for turning

# emergency stop threshold (slightly lower than COLLISION_SENSOR_THRESHOLD to brake earlier)
EMERGENCY_SENSOR_THRESHOLD = 480.0

# === NEW: action repeat count ===
ACTION_REPEAT = 4

# ============= MAP CONFIGURATION =============
# Change MAP_TYPE to test different environments
MAP_TYPE = "test1"  # Options: "test1", "test2", "original"

if MAP_TYPE == "test1":
    # Test1: 2x2 map with 3 barrels + 3 panels
    GOAL_POSITION = (0.8, -0.3)
    START_POSITION = (0.0, -0.8)
    FALLBACK_POSITION = (0.0, -0.6)
    print("Map: test1 (2x2 with obstacles)")

elif MAP_TYPE == "test2":
    # Test2: 2x2 maze with panels
    GOAL_POSITION = (0.6, 0.6)
    START_POSITION = (0.0, -0.8)
    FALLBACK_POSITION = (0.0, -0.6)
    print("Map: test2 (2x2 maze)")

else:  # original (Ultron.wbt)
    # Original: 2x2 open map
    GOAL_POSITION = (0.4, 0.4)
    START_POSITION = (-0.7, -0.7)
    FALLBACK_POSITION = (0.2, 0.2)
    print("Map: original (2x2 open)")

# Common settings for all maps
START_Y = 0.0
START_ROTATION = [0, 1, 0, 0]
MAP_BOUNDARY_X = 0.95
MAP_BOUNDARY_Z = 0.95
SUCCESS_RADIUS = 0.4
# =============================================

PS_GROUP_FRONT = [0, 7]
PS_GROUP_LEFT = [5, 6]
PS_GROUP_RIGHT = [1, 2]
SENSOR_THRESHOLDS = [100.0, 500.0]

# training params
NUM_EPISODES = 1000
MAX_STEPS = 1000

# === Updated: state size (L*3 x F*3 x R*3 x G*5 = 135) ===
STATE_SIZE = 135
ACTION_SIZE = 4

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

LOAD_EXISTING_QTABLE = False
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
QTABLE_PATH = os.path.join(RESULTS_DIR, 'sarsa_q_table.npy')
WARMUP_EPISODES = 100
WARMUP_MIN_EPSILON = 0.8


def run_sarsa_training():
    robot = Supervisor()

    # setup sensors
    ps = [robot.getDevice(f'ps{i}') for i in range(8)]
    for s in ps:
        s.enable(TIMESTEP)

    gps = robot.getDevice('gps')
    gps.enable(TIMESTEP)
    imu = robot.getDevice('inertial unit')
    imu.enable(TIMESTEP)

    lw = robot.getDevice('left wheel motor')
    rw = robot.getDevice('right wheel motor')
    lw.setPosition(float('inf'))
    rw.setPosition(float('inf'))
    lw.setVelocity(0.0)
    rw.setVelocity(0.0)

    agent = SARSAAgent(STATE_SIZE, ACTION_SIZE)

    if LOAD_EXISTING_QTABLE and os.path.exists(QTABLE_PATH):
        try:
            agent.load(QTABLE_PATH)
            print(f"Loaded existing SARSA Q-table from {QTABLE_PATH}")
        except Exception as exc:
            print(f"Warning: failed to load SARSA Q-table ({exc}), starting fresh")
    else:
        print("Starting fresh training (New State Structure)")

    print(f"Starting SARSA training: {NUM_EPISODES} episodes")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(RESULTS_DIR, f"sarsa_results_{timestamp}.csv")
    print(f"Saving results to: {csv_path}")

    robot_node = robot.getSelf()
    tField = robot_node.getField('translation')
    rField = robot_node.getField('rotation')

    episode_data = []

    for episode in range(NUM_EPISODES):
        # reset robot before each episode
        lw.setVelocity(0.0)
        rw.setVelocity(0.0)
        robot.step(TIMESTEP)

        start_x, start_z = START_POSITION
        robot_node.resetPhysics()
        tField.setSFVec3f([start_x, start_z, START_Y])
        rField.setSFRotation(START_ROTATION)

        # settle robot for some steps
        for _ in range(30):
            robot.step(TIMESTEP)

        total_reward = 0
        steps = 0
        success = False
        collisions = 0
        prev_dist = None

        # check initial collision status
        max_sensor_value = 0
        for _ in range(10):
            ps_values = [s.getValue() for s in ps]
            val_front = max(ps_values[i] for i in PS_GROUP_FRONT)
            val_left = max(ps_values[i] for i in PS_GROUP_LEFT)
            val_right = max(ps_values[i] for i in PS_GROUP_RIGHT)
            current_max = max([val_left, val_front, val_right])
            if current_max > max_sensor_value:
                max_sensor_value = current_max
            robot.step(TIMESTEP)

        # use fallback position if starting position is unsafe
        if max_sensor_value > 500:
            robot_node.resetPhysics()
            tField.setSFVec3f([FALLBACK_POSITION[0], FALLBACK_POSITION[1], START_Y])
            rField.setSFRotation(START_ROTATION)
            lw.setVelocity(0.0)
            rw.setVelocity(0.0)
            for _ in range(30):
                robot.step(TIMESTEP)

            gps_x, _, gps_z = gps.getValues()
            if abs(gps_x - FALLBACK_POSITION[0]) > 0.1 or abs(gps_z - FALLBACK_POSITION[1]) > 0.1:
                # allow extra settling if fallback isn't applied correctly
                for _ in range(20):
                    robot.step(TIMESTEP)

            # re-check collision
            for _ in range(10):
                robot.step(TIMESTEP)

            ps_values = [s.getValue() for s in ps]
            val_front = max(ps_values[i] for i in PS_GROUP_FRONT)
            val_left = max(ps_values[i] for i in PS_GROUP_LEFT)
            val_right = max(ps_values[i] for i in PS_GROUP_RIGHT)

            if max([val_left, val_front, val_right]) > 500:
                print(
                    f"Warning: SARSA Episode {episode + 1} - Robot starts in collision (sensor={max_sensor_value:.1f}), skipping...")
                episode_data.append({
                    'episode': episode + 1,
                    'steps': 0,
                    'total_reward': COLLISION_PENALTY,
                    'success': False,
                    'collisions': 1,
                    'epsilon': agent.epsilon
                })
                # warm-up epsilon policy
                if episode < WARMUP_EPISODES:
                    agent.epsilon = max(agent.epsilon, WARMUP_MIN_EPSILON)
                else:
                    agent.endEpisode()
                continue
            else:
                start_x, start_z = FALLBACK_POSITION
                print(
                    f"Info: SARSA Episode {episode + 1} - Moved robot to fallback position (sensor={max([val_left, val_front, val_right]):.1f})")

        # get initial distance to target
        x, _, z = gps.getValues()
        prev_dist = math.sqrt((x - GOAL_POSITION[0]) ** 2 + (z - GOAL_POSITION[1]) ** 2)

        # retrieve sensor values and orientation
        ps_values = [s.getValue() for s in ps]
        x, _, z = gps.getValues()
        yaw = imu.getRollPitchYaw()[2]

        # sensor groups' values
        val_front = max(ps_values[i] for i in PS_GROUP_FRONT)
        val_left = max(ps_values[i] for i in PS_GROUP_LEFT)
        val_right = max(ps_values[i] for i in PS_GROUP_RIGHT)

        # discretize sensor readings (3 levels)
        state_F = 0 if val_front < 100 else (1 if val_front < 500 else 2)
        state_L = 0 if val_left < 100 else (1 if val_left < 500 else 2)
        state_R = 0 if val_right < 100 else (1 if val_right < 500 else 2)

        # compute relative direction to target
        target_angle = math.atan2(GOAL_POSITION[1] - z, GOAL_POSITION[0] - x)
        relative_angle = target_angle - yaw
        if relative_angle > math.pi:
            relative_angle -= 2 * math.pi
        if relative_angle <= -math.pi:
            relative_angle += 2 * math.pi

        # === Updated: more detailed angle state (5 levels) ===
        threshold_center = 0.26  # approx 15 degrees
        threshold_side = 1.57  # approx 90 degrees

        if abs(relative_angle) < threshold_center:
            state_G = 0  # facing target
        elif -threshold_side < relative_angle <= -threshold_center:
            state_G = 1  # target front-left
        elif relative_angle <= -threshold_side:
            state_G = 2  # target back-left
        elif threshold_center <= relative_angle < threshold_side:
            state_G = 3  # target front-right
        else:
            state_G = 4  # target back-right

        # === Updated: state ID encoding formula ===
        state = (state_L * 45) + (state_F * 15) + (state_R * 5) + state_G

        if state < 0 or state >= STATE_SIZE:
            print(f"Warning: Invalid state {state}, clamping to valid range")
            state = max(0, min(STATE_SIZE - 1, state))

        # choose initial action for SARSA
        action = agent.chooseAction(state)

        for step in range(MAX_STEPS):

            # ==== 1. Safety check ====
            # Note: using previous loop’s ps_values on step 0
            if step == 0:
                val_front_prev = 0.0
            else:
                val_front_prev = max(ps_values[i] for i in PS_GROUP_FRONT)

            # determine if emergency stop is needed
            emergency_stop = val_front_prev > EMERGENCY_SENSOR_THRESHOLD

            # ==== 2. Action repeat loop (frame skipping) ====
            # Execute the chosen action ACTION_REPEAT times unless collision or emergency stop happens
            for _ in range(ACTION_REPEAT):
                if emergency_stop:
                    # force stop
                    lw.setVelocity(0.0)
                    rw.setVelocity(0.0)
                else:
                    # normal action execution
                    if action == 1:
                        lw.setVelocity(FORWARD_V)
                        rw.setVelocity(FORWARD_V)
                    elif action == 0:
                        lw.setVelocity(SLOW_FORWARD_V)
                        rw.setVelocity(SLOW_FORWARD_V)
                    elif action == 2:
                        lw.setVelocity(-TURN_V)
                        rw.setVelocity(TURN_V)
                    elif action == 3:
                        lw.setVelocity(TURN_V)
                        rw.setVelocity(-TURN_V)
                    else:
                        lw.setVelocity(0.0)
                        rw.setVelocity(0.0)

                if robot.step(TIMESTEP) == -1:
                    break

            # update sensor readings and position
            ps_values = [s.getValue() for s in ps]
            x, y, z = gps.getValues()
            yaw = imu.getRollPitchYaw()[2]

            # check for GPS anomaly
            if math.isnan(x) or math.isnan(z) or abs(x) > 10 or abs(z) > 10:
                print(
                    f"Warning: SARSA Episode {episode + 1} Step {step + 1} - GPS anomaly detected ({x}, {z}), ending episode")
                break

            # check map boundaries
            if abs(x) > MAP_BOUNDARY_X or abs(z) > MAP_BOUNDARY_Z:
                print(
                    f"Warning: SARSA Episode {episode + 1} Step {step + 1} - Robot out of bounds ({x:.2f}, {z:.2f}), ending episode")
                reward = COLLISION_PENALTY
                total_reward += reward
                collisions += 1
                break

            # update sensor group values
            val_front = max(ps_values[i] for i in PS_GROUP_FRONT)
            val_left = max(ps_values[i] for i in PS_GROUP_LEFT)
            val_right = max(ps_values[i] for i in PS_GROUP_RIGHT)

            # discretize sensors
            state_F = 0 if val_front < 100 else (1 if val_front < 500 else 2)
            state_L = 0 if val_left < 100 else (1 if val_left < 500 else 2)
            state_R = 0 if val_right < 100 else (1 if val_right < 500 else 2)

            # update target direction state
            target_angle = math.atan2(GOAL_POSITION[1] - z, GOAL_POSITION[0] - x)
            relative_angle = target_angle - yaw
            if relative_angle > math.pi:
                relative_angle -= 2 * math.pi
            if relative_angle <= -math.pi:
                relative_angle += 2 * math.pi

            if abs(relative_angle) < threshold_center:
                state_G = 0
            elif -threshold_side < relative_angle <= -threshold_center:
                state_G = 1
            elif relative_angle <= -threshold_side:
                state_G = 2
            elif threshold_center <= relative_angle < threshold_side:
                state_G = 3
            else:
                state_G = 4

            # compute next state
            next_state = (state_L * 45) + (state_F * 15) + (state_R * 5) + state_G

            if next_state < 0 or next_state >= STATE_SIZE:
                print(f"Warning: Invalid next_state {next_state}, clamping to valid range")
                next_state = max(0, min(STATE_SIZE - 1, next_state))

            # compute reward
            robot_pos = (x, z)
            sensor_values = [val_left, val_front, val_right]

            reward, done, collided = calculate_reward(
                robot_pos,
                GOAL_POSITION,
                sensor_values,
                action,
                prev_dist,
                success_radius=SUCCESS_RADIUS,
                alignment_angle=relative_angle
            )

            if collided:
                collisions += 1

            prev_dist = math.sqrt((x - GOAL_POSITION[0]) ** 2 + (z - GOAL_POSITION[1]) ** 2)

            if done and not collided:
                success = True

            next_action = agent.chooseAction(next_state)

            # SARSA update
            agent.update(state, action, reward, next_state, next_action)

            total_reward += reward
            steps += 1

            if done:
                break

            state = next_state
            action = next_action

        if episode < WARMUP_EPISODES:
            agent.epsilon = max(agent.epsilon, WARMUP_MIN_EPSILON)
        else:
            agent.endEpisode()

        episode_data.append({
            'episode': episode + 1,
            'steps': steps,
            'total_reward': total_reward,
            'success': success,
            'collisions': collisions,
            'epsilon': agent.epsilon
        })

        status = "SUCCESS" if success else "FAILED"
        print(
            f"SARSA Ep {episode + 1}/{NUM_EPISODES} - {status} - Steps: {steps}, Reward: {total_reward:.2f}, Collisions: {collisions}, ε: {agent.epsilon:.3f}")

        # print stats every 10 episodes
        if (episode + 1) % 10 == 0:
            recent = get_recent_performance(episode_data, 10)
            print(f"  Last 10: {recent['success_rate'] * 100:.1f}% success, {recent['avg_steps']:.1f} avg steps")
            # save to CSV
            try:
                with open(csv_path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=['episode', 'steps', 'total_reward', 'success', 'collisions',
                                                           'epsilon'])
                    writer.writeheader()
                    writer.writerows(episode_data)
            except PermissionError:
                backup_path = csv_path.replace('.csv', '_backup.csv')
                print(f"Warning: Could not save to {csv_path} (in use). Saving to {backup_path} instead.")
                try:
                    with open(backup_path, 'w', newline='') as f:
                        writer = csv.DictWriter(f,
                                                fieldnames=['episode', 'steps', 'total_reward', 'success', 'collisions',
                                                            'epsilon'])
                        writer.writeheader()
                        writer.writerows(episode_data)
                except Exception as e:
                    print(f"Error writing backup CSV: {e}")
            except Exception as e:
                print(f"Warning: Error saving CSV: {e}")

    # save final Q-table
    agent.save(QTABLE_PATH)
    print_training_summary(episode_data)


if __name__ == "__main__":
    run_sarsa_training()