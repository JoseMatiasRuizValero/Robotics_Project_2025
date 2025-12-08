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
from utils.rewards import calculate_reward, COLLISION_PENALTY
from utils.metrics import print_training_summary, get_recent_performance

TIMESTEP = 64
MAX_V = 6.28

# motion parameters
FORWARD_V = 0.5 * MAX_V       # normal forward speed
SLOW_FORWARD_V = 0.3 * MAX_V  # slow forward speed
TURN_V = 0.15 * MAX_V         # angular velocity for turning

# Emergency stop threshold (slightly lower than COLLISION_SENSOR_THRESHOLD to brake earlier)
EMERGENCY_SENSOR_THRESHOLD = 480.0

# Action repeat count (frame skipping)
ACTION_REPEAT = 4

# map config
MAP_TYPE = "test1"  # "test1", "test2", "original"

if MAP_TYPE == "test1":
    # Test1: 2x2 map with 3 barrels + 3 panels
    GOAL_POSITION = (0.4, 0.1)
    START_POSITION = (0.0, -0.8)
    FALLBACK_POSITION = (0.0, -0.6)
    print("Map: test1 (2x2 with obstacles)")
    
elif MAP_TYPE == "test2":
    # Test2: 1x2 maze with panels
    GOAL_POSITION = (-0.4, -0.1)
    START_POSITION = (-0.2, 0.942588)
    FALLBACK_POSITION = (-0.2, 0.8)
    print("Map: test2 (1x2 maze)")
    
else:  # original (Ultron.wbt)
    # Original: 2x2 open map
    GOAL_POSITION = (0.4, 0.4)
    START_POSITION = (-0.7, -0.7)
    FALLBACK_POSITION = (0.2, 0.2)
    print("Map: original (2x2 open)")

# Common settings for all maps
START_Z = 0.0  # Z is height
START_ROTATION = [0, 1, 0, 0]
MAP_BOUNDARY_X = 0.95
MAP_BOUNDARY_Y = 0.95
SUCCESS_RADIUS = 0.4

PS_GROUP_FRONT = [0, 7]
PS_GROUP_LEFT = [5, 6]
PS_GROUP_RIGHT = [1, 2]
SENSOR_THRESHOLDS = [100.0, 500.0]

# training params
NUM_EPISODES = 1000
MAX_STEPS = 1000
# State size: L*3 x F*3 x R*3 x G*5 = 135
STATE_SIZE = 135
ACTION_SIZE = 4
LEARNING_RATE = 0.2
DISCOUNT_FACTOR = 0.90
EPSILON = 1.0
EPSILON_MIN = 0.03
EPSILON_DECAY = 0.99

# Angle thresholds for 5-level direction state
ANGLE_THRESHOLD_CENTER = 0.26  # approx 15 degrees
ANGLE_THRESHOLD_SIDE = 1.57    # approx 90 degrees

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
QTABLE_PATH = os.path.join(RESULTS_DIR, 'sarsa_q_table.npy')

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

    agent = SARSAAgent(STATE_SIZE, ACTION_SIZE, learningRate=LEARNING_RATE, 
                       discountFactor=DISCOUNT_FACTOR, epsilon=EPSILON, 
                       epsilonMin=EPSILON_MIN, epsilonDecay=EPSILON_DECAY)

    print(f"Starting SARSA training: {NUM_EPISODES} episodes")
    print(f"Hyperparameters: LR={LEARNING_RATE}, γ={DISCOUNT_FACTOR}, ε_min={EPSILON_MIN}, decay={EPSILON_DECAY}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(RESULTS_DIR, f"sarsa_results_{timestamp}.csv")
    print(f"Saving results to: {csv_path}")

    robot_node = robot.getSelf()
    tField = robot_node.getField('translation')
    rField = robot_node.getField('rotation')
    
    episode_data = []

    for episode in range(NUM_EPISODES):
        # reset robot
        lw.setVelocity(0.0)
        rw.setVelocity(0.0)
        robot.step(TIMESTEP)
        
        start_x, start_y = START_POSITION
        robot_node.resetPhysics()
        tField.setSFVec3f([start_x, start_y, START_Z])
        rField.setSFRotation(START_ROTATION)
        
        for _ in range(30):
            robot.step(TIMESTEP)

        total_reward = 0
        steps = 0
        success = False
        collisions = 0
        prev_dist = None

        # check initial collision
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
        
        # try fallback position if collision
        if max_sensor_value > 500:
            robot_node.resetPhysics()
            tField.setSFVec3f([FALLBACK_POSITION[0], FALLBACK_POSITION[1], START_Z])
            rField.setSFRotation(START_ROTATION)
            lw.setVelocity(0.0)
            rw.setVelocity(0.0)
            for _ in range(30):
                robot.step(TIMESTEP)
            
            gps_x, gps_y, _ = gps.getValues()
            if abs(gps_x - FALLBACK_POSITION[0]) > 0.1 or abs(gps_y - FALLBACK_POSITION[1]) > 0.1:
                for _ in range(20):
                    robot.step(TIMESTEP)
            
            for _ in range(10):
                robot.step(TIMESTEP)
            
            ps_values = [s.getValue() for s in ps]
            val_front = max(ps_values[i] for i in PS_GROUP_FRONT)
            val_left = max(ps_values[i] for i in PS_GROUP_LEFT)
            val_right = max(ps_values[i] for i in PS_GROUP_RIGHT)
            
            if max([val_left, val_front, val_right]) > 500:
                print(f"Warning: SARSA Episode {episode+1} - Robot starts in collision (sensor={max_sensor_value:.1f}), skipping...")
                episode_data.append({
                    'episode': episode + 1,
                    'steps': 0,
                    'total_reward': COLLISION_PENALTY,
                    'success': False,
                    'collisions': 1,
                    'epsilon': agent.epsilon
                })
                agent.endEpisode()
                continue
            else:
                start_x, start_y = FALLBACK_POSITION
                print(f"Info: SARSA Episode {episode+1} - Moved robot to fallback position (sensor={max([val_left, val_front, val_right]):.1f})")

        # get starting position
        x, y, _ = gps.getValues()
        prev_dist = math.sqrt((x - GOAL_POSITION[0])**2 + (y - GOAL_POSITION[1])**2)

        ps_values = [s.getValue() for s in ps]
        x, y, _ = gps.getValues()
        yaw = imu.getRollPitchYaw()[2]

        val_front = max(ps_values[i] for i in PS_GROUP_FRONT)
        val_left = max(ps_values[i] for i in PS_GROUP_LEFT)
        val_right = max(ps_values[i] for i in PS_GROUP_RIGHT)

        state_F = 0 if val_front < 100 else (1 if val_front < 500 else 2)
        state_L = 0 if val_left < 100 else (1 if val_left < 500 else 2)
        state_R = 0 if val_right < 100 else (1 if val_right < 500 else 2)

        # goal direction (5 levels)
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
        
        if state < 0 or state >= STATE_SIZE:
            print(f"Warning: Invalid state {state}, clamping to valid range")
            state = max(0, min(STATE_SIZE - 1, state))

        # choose initial action for SARSA
        action = agent.chooseAction(state)

        for step in range(MAX_STEPS):
            # safety check
            emergency_stop = val_front > EMERGENCY_SENSOR_THRESHOLD

            # action repeat
            for _ in range(ACTION_REPEAT):
                if emergency_stop:
                    # force stop
                    lw.setVelocity(0.0)
                    rw.setVelocity(0.0)
                else:
                    # execute action: 0=slow forward, 1=fast forward, 2=left, 3=right
                    if action == 1:  # fast forward
                        lw.setVelocity(FORWARD_V)
                        rw.setVelocity(FORWARD_V)
                    elif action == 0:  # slow forward
                        lw.setVelocity(SLOW_FORWARD_V)
                        rw.setVelocity(SLOW_FORWARD_V)
                    elif action == 2:  # left
                        lw.setVelocity(-TURN_V)
                        rw.setVelocity(TURN_V)
                    elif action == 3:  # right
                        lw.setVelocity(TURN_V)
                        rw.setVelocity(-TURN_V)
                    else:
                        lw.setVelocity(0.0)
                        rw.setVelocity(0.0)
                
                if robot.step(TIMESTEP) == -1:
                    break

            # get sensor and position data
            ps_values = [s.getValue() for s in ps]
            x, y, _ = gps.getValues()
            yaw = imu.getRollPitchYaw()[2]
            
            # check for GPS anomaly
            if math.isnan(x) or math.isnan(y) or abs(x) > 10 or abs(y) > 10:
                print(f"Warning: SARSA Episode {episode+1} Step {step+1} - GPS anomaly detected ({x}, {y}), ending episode")
                break
            
            # check boundaries
            if abs(x) > MAP_BOUNDARY_X or abs(y) > MAP_BOUNDARY_Y:
                print(f"Warning: SARSA Episode {episode+1} Step {step+1} - Robot out of bounds ({x:.2f}, {y:.2f}), ending episode")
                reward = COLLISION_PENALTY
                total_reward += reward
                collisions += 1
                break

            val_front = max(ps_values[i] for i in PS_GROUP_FRONT)
            val_left = max(ps_values[i] for i in PS_GROUP_LEFT)
            val_right = max(ps_values[i] for i in PS_GROUP_RIGHT)

            state_F = 0 if val_front < 100 else (1 if val_front < 500 else 2)
            state_L = 0 if val_left < 100 else (1 if val_left < 500 else 2)
            state_R = 0 if val_right < 100 else (1 if val_right < 500 else 2)

            # goal direction (5 levels)
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
            next_state = (state_L * 45) + (state_F * 15) + (state_R * 5) + state_G
            
            if next_state < 0 or next_state >= STATE_SIZE:
                print(f"Warning: Invalid next_state {next_state}, clamping to valid range")
                next_state = max(0, min(STATE_SIZE - 1, next_state))

            # calculate reward (with alignment_angle for heading reward)
            robot_pos = (x, y)
            sensor_values = [val_left, val_front, val_right]
            reward, done, collided = calculate_reward(
                robot_pos, GOAL_POSITION, sensor_values, action, prev_dist,
                success_radius=SUCCESS_RADIUS, alignment_angle=relative_angle
            )

            # track collisions
            if collided:
                collisions += 1

            prev_dist = math.sqrt((x - GOAL_POSITION[0])**2 + (y - GOAL_POSITION[1])**2)

            # check success based on reward function result
            if done and not collided:
                success = True

            # choose next action
            next_action = agent.chooseAction(next_state)

            # SARSA update
            agent.update(state, action, reward, next_state, next_action)

            total_reward += reward
            steps += 1

            if done:
                break

            # move to next state
            state = next_state
            action = next_action

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
        print(f"SARSA Ep {episode+1}/{NUM_EPISODES} - {status} - Steps: {steps}, Reward: {total_reward:.2f}, Collisions: {collisions}, ε: {agent.epsilon:.3f}")

        # print stats every 10 episodes
        if (episode + 1) % 10 == 0:
            recent = get_recent_performance(episode_data, 10)
            print(f"  Last 10: {recent['success_rate']*100:.1f}% success, {recent['avg_steps']:.1f} avg steps")
        
        # save Q-table every 100 episodes
        if (episode + 1) % 100 == 0:
            agent.save(QTABLE_PATH)
            print(f"  Q-table saved at episode {episode + 1}")
        
        # save to CSV every 10 episodes
        if (episode + 1) % 10 == 0:
            try:
                with open(csv_path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=['episode', 'steps', 'total_reward', 'success', 'collisions', 'epsilon'])
                    writer.writeheader()
                    writer.writerows(episode_data)
            except PermissionError:
                backup_path = csv_path.replace('.csv', '_backup.csv')
                print(f"Warning: Could not save to {csv_path} (in use). Saving to {backup_path} instead.")
                try:
                    with open(backup_path, 'w', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=['episode', 'steps', 'total_reward', 'success', 'collisions', 'epsilon'])
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

