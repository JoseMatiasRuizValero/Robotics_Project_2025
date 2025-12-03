from controller import Robot, Supervisor
import sys
import math
import csv
import os
from datetime import datetime

# add paths for imports
sys.path.append('..')
sys.path.append('../..')

from q_learning_agent.q_learning_agent import QLearningAgent
from utils.rewards import calculate_reward
from utils.metrics import print_training_summary, get_recent_performance

TIMESTEP = 64
MAX_V = 6.28
GOAL_POSITION = (0.6, 0.6)

PS_GROUP_FRONT = [0, 7]
PS_GROUP_LEFT = [5, 6]
PS_GROUP_RIGHT = [1, 2]
SENSOR_THRESHOLDS = [100.0, 500.0]

# training params
NUM_EPISODES = 1000
MAX_STEPS = 1000
STATE_SIZE = 81
ACTION_SIZE = 4

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

SUCCESS_RADIUS_STAGES = [
    (0.6, 200),
    (0.58, 300),
    (0.56, 500),
]

STAGE_TRANSITION_EPSILON = 0.5

LOAD_EXISTING_QTABLE = True
QTABLE_WARM_START_PATH = os.path.join(PROJECT_ROOT, 'q_table_stage.npy')
WARMUP_EPISODES = 100
WARMUP_MIN_EPSILON = 0.8
SAVE_STAGE_QTABLE = True

def run_training():
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

    agent = QLearningAgent(STATE_SIZE, ACTION_SIZE)

    if LOAD_EXISTING_QTABLE and os.path.exists(QTABLE_WARM_START_PATH):
        try:
            agent.load(QTABLE_WARM_START_PATH)
            print(f"Loaded existing Q-table from {QTABLE_WARM_START_PATH}")
        except Exception as exc:
            print(f"Warning: failed to load Q-table ({exc}), starting from scratch")

    print(f"Starting training: {NUM_EPISODES} episodes")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(PROJECT_ROOT, f"results_{timestamp}.csv")
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
        
        start_x, start_z = 0.0, 0.0
        robot_node.resetPhysics()
        tField.setSFVec3f([start_x, 0.005, start_z])
        rField.setSFRotation([0, 1, 0, 0])
        
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
            tField.setSFVec3f([0.2, 0.005, 0.2])
            rField.setSFRotation([0, 1, 0, 0])
            lw.setVelocity(0.0)
            rw.setVelocity(0.0)
            for _ in range(30):
                robot.step(TIMESTEP)
            
            gps_x, _, gps_z = gps.getValues()
            if abs(gps_x - 0.2) > 0.1 or abs(gps_z - 0.2) > 0.1:
                for _ in range(20):
                    robot.step(TIMESTEP)
            
            for _ in range(10):
                robot.step(TIMESTEP)
            
            ps_values = [s.getValue() for s in ps]
            val_front = max(ps_values[i] for i in PS_GROUP_FRONT)
            val_left = max(ps_values[i] for i in PS_GROUP_LEFT)
            val_right = max(ps_values[i] for i in PS_GROUP_RIGHT)
            
            if max([val_left, val_front, val_right]) > 500:
                print(f"Warning: Episode {episode+1} - Robot starts in collision (sensor={max_sensor_value:.1f}), skipping...")
                episode_data.append({
                    'episode': episode + 1,
                    'steps': 0,
                    'total_reward': -100,
                    'success': False,
                    'collisions': 1,
                    'epsilon': agent.epsilon
                })
                if episode < WARMUP_EPISODES:
                    agent.epsilon = max(agent.epsilon, WARMUP_MIN_EPSILON)
                else:
                    agent.endEpisode()
                continue
            else:
                start_x, start_z = 0.2, 0.2
                print(f"Info: Episode {episode+1} - Moved robot to (0.2, 0.2) (sensor={max([val_left, val_front, val_right]):.1f})")

        # get starting position
        x, _, z = gps.getValues()
        prev_dist = math.sqrt((x - GOAL_POSITION[0])**2 + (z - GOAL_POSITION[1])**2)

        # determine current stage
        cumulative_episodes = 0
        current_stage = 0
        current_success_radius = SUCCESS_RADIUS_STAGES[-1][0]
        for stage_idx, (radius, stage_episodes) in enumerate(SUCCESS_RADIUS_STAGES):
            if episode < cumulative_episodes + stage_episodes:
                current_success_radius = radius
                current_stage = stage_idx
                break
            cumulative_episodes += stage_episodes

        # check for stage transition
        if episode > 0:
            prev_cumulative = 0
            for stage_idx, (radius, stage_episodes) in enumerate(SUCCESS_RADIUS_STAGES):
                if episode == prev_cumulative + stage_episodes:
                    agent.epsilon = max(agent.epsilon, STAGE_TRANSITION_EPSILON)
                    print(f">>> Stage transition: radius changed to {SUCCESS_RADIUS_STAGES[stage_idx + 1][0] if stage_idx + 1 < len(SUCCESS_RADIUS_STAGES) else current_success_radius}, epsilon reset to {agent.epsilon:.3f}")
                    break
                prev_cumulative += stage_episodes

        for step in range(MAX_STEPS):
            # get sensor and position data
            ps_values = [s.getValue() for s in ps]
            x, y, z = gps.getValues()
            yaw = imu.getRollPitchYaw()[2]
            
            # check for GPS anomaly
            if math.isnan(x) or math.isnan(z) or abs(x) > 10 or abs(z) > 10:
                print(f"Warning: Episode {episode+1} Step {step+1} - GPS anomaly detected ({x}, {z}), ending episode")
                break
            
            # check boundaries
            if abs(x) > 0.95 or abs(z) > 0.95:
                print(f"Warning: Episode {episode+1} Step {step+1} - Robot out of bounds ({x:.2f}, {z:.2f}), ending episode")
                reward = -100
                total_reward += reward
                collisions += 1
                break

            # discretize sensors
            val_front = max(ps_values[i] for i in PS_GROUP_FRONT)
            val_left = max(ps_values[i] for i in PS_GROUP_LEFT)
            val_right = max(ps_values[i] for i in PS_GROUP_RIGHT)

            state_F = 0 if val_front < 100 else (1 if val_front < 500 else 2)
            state_L = 0 if val_left < 100 else (1 if val_left < 500 else 2)
            state_R = 0 if val_right < 100 else (1 if val_right < 500 else 2)

            # calculate goal direction
            target_angle = math.atan2(GOAL_POSITION[1] - z, GOAL_POSITION[0] - x)
            relative_angle = target_angle - yaw
            if relative_angle > math.pi:
                relative_angle -= 2 * math.pi
            if relative_angle <= -math.pi:
                relative_angle += 2 * math.pi

            if abs(relative_angle) < (math.pi / 3):
                state_G = 0
            elif relative_angle < 0:
                state_G = 1
            else:
                state_G = 2

            # calculate state ID
            state = (state_L * 27) + (state_F * 9) + (state_R * 3) + state_G
            
            if state < 0 or state >= STATE_SIZE:
                print(f"Warning: Invalid state {state}, clamping to valid range")
                state = max(0, min(STATE_SIZE - 1, state))

            # choose action
            action = agent.chooseAction(state)

            # execute action (0=stop, 1=forward, 2=left, 3=right)
            if action == 1:
                lw.setVelocity(0.5 * MAX_V)
                rw.setVelocity(0.5 * MAX_V)
            elif action == 2:
                lw.setVelocity(-0.08 * MAX_V)
                rw.setVelocity(0.08 * MAX_V)
            elif action == 3:
                lw.setVelocity(0.08 * MAX_V)
                rw.setVelocity(-0.08 * MAX_V)
            else:
                lw.setVelocity(0.0)
                rw.setVelocity(0.0)

            robot.step(TIMESTEP)

            # get next state
            ps_values = [s.getValue() for s in ps]
            x, _, z = gps.getValues()
            yaw = imu.getRollPitchYaw()[2]

            val_front = max(ps_values[i] for i in PS_GROUP_FRONT)
            val_left = max(ps_values[i] for i in PS_GROUP_LEFT)
            val_right = max(ps_values[i] for i in PS_GROUP_RIGHT)

            state_F = 0 if val_front < 100 else (1 if val_front < 500 else 2)
            state_L = 0 if val_left < 100 else (1 if val_left < 500 else 2)
            state_R = 0 if val_right < 100 else (1 if val_right < 500 else 2)

            target_angle = math.atan2(GOAL_POSITION[1] - z, GOAL_POSITION[0] - x)
            relative_angle = target_angle - yaw
            if relative_angle > math.pi:
                relative_angle -= 2 * math.pi
            if relative_angle <= -math.pi:
                relative_angle += 2 * math.pi

            if abs(relative_angle) < (math.pi / 3):
                state_G = 0
            elif relative_angle < 0:
                state_G = 1
            else:
                state_G = 2

            next_state = (state_L * 27) + (state_F * 9) + (state_R * 3) + state_G
            
            if next_state < 0 or next_state >= STATE_SIZE:
                print(f"Warning: Invalid next_state {next_state}, clamping to valid range")
                next_state = max(0, min(STATE_SIZE - 1, next_state))

            # calculate reward
            robot_pos = (x, z)
            sensor_values = [val_left, val_front, val_right]
            reward, done = calculate_reward(robot_pos, GOAL_POSITION, sensor_values, action, prev_dist, success_radius=current_success_radius)

            if reward == -100:
                collisions = collisions + 1

            prev_dist = math.sqrt((x - GOAL_POSITION[0])**2 + (z - GOAL_POSITION[1])**2)

            if done and reward > 0:
                success = True

            # update Q-table
            agent.update(state, action, reward, next_state)

            total_reward += reward
            steps += 1

            if done:
                break

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
        print(f"Ep {episode+1}/{NUM_EPISODES} - {status} - Steps: {steps}, Reward: {total_reward:.2f}, Collisions: {collisions}, ε: {agent.epsilon:.3f}")

        # print stats every 10 episodes
        if (episode + 1) % 10 == 0:
            recent = get_recent_performance(episode_data, 10)
            print(f"  Last 10: {recent['success_rate']*100:.1f}% success, {recent['avg_steps']:.1f} avg steps")
            # save to CSV
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
    agent.save(os.path.join(PROJECT_ROOT, 'q_table_final.npy'))
    if SAVE_STAGE_QTABLE:
        agent.save(QTABLE_WARM_START_PATH)
    print_training_summary(episode_data)

if __name__ == "__main__":
    run_training()
