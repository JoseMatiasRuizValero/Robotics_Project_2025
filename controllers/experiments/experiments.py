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
GOAL_POSITION = (0.4, 0.4)
SUCCESS_RADIUS = 0.8

# sensor groups from Hanpei's code
PS_GROUP_FRONT = [0, 7]
PS_GROUP_LEFT = [5, 6]
PS_GROUP_RIGHT = [1, 2]
SENSOR_THRESHOLDS = [100.0, 500.0]

# training params
NUM_EPISODES = 1000
MAX_STEPS = 650
STATE_SIZE = 81
ACTION_SIZE = 4

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

    # motors
    lw = robot.getDevice('left wheel motor')
    rw = robot.getDevice('right wheel motor')
    lw.setPosition(float('inf'))
    rw.setPosition(float('inf'))
    lw.setVelocity(0.0)
    rw.setVelocity(0.0)

    # create agent
    agent = QLearningAgent(STATE_SIZE, ACTION_SIZE)

    print(f"Starting training: {NUM_EPISODES} episodes")

    # create unique csv path for this run
    project_root = os.path.join(os.path.dirname(__file__), '../..')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(project_root, f"results_{timestamp}.csv")
    print(f"Saving results to: {csv_path}")
    
    # get robot node for position reset
    robot_node = robot.getSelf()
    tField = robot_node.getField('translation')
    rField = robot_node.getField('rotation')
    
    # initialize episode data list
    episode_data = []
    
    for episode in range(NUM_EPISODES):
        # reset robot - stop motors first
        lw.setVelocity(0.0)
        rw.setVelocity(0.0)
        robot.step(TIMESTEP)
        
        # reset robot position
        start_x, start_z = -0.7, -0.7
        tField.setSFVec3f([start_x, 0.0, start_z])
        rField.setSFRotation([0, 1, 0, 0])
        #reset physics
        robot_node.resetPhysics()
        
        # wait for sensors
        for _ in range(20):
            robot.step(TIMESTEP)

        total_reward = 0
        steps = 0
        success = False
        collisions = 0  # track number of collisions in this episode
        prev_dist = None

        # check initial collision
        max_sensor_value = 0
        for _ in range(5):
            ps_values = [s.getValue() for s in ps]
            val_front = max(ps_values[i] for i in PS_GROUP_FRONT)
            val_left = max(ps_values[i] for i in PS_GROUP_LEFT)
            val_right = max(ps_values[i] for i in PS_GROUP_RIGHT)
            current_max = max([val_left, val_front, val_right])
            if current_max > max_sensor_value:
                max_sensor_value = current_max
            robot.step(TIMESTEP)
        
        # check if robot starts in collision
        if max_sensor_value > 500:
            # try (0.2, 0.2) as fallback
            tField.setSFVec3f([0.2, 0.0, 0.2])
            robot_node.resetPhysics()
            for _ in range(30):
                robot.step(TIMESTEP)
            
            # verify GPS updated
            gps_x, _, gps_z = gps.getValues()
            if abs(gps_x - 0.2) > 0.1 or abs(gps_z - 0.2) > 0.1:
                for _ in range(20):
                    robot.step(TIMESTEP)
            
            # check if safe now
            ps_values = [s.getValue() for s in ps]
            val_front = max(ps_values[i] for i in PS_GROUP_FRONT)
            val_left = max(ps_values[i] for i in PS_GROUP_LEFT)
            val_right = max(ps_values[i] for i in PS_GROUP_RIGHT)
            
            if max([val_left, val_front, val_right]) > 500:
                # (0.2, 0.2) also in collision, skip episode
                print(f"Warning: Episode {episode+1} - Robot starts in collision (sensor={max_sensor_value:.1f}), skipping...")
                episode_data.append({
                    'episode': episode + 1,
                    'steps': 0,
                    'total_reward': -200,  # collision penalty
                    'success': False,
                    'collisions': 1,
                    'epsilon': agent.epsilon
                })
                agent.endEpisode()
                continue
            else:
                # successfully moved to (0.2, 0.2)
                start_x, start_z = 0.2, 0.2
                print(f"Info: Episode {episode+1} - Moved robot to (0.2, 0.2) (sensor={max([val_left, val_front, val_right]):.1f})")

        # get starting position for distance calculation
        x, _, z = gps.getValues()
        prev_dist = math.sqrt((x - GOAL_POSITION[0])**2 + (z - GOAL_POSITION[1])**2)

        dist_to_goal = prev_dist
        min_dist = prev_dist

        for step in range(MAX_STEPS):
            # get sensor values
            ps_values = [s.getValue() for s in ps]
            x, _, z = gps.getValues()
            yaw = imu.getRollPitchYaw()[2]

            # discretize sensors (copied from Hanpei)
            val_front = max(ps_values[i] for i in PS_GROUP_FRONT)
            val_left = max(ps_values[i] for i in PS_GROUP_LEFT)
            val_right = max(ps_values[i] for i in PS_GROUP_RIGHT)

            state_F = 0 if val_front < 100 else (1 if val_front < 500 else 2)
            state_L = 0 if val_left < 100 else (1 if val_left < 500 else 2)
            state_R = 0 if val_right < 100 else (1 if val_right < 500 else 2)

            # goal direction
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
            
            # ensure state is within valid range
            if state < 0 or state >= STATE_SIZE:
                print(f"Warning: Invalid state {state}, clamping to valid range")
                state = max(0, min(STATE_SIZE - 1, state))

            # choose action
            action = agent.chooseAction(state)

            # execute action
            # Note: action is 0,1,2,3 (stop, forward, left, right)
            if action == 1:  # forward
                lw.setVelocity(0.5 * MAX_V)
                rw.setVelocity(0.5 * MAX_V)
            elif action == 2:  # left
                lw.setVelocity(-0.15 * MAX_V)
                rw.setVelocity(0.15 * MAX_V)
            elif action == 3:  # right
                lw.setVelocity(0.15 * MAX_V)
                rw.setVelocity(-0.15 * MAX_V)
            else:  # stop (action == 0)
                lw.setVelocity(0.0)
                rw.setVelocity(0.0)

            robot.step(TIMESTEP)

            # get next state
            ps_values = [s.getValue() for s in ps]
            x, _, z = gps.getValues()
            yaw = imu.getRollPitchYaw()[2]

            # recalculate state (same as above)
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
            
            # ensure next_state is within valid range
            if next_state < 0 or next_state >= STATE_SIZE:
                print(f"Warning: Invalid next_state {next_state}, clamping to valid range")
                next_state = max(0, min(STATE_SIZE - 1, next_state))

            # get reward
            robot_pos = (x, z)
            sensor_values = [val_left, val_front, val_right]
            reward, done = calculate_reward(robot_pos, GOAL_POSITION, sensor_values, action, prev_dist)

            # track collisions
            dist_to_goal = math.sqrt((x - GOAL_POSITION[0])**2 + (z - GOAL_POSITION[1])**2)

            if reward <= -70:
                collisions += 1

            # update prev_dist for next step
            prev_dist = dist_to_goal

            if dist_to_goal < SUCCESS_RADIUS:
                success = True
                done = True

            # update
            agent.update(state, action, reward, next_state)

            total_reward += reward
            steps += 1

            x, _, z = gps.getValues()
            dist_to_goal = math.sqrt((x - GOAL_POSITION[0])**2 + (z - GOAL_POSITION[1])**2)

            prev_dist = dist_to_goal

            if dist_to_goal < min_dist:
                min_dist = dist_to_goal

            if done:
                break

        agent.endEpisode()

        # save episode data
        episode_data.append({
            'episode': episode + 1,
            'steps': steps,
            'total_reward': total_reward,
            'success': success,
            'collisions': collisions,
            'epsilon': agent.epsilon
        })

        status = "SUCCESS" if success else "FAILED"
        print(f"Ep {episode+1}/{NUM_EPISODES} - {status} - Steps: {steps}, Reward: {total_reward:.2f}, Collisions: {collisions}, Distance: {dist_to_goal:.2f}, MinDist: {min_dist:.2f}, ε: {agent.epsilon:.3f}")

        if (episode + 1) % 10 == 0:
            recent = get_recent_performance(episode_data, 10)
            print(f"  Last 10: {recent['success_rate']*100:.1f}% success, {recent['avg_steps']:.1f} avg steps")
            # save every 10 episodes to the same file
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['episode', 'steps', 'total_reward', 'success', 'collisions', 'epsilon'])
                writer.writeheader()
                writer.writerows(episode_data)

    # save final results
    project_root = os.path.join(os.path.dirname(__file__), '../..')
    agent.save(os.path.join(project_root, 'q_table_final.npy'))
    print_training_summary(episode_data)

if __name__ == "__main__":
    run_training()
