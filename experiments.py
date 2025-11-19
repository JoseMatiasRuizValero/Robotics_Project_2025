from controller import Robot, Supervisor
import sys
import math
import csv
from datetime import datetime

sys.path.append('controllers')
from q_learning_agent.q_learning_agent import QLearningAgent
from utils.rewards import calculate_reward
from utils.metrics import print_training_summary, get_recent_performance

TIMESTEP = 64
MAX_V = 6.28
GOAL_POSITION = (0.8, 0.8)

# sensor groups from Hanpei's code
PS_GROUP_FRONT = [0, 7]
PS_GROUP_LEFT = [5, 6]
PS_GROUP_RIGHT = [1, 2]
SENSOR_THRESHOLDS = [100.0, 500.0]

# training params
NUM_EPISODES = 100
MAX_STEPS = 500
STATE_SIZE = 81
ACTION_SIZE = 4

episode_data = []

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

    for episode in range(NUM_EPISODES):
        # reset robot
        robot_node = robot.getFromDef('EPUCK')
        if robot_node:
            robot_node.getField('translation').setSFVec3f([0.0, 0.0, 0.0])
            robot_node.getField('rotation').setSFRotation([0, 1, 0, 0])

        robot.step(TIMESTEP)

        total_reward = 0
        steps = 0
        success = False
        prev_dist = None

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

            # choose action
            action = agent.chooseAction(state)

            # execute action
            if action == 1:  # forward
                lw.setVelocity(0.5 * MAX_V)
                rw.setVelocity(0.5 * MAX_V)
            elif action == 2:  # left
                lw.setVelocity(-0.2 * MAX_V)
                rw.setVelocity(0.2 * MAX_V)
            elif action == 3:  # right
                lw.setVelocity(0.2 * MAX_V)
                rw.setVelocity(-0.2 * MAX_V)
            else:  # stop
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

            # get reward
            robot_pos = (x, z)
            reward, done = calculate_reward(robot_pos, GOAL_POSITION, ps_values, action, prev_dist)

            prev_dist = math.sqrt((x - GOAL_POSITION[0])**2 + (z - GOAL_POSITION[1])**2)

            if done and reward > 0:
                success = True

            # update
            agent.update(state, action, reward, next_state)

            total_reward += reward
            steps += 1

            if done:
                break

        agent.endEpisode()

        # save data
        episode_data.append({
            'episode': episode + 1,
            'steps': steps,
            'total_reward': total_reward,
            'success': success,
            'epsilon': agent.epsilon
        })

        status = "SUCCESS" if success else "FAILED"
        print(f"Ep {episode+1}/{NUM_EPISODES} - {status} - Steps: {steps}, Reward: {total_reward:.2f}, ε: {agent.epsilon:.3f}")

        if (episode + 1) % 10 == 0:
            recent = get_recent_performance(episode_data, 10)
            print(f"  Last 10: {recent['success_rate']*100:.1f}% success, {recent['avg_steps']:.1f} avg steps")
            # save every 10 episodes
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            with open(f"results_{timestamp}.csv", 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['episode', 'steps', 'total_reward', 'success', 'epsilon'])
                writer.writeheader()
                writer.writerows(episode_data)

    # save final results
    agent.save('q_table_final.npy')
    print_training_summary(episode_data)

if __name__ == "__main__":
    run_training()