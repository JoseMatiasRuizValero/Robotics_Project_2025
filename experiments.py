from controllers.q_learning_agent.q_learning_agent import QLearningAgent
from utils.rewards import calculate_reward
import csv
from datetime import datetime

# hyperparameters
NUM_EPISODES = 100
MAX_STEPS = 500
STATE_SIZE = 81  # 3*3*3*3 from Hanpei's state representation
ACTION_SIZE = 4  # forward, left, right, stop

# data tracking
episode_data = []

def run_training():
    # create agent
    agent = QLearningAgent(
        stateSize=STATE_SIZE,
        actionSize=ACTION_SIZE
    )

    # training loop
    for episode in range(NUM_EPISODES):
        # reset environment
        # TODO: need to connect to Webots here

        total_reward = 0
        steps = 0
        success = False

        # get initial state
        state = 0  # placeholder

        # episode loop
        for step in range(MAX_STEPS):
            # choose action
            action = agent.chooseAction(state)

            # take action in environment
            # TODO: send action to robot

            # get next state and reward
            next_state = 0  # placeholder
            robot_pos = (0, 0)  # placeholder
            goal_pos = (0.8, 0.8)
            sensors = [0] * 8  # placeholder

            reward, done = calculate_reward(robot_pos, goal_pos, sensors, action)

            # check if goal reached
            if done and reward > 0:
                success = True

            # update agent
            agent.update(state, action, reward, next_state)

            total_reward += reward
            state = next_state
            steps += 1

            if done:
                break

        # end of episode
        agent.endEpisode()

        # store episode data
        episode_info = {
            'episode': episode + 1,
            'steps': steps,
            'total_reward': total_reward,
            'success': success,
            'epsilon': agent.epsilon
        }
        episode_data.append(episode_info)

        # print progress
        status = "SUCCESS" if success else "FAILED"
        print(f"Episode {episode+1}/{NUM_EPISODES} - {status} - Steps: {steps}, Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.3f}")

        # save every 10 episodes
        if (episode + 1) % 10 == 0:
            save_results()

    # final save
    save_results()
    print(f"\nTraining complete! Results saved to results.csv")

def save_results():
    """Save episode data to CSV file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results_{timestamp}.csv"

    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['episode', 'steps', 'total_reward', 'success', 'epsilon'])
        writer.writeheader()
        writer.writerows(episode_data)

if __name__ == "__main__":
    run_training()