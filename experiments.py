from controllers.q_learning_agent.q_learning_agent import QLearningAgent
from utils.rewards import calculate_reward

# hyperparameters
NUM_EPISODES = 100
MAX_STEPS = 500
STATE_SIZE = 81  # 3*3*3*3 from Hanpei's state representation
ACTION_SIZE = 4  # forward, left, right, stop

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

            # update agent
            agent.update(state, action, reward, next_state)

            total_reward += reward
            state = next_state
            steps += 1

            if done:
                break

        # end of episode
        agent.endEpisode()

        # print progress
        print(f"Episode {episode+1}/{NUM_EPISODES}, Steps: {steps}, Reward: {total_reward:.2f}")

if __name__ == "__main__":
    run_training()