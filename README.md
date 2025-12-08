# Robotics_Project_2025
## Authors
Jose Matías Ruiz Valero

Hanpei Mo

Yin-Chu Huang

Jia-En Lu

## File hierarchy
```yaml

Robotics_Project_2025
  - controllers:
      - Roboticcontroller:
            Roboticcontroller.py
      - evaluation:
            evaluation.py
      - experiments:
            experiments.py
      - pid:
            pid.py
      - q_learning_agent:
            __init__.py
            q_learning_agent.py
      - sarsa_agent:
            sarsa_agent.py
      - sarsa_experiments:
            sarsa_experiments.py
  - utils:
        metrics.py
        rewards.py
  - worlds:
        Ultron.wbt
        test1.wbt
        test2.wbt
        test3.wbt
  .gitignore
  README.md
```

## Robotic Controler
This controller makes the e-puck robot be able to use the QLearningAgent to learn how to navigate towards a goal avoiding obstacles. It has many well defined constants. A short description of the most important:
- *TIMESTEP:* Simulation time step
- *MAX_V:* Max e-puck speed
- *GOAL_POSITION:* The goal x,z position
- *PS_GROUP_FRONT ; PS_GROUP_LEFT ; PS_GROUP_RIGHT:* The sensor goruping for each of this directions
- *SENSOR_THRESHOLDS:* Divided in three levels, so the robot knows how far away from an object is, between 0-100 it is safe/far awray, between 100-500 it is at a medium distance, and above 500 it is very close to an object, almost at a collision level.
- *ACTION_STOP; ACTION_FORWARD; ACTION_LEFT; ACTION_RIGHT:* Enum of the actions that the robot can take
- *ACTION_LIST ; ACTION_SIZE:* Array of the possible actions the robot can take and size of this array
- *MAX_EPISODE:* Maximum number of episodes
- *MAX_STEPS_PER_EPISODE:* Maximum number of steps until an episode ends

Then, the controller proceeds to enable all the desired sensors, as well as creating variables referencing them so we can identify each of them at any time. We will define two auxiliary methods used in this controller:
- `reset_robot:` This method resets the robot in each episode, returning it to its original starting position and stopping it.
- `velocityAction:` Returns the speed for each of the wheels based on what action we are executing.

Now the robot can start training. We use a while loop that works as long as webots dont stop the controller or whenever we break the loop.

- *GOAL_POSITION*
Every iteration of this loop it reads the sensor data (proximity sensors, GPS and distance to the goal), computes the goal direction,  discretizes it (says if its on front, to the left or to the right) and then combines it into a final integer called StateID which we can use for later debugging.

Now the agent chooses an action via the functions described below in _**[Q Learning Agent](https://github.com/JoseMatiasRuizValero/Robotics_Project_2025/blob/main/README.md#q-learning-agent)**_ and then executes it. Then calculates the reward from the _**[calculate_reward](https://github.com/JoseMatiasRuizValero/Robotics_Project_2025/blob/main/README.md#metrics-and-rewards)**_ function described below, updates the previous distance as the current distance and updates the qTable accordingly.

Then it stores the current state for the next step and checks for the robot to be done or for the steps to surpass the mas number of steps, if thats the case, we prepare everything to start a new episode, saving the results, resetting the robot and seeing if we are finished training, (seeing if we have reached the max number of episodes), if that is the case, the loop breaks and the program finishes, otherwise, we start a new episode.

## Experiments
Controller for experimenting with the Q Learning Agent
## Pid
## Q Learning Agent
Implements Q-Learning Algorithm, with its off-policy update, selecting the highest Q-value (greedy) action.

- `__init__:` this method creates the agent with all the initial values
- `chooseAction:` Ensures the state remains within valid bounds (this point will hold for all the other functions) and selects a random action with $\epsilon$ probability (in turn, explore); and an action with the highest Q-value for probability 1 - $\epsilon$. It returns an integer with the action index.
- `update:` Updates qTable using Q-Learning update rule, computing the maximum future Q-value, using the equation

    $Q(s,a) \leftarrow Q(s,a) + \alpha \left[ r + \gamma \max_{a'} Q(s',a') - Q(s,a) \right]$

  where $s$ is the current state, $a$ is the current action, $s′$ is the next state, $r$ is the reward observed after taking action $a$ in state $s$, $\alpha$ is the learning rate, and $\gamma$ is the discount factor.
- `endEpisode:` Reduces the exploration parameter $\epsilon$ so the agent explores less as it learns more. If $\epsilon$ is above the minimun value, multiply it by the decay factor. This gradually reduces exploration the longer the training goes, so it shifts towards better learned knowledge.
- `save:` Saves the current qTable data to the path specified (npy file).
- `load:` Loads to the qTable the selected data from the path (npy file).
  
## SARSA Agent
Implements SARSA algorithm, with its on-policy update. Its methodology relies within its methods, in this order:

- `__init__:` this method creates the agent with all the initial values.
- `choose_action:` Ensures the state remains within valid bounds (this point holds for all the other functions as well) and selects a random action with $\epsilon$ probability (in turn, explore); and an action with the highest Q-value for probability 1 - $\epsilon$. It returns an integer with the action index.
- `update:` This method updates the qTable using on-policy SARSA rule. Reads current and next Q-values (currentQ, nextQ), computes the SARSA target, and finally, computes the error and updates qTable. It uses this equation:

  $Q(s,a) \leftarrow Q(s,a) + \alpha \left[ r + \gamma  Q(s',a') - Q(s,a) \right]$

  where $s$ is the current state, $a$ is the current action, $s′$ is the next state, $r$ is the reward observed after taking action $a$ in state $s$, $\alpha$ is the learning rate, and $\gamma$ is the discount factor. This differentiates from Q-Leaning in the sense that SARSA uses the next action chosen by the policy, not always the best possible action.
  
- `endEpisode:` Decays the exploration rate $\epsilon$ after each episode ends. If $\epsilon$ is above the minimun value, multiply it by the decay factor. This gradually reduces exploration the longer the training goes, so it shifts towards better learned knowledge.
- `save:` Saves the current qTable data to the path specified (npy).
- `load:` Loads to the qTable the selected data from the path (npy).

## SARSA Experiments
Controller for experimenting with the SARSA Agent
## Metrics and Rewards
The **metrics.py** file has important functions to analyze the training results. In order:
 - ``calculate_success_rate:`` Takes an episodes array as the parameter and calculates the success rate of all the episodes with the classical definition of probabily
   
   $SuccessRate = \dfrac{SuccessEpisodes}{TotalEpisodes}$
- ``calculate_avg_steps:`` Same as before, but this one calculates the average steps between all episodes using the mean formula
 
   AvgSteps = $\sum_{Ep=0}^{nEp} \dfrac{Step_{Ep}} {nEp}$

- ``calculate_avg_rewards:`` This function calculates the average reward between all functions using the same mean formula but with different parameters.

  AvgReward = $\sum_{Ep=0}^{nEp} \dfrac{Reward_{Ep}} {nEp}$
- ``get_recent_performance:`` Calculates the three above functions for the last 10 episodes
- ``print_training_summary:`` Formats all the information of the above functions and prints a summary of all the episodes and another one of the last 10 episodes

The **rewards.py** file has a lot of Constants used to calculate the reward at each step, it has one auxiliary function and one main function. In this order:
- ``get_distance:`` Calculates the distance between two points, mainly, between the robot and the goal.

- ``calculate_reward:`` Calculates the reward for a given action, giving huge rewards for reaching the goal, penalising a lot the collisions, penalising stopping, giving a reward if the distance to the goal in this step is less than the previous one, giving a penalty the farther away from the goal you are, giving another bonus when very close to the goal and finally giving some mild sensor-based penalties.

## Worlds
Test worlds

- `Ultron.wbt: The default 2x2 grid with no obstacles`
- `test1.wbt: A 2x2 grid with some obstacles like barrels and walls`
- `test2.wbt: A 2x1 maze-like map`
- `test3.wbt: Morelike test1.wbt, it is a 2x2 grid with obstacles and walls `
