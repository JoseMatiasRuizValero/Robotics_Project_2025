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

## Experiments
## Pid
## Q Learning Agent
## SARSA Agent
Implements SARSA algorithm, with its on-policy update. Its methodology relies within its methods, in this order

- `__init__:` this method creates the agent with all the initial values.
- `choose_action:`
- `update:`
- `endEpisode:`
- `save:` Saves the current qTable data to the path specified.
- `load:` Loads to the qTable the selected data from the path.

## SARSA Experiments
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
