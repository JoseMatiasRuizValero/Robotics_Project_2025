import pandas as pd
import matplotlib.pyplot as plt
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

# q-learning (3 runs on test1 map)
q1 = pd.read_csv(os.path.join(script_dir, "results", "results_20251208_165504.csv"))
q2 = pd.read_csv(os.path.join(script_dir, "results", "results_20251208_163435.csv"))
q3 = pd.read_csv(os.path.join(script_dir, "results", "results_20251208_162655.csv"))

# sarsa (3 runs on test1 map)
s1 = pd.read_csv(os.path.join(script_dir, "results", "sarsa_results_20251208_171154.csv"))
s2 = pd.read_csv(os.path.join(script_dir, "results", "sarsa_results_20251208_172118.csv"))
s3 = pd.read_csv(os.path.join(script_dir, "results", "sarsa_results_20251208_173135.csv"))

window = 20

# average of 3 runs
q_reward = (q1['total_reward'].rolling(window=window).mean() + 
            q2['total_reward'].rolling(window=window).mean() + 
            q3['total_reward'].rolling(window=window).mean()) / 3

s_reward = (s1['total_reward'].rolling(window=window).mean() + 
            s2['total_reward'].rolling(window=window).mean() + 
            s3['total_reward'].rolling(window=window).mean()) / 3

plt.figure(figsize=(10, 6))
plt.plot(q_reward, label='Q-Learning (avg)', color='blue', linewidth=2)
plt.plot(s_reward, label='SARSA (avg)', color='red', linewidth=2)

plt.title('Q-Learning vs SARSA Total Reward (Test1 Map - 3 Runs Average)')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
