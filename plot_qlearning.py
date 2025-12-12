import pandas as pd
import matplotlib.pyplot as plt
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

# Q-Learning on test1 map
q1 = pd.read_csv(os.path.join(script_dir, "results", "results_20251208_165504.csv"))
q2 = pd.read_csv(os.path.join(script_dir, "results", "results_20251208_163435.csv"))
q3 = pd.read_csv(os.path.join(script_dir, "results", "results_20251208_162655.csv"))

window = 20

q1_success = q1['success'].rolling(window=window).mean() * 100
q2_success = q2['success'].rolling(window=window).mean() * 100
q3_success = q3['success'].rolling(window=window).mean() * 100
avg = (q1_success + q2_success + q3_success) / 3

plt.figure(figsize=(10, 6))
plt.plot(q1_success, label='Run 1', alpha=0.7)
plt.plot(q2_success, label='Run 2', alpha=0.7)
plt.plot(q3_success, label='Run 3', alpha=0.7)
plt.plot(avg, label='Average', color='black', linewidth=2.5)

plt.title('Q-Learning Success Rate (3 Runs - Test1 Map)')
plt.xlabel('Episode')
plt.ylabel('Success Rate (%)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

