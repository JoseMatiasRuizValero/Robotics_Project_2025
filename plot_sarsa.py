import pandas as pd
import matplotlib.pyplot as plt
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

# SARSA on test1 map (3 runs)
s1 = pd.read_csv(os.path.join(script_dir, "results", "sarsa_results_20251208_171154.csv"))
s2 = pd.read_csv(os.path.join(script_dir, "results", "sarsa_results_20251208_172118.csv"))
s3 = pd.read_csv(os.path.join(script_dir, "results", "sarsa_results_20251208_173135.csv"))

window = 20

s1_success = s1['success'].rolling(window=window).mean() * 100
s2_success = s2['success'].rolling(window=window).mean() * 100
s3_success = s3['success'].rolling(window=window).mean() * 100
avg = (s1_success + s2_success + s3_success) / 3

plt.figure(figsize=(10, 6))
plt.plot(s1_success, label='Run 1', alpha=0.7)
plt.plot(s2_success, label='Run 2', alpha=0.7)
plt.plot(s3_success, label='Run 3', alpha=0.7)
plt.plot(avg, label='Average', color='black', linewidth=2.5)

plt.title('SARSA Success Rate (3 Runs - Test1 Map)')
plt.xlabel('Episode')
plt.ylabel('Success Rate (%)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

