import pandas as pd
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

print("Q-Learning (Test1 Map)")
for i, q in enumerate([q1, q2, q3], 1):
    print(f"Run {i}: Success={q['success'].mean()*100:.1f}%, Steps={q['steps'].mean():.1f}, Reward={q['total_reward'].mean():.2f}, Collisions={q['collisions'].sum()}")

q_all = pd.concat([q1, q2, q3])
print(f"Average: Success={q_all['success'].mean()*100:.1f}%, Steps={q_all['steps'].mean():.1f}, Reward={q_all['total_reward'].mean():.2f}")

print("\nSARSA (Test1 Map)")
for i, s in enumerate([s1, s2, s3], 1):
    print(f"Run {i}: Success={s['success'].mean()*100:.1f}%, Steps={s['steps'].mean():.1f}, Reward={s['total_reward'].mean():.2f}, Collisions={s['collisions'].sum()}")

s_all = pd.concat([s1, s2, s3])
print(f"Average: Success={s_all['success'].mean()*100:.1f}%, Steps={s_all['steps'].mean():.1f}, Reward={s_all['total_reward'].mean():.2f}")
