# metrics for analyzing training results

def calculate_success_rate(episodes):
    if len(episodes) == 0:
        return 0
    successes = 0
    for ep in episodes:
        if ep['success']:
            successes += 1
    return successes / len(episodes)

def calculate_avg_steps(episodes):
    if not episodes:
        return 0
    return sum(ep['steps'] for ep in episodes) / len(episodes)

def calculate_avg_reward(episodes):
    if len(episodes) == 0:
        return 0
    total = 0
    for ep in episodes:
        total = total + ep['total_reward']
    return total / len(episodes)

def get_recent_performance(episodes, last_n=10):
    # get last N episodes
    recent = episodes[-last_n:]

    success_rate = calculate_success_rate(recent)
    avg_steps = calculate_avg_steps(recent)
    avg_reward = calculate_avg_reward(recent)

    return {
        'success_rate': success_rate,
        'avg_steps': avg_steps,
        'avg_reward': avg_reward
    }

def print_training_summary(episodes):
    if len(episodes) == 0:
        print("No data")
        return

    total = len(episodes)
    success_rate = calculate_success_rate(episodes)
    avg_steps = calculate_avg_steps(episodes)
    avg_reward = calculate_avg_reward(episodes)

    print("\n" + "="*50)
    print("TRAINING SUMMARY")
    print("="*50)
    print(f"Total Episodes: {total}")
    print(f"Success Rate: {success_rate*100:.1f}%")
    print(f"Average Steps: {avg_steps:.1f}")
    print(f"Average Reward: {avg_reward:.2f}")

    # last 10 episodes
    n_recent = min(10, total)
    recent = get_recent_performance(episodes, n_recent)
    print(f"\nLast {n_recent} Episodes:")
    print(f"  Success Rate: {recent['success_rate']*100:.1f}%")
    print(f"  Average Steps: {recent['avg_steps']:.1f}")
    print(f"  Average Reward: {recent['avg_reward']:.2f}")
    print("="*50)
