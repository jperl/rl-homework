# Exercise 2.5

# Design and conduct an experiment to demonstrate the difficulties that sample-average methods have for nonstationary problems.
# Use a modified version of the 10-armed testbed in which all the q*(a) start out equal and then take independent random walks
# (say by adding a normally distributed increment with mean zero and standard deviation 0.01 to all the q*(a) on each step).
# Prepare plots like Figure 2.2 for an action-value method using sample averages, incrementally computed, and another action-value
# method using a constant step-size parameter, a = 0.1. Use e = 0.1 and longer runs, say of 10,000 steps.
import numpy as np
import pandas as pd

def run(epsilon = 0.1, alpha = 0.1, num_steps = 10000):
  # Create 10-armed testbed
  q_star = np.zeros(10)

  # Calculate action values
  Q_avg = np.zeros(10)
  Q_const = np.zeros(10)

  rewards_from_avg, rewards_from_const, avg_is_optimal, const_is_optimal = [], [], [], []

  for n in range(1, num_steps + 1):
    # random walk q*
    q_star += np.random.normal(0, 0.01, 10)

    # select a sample index
    is_greedy = np.random.choice(2, p=[epsilon, 1 - epsilon])
    if is_greedy:
      avg_i = np.argmax(Q_avg)
      const_i = np.argmax(Q_const)
    else:
      avg_i = np.random.choice(10)
      const_i = np.random.choice(10)

    # average the sample value
    Q_avg[avg_i] = Q_avg[avg_i] + ((1 / n) * (q_star[avg_i] - Q_avg[avg_i]))

    # use constant update
    Q_const[const_i] = Q_const[const_i] + (alpha * (q_star[const_i] - Q_const[const_i]))

    # store sample reward
    rewards_from_avg.append(q_star[avg_i])
    rewards_from_const.append(q_star[const_i])

    # store sample is optimal
    optimal_i = np.argmax(q_star)
    avg_is_optimal.append(float(avg_i == optimal_i))
    const_is_optimal.append(float(const_i == optimal_i))

  return rewards_from_avg, rewards_from_const, avg_is_optimal, const_is_optimal


# average 2k runs
rewards_from_avg, rewards_from_const, avg_is_optimal, const_is_optimal = [], [], [], []
for r in range(2000):
  print(f'running {r}')
  results = run()
  rewards_from_avg.append(results[0])
  rewards_from_const.append(results[1])
  avg_is_optimal.append(results[2])
  const_is_optimal.append(results[3])

rewards_from_avg = np.array(rewards_from_avg)
rewards_from_const = np.array(rewards_from_const)
avg_is_optimal = np.array(avg_is_optimal)
const_is_optimal = np.array(const_is_optimal)

df = pd.DataFrame()
df['average'] = rewards_from_avg.mean(axis=0)
# df['average'] = df['average'].rolling(20).mean()
df['constant'] = rewards_from_const.mean(axis=0)
# df['constant'] = df['constant'].rolling(20).mean()
ax = df.plot()
fig = ax.get_figure()
fig.savefig('../output/sutton_2.5_avg_reward.png')

df = pd.DataFrame()
df['average'] = avg_is_optimal.mean(axis=0)
# df['average'] = df['average'].rolling(20).mean()
df['constant'] = const_is_optimal.mean(axis=0)
# df['constant'] = df['constant'].rolling(20).mean()
ax = df.plot()
fig = ax.get_figure()
fig.savefig('../output/sutton_2.5_optimal.png')
