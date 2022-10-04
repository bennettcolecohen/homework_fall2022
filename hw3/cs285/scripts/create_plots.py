from tensorflow.python.summary.summary_iterator import summary_iterator
import pandas as pd
import matplotlib.pyplot as plt
import os

def read_data(path): 
    full_path = 'data/' + [el for el in os.listdir('data') if path in el][0]
    full_path += '/' + os.listdir(full_path)[0]
    steps = []
    avg_returns = []
    best_returns = []
    for e in summary_iterator(full_path): 
        for v in e.summary.value: 
            if v.tag == 'Train_EnvstepsSoFar': 
                steps.append(v.simple_value)
            elif v.tag == 'Train_AverageReturn': 
                avg_returns.append(v.simple_value)
            elif v.tag == 'Train_BestReturn': 
                best_returns.append(v.simple_value)
    return steps, avg_returns, best_returns

# Question 1 Plot
q1_path = 'q1_MsPacman'
steps, avg_returns, best_returns = read_data(q1_path)
steps = steps[:steps.index(1000001.0)+1]
avg_returns = avg_returns[:len(steps)]
best_returns = best_returns[:len(steps)]

fig, ax = plt.subplots(figsize = (8,6))
plt.plot(steps, avg_returns, label = 'Average Return')
plt.plot(steps, best_returns, label = 'Best Return')
plt.xlabel('Steps')
plt.ylabel('Return')
plt.title('MsPacman-v0 DQN Returns')
plt.legend()
fig.savefig('q1_dqn.png')