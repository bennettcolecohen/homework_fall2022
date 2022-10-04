from tensorflow.python.summary.summary_iterator import summary_iterator
import pandas as pd
import numpy as np
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

def plot_data(x, list_of_y, list_of_labels, title, xlabel, ylabel, legend, output_file):

    fig, ax = plt.subplots(figsize = (8,6))

    for idx, y in enumerate(list_of_y): 
        plt.plot(x, y, label = list_of_labels[idx])

    plt.title(title), plt.xlabel(xlabel), plt.ylabel(ylabel)
    if legend: 
        plt.legend()
    fig.savefig(output_file)



#Q1 
# q1_path = 'q1_MsPacman'
# steps, avg_returns, best_returns = read_data(q1_path)
# steps = steps[:steps.index(1000001.0)+1]
# avg_returns = avg_returns[:len(steps)]
# best_returns = best_returns[:len(steps)]
# plot_data(steps, [avg_returns, best_returns], ['Avg. Return', 'Best Return'],'MsPacman-v0 DQN Returns', 'Steps', 'Returns', True, 'q1_dqn.png')

#Q2
dqn_avg_returns, ddqn_avg_returns = [],[],
for i in [1,2,3]: 
    dqn_steps, dqn_avg, _ = read_data(f'q2_dqn_{i}')
    _, ddqn_avg, _ = read_data(f'q2_doubledqn_{i}')

    dqn_avg_returns.append(dqn_avg)
    ddqn_avg_returns.append(ddqn_avg)

dqn_avg_avg = np.mean(np.array(dqn_avg_returns), axis = 0)
ddqn_avg_avg = np.mean(np.array(dqn_avg_returns), axis = 0)


plot_data(dqn_steps[1:], [dqn_avg_avg, ddqn_avg_avg], 
['DQN Avg.', 'Double DQN Avg.'],
'LunarLander-v3 DQN & Double DQN Returns', 'Steps', 'Returns', True, 'q2.png')


