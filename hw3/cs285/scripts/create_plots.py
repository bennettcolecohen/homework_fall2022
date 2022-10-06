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

# #Q2
# steps, dqn_avg1, _ = read_data(f'q2_dqn_1')
# _, dqn_avg2, _ = read_data(f'q2_dqn_2')
# _, dqn_avg3, _ = read_data(f'q2_dqn_3')
# dqn_avg = np.mean(np.array([dqn_avg1,dqn_avg2,dqn_avg3]), axis = 0)

# steps, double_dqn_avg1, _ = read_data(f'q2_doubledqn_1')
# _, double_dqn_avg2, _ = read_data(f'q2_doubledqn_2')
# _, double_dqn_avg3, _ = read_data(f'q2_doubledqn_3')
# double_dqn_avg = np.mean(np.array([double_dqn_avg1,double_dqn_avg2,double_dqn_avg3]), axis = 0)

# goal = np.repeat(150, len(steps[1:]))

# plot_data(
#     steps[1:], 
#     [dqn_avg, double_dqn_avg, goal],
#     ['Vanilla DQN', 'Double DQN', 'Target at 350k steps'],
#     'LunarLander-v3 DQN & Double DQN Returns','Steps', 'Returns', True, 'q2_with_target.png')


#Q4
# steps, ac1, _ = read_data(f'q4_ac_1_1_CartPole-v0')
# _, ac2, _ = read_data(f'q4_100_1_CartPole-v0')
# _, ac3, _ = read_data(f'q4_1_100_CartPole-v0')
# _, ac4, _ = read_data(f'q4_10_10_CartPole-v0')

# plot_data(
#     steps[:], 
#     [ac1,ac2,ac3,ac4], 
#     ['q4_1_1', 'q4_100_1', 'q4_1_100', 'q4_10_10'],
#      'Cartpole-V0 Actor-Critic', 
#      'Steps',
#      'Returns', 
#      True, 
#      'q4_actor_critic.png')

#Q5 - Pendulumn 
steps, pend_data, _ = read_data('q5_1_100_InvertedPendulum-v4')
plot_data(np.arange(100, step = 10), [pend_data], ['NTU=1, NGSPTU=100'], 'InvertedPendulum Actor-Critic', 
'Iterations', 'Return', True, 'q5_pendulum_iters.png')

#Q5 - Cheetah 
steps, cheetah_data, _ = read_data('q5_1_100_HalfCheetah-v4')
plot_data(np.arange(len(cheetah_data)), [cheetah_data], ['NTU=1, NGSPTU=100'], 'HalfCheetah Actor-Critic', 
'Iterations', 'Return', True, 'q5_cheetah_iters.png')