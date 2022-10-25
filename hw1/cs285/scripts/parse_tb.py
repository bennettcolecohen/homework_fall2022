from turtle import color
from tensorflow.python.summary.summary_iterator import summary_iterator
import pandas as pd
import matplotlib.pyplot as plt
import os

def get_log_data(bc_path, dagger_path): 

    bc_file_path = bc_path + os.listdir(bc_path)[0]
    dagger_file_path = dagger_path + os.listdir(dagger_path)[0]
    # Behavior cloning mean return
    for e in summary_iterator(bc_file_path): 
        for v in e.summary.value: 
            if v.tag == 'Train_AverageReturn': 
                expert_mean = v.simple_value
            if v.tag == 'Eval_AverageReturn': 
                bc_mean = v.simple_value

    # DAgger mean and std
    means, stds = [],[]
    for e in summary_iterator(dagger_file_path): 
        for v in e.summary.value: 
            if v.tag == 'Eval_AverageReturn': 
                means.append(v.simple_value)
            elif v.tag == 'Eval_StdReturn': 
                stds.append(v.simple_value)

    return expert_mean, bc_mean, means, stds

def plot_returns(env_name, expert_mean, bc_mean, means, stds): 
    fig, ax = plt.subplots(figsize = (8,4))
    x = range(len(means))
    y = means

    plt.plot(x,y, label = 'Mean')
    plt.errorbar(x, y, yerr = stds,fmt ='o', label = 'Std')
    ax.axhline(expert_mean, 0, 20, color='green', ls='--', label = 'Avg. Expert Return')
    ax.axhline(bc_mean, 0, 20, color = 'red', ls= '--', label = 'Avg Behavior Cloning Return')

    plt.xlabel('Iteration')
    plt.ylabel('Average Return')
    plt.title(f'{env_name} DAgger Average Returns')
    plt.legend();
    fig.savefig(f'{env_name}_dagger_plot.png')

### WILLL NEED TO FIX ALL OF THESE PATHS LATER
if __name__ == '__main__': 

    ant_bc_path = 'data/q1_bc_ant/'
    ant_dagger_path = 'data/q2_dagger_ant/'

    walker_bc_path = 'data/q1_bc_walker/'
    walker_dagger_path = 'data/q2_dagger_walker/'

    # Parse log data from tensorboard
    ant_expert, ant_bc, ant_means, ant_std = get_log_data(ant_bc_path, ant_dagger_path)
    walker_expert, walker_bc, walker_means, walker_std = get_log_data(walker_bc_path, walker_dagger_path)

    # Plot 
    plot_returns('AntV4', ant_expert, ant_bc, ant_means, ant_std)
    plot_returns('Walker2d', walker_expert, walker_bc, walker_means, walker_std)
