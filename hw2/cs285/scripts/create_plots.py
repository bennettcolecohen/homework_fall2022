from turtle import color
from tensorflow.python.summary.summary_iterator import summary_iterator
import pandas as pd
import matplotlib.pyplot as plt
import os

def get_batch_data(batch = 'sb'): 

    # Get Correct Directories
    no_rtg_dsa = 'data' + '/' + [el for el in os.listdir('data') if f'{batch}_no_rtg_dsa' in el][0]
    rtg_dsa = 'data' + '/' + [el for el in os.listdir('data') if f'{batch}_rtg_dsa' in el][0]
    rtg_na = 'data' + '/' + [el for el in os.listdir('data') if f'{batch}_rtg_na' in el][0]

    # Get Correct Folders
    no_rtg_dsa += '/' + os.listdir(no_rtg_dsa)[0]
    rtg_dsa += '/' + os.listdir(rtg_dsa)[0]
    rtg_na += '/' + os.listdir(rtg_na)[0]

    data=[]
    for path in [no_rtg_dsa, rtg_dsa, rtg_na]: 
        vals = []
        for e in summary_iterator(path): 
            for v in e.summary.value: 
                if v.tag == 'Eval_AverageReturn': 
                    vals.append(v.simple_value)
        data.append(vals)

    return data


# def plot_sb_data(data, title ="CartPole Small Batch Avg Returns", file = 'cartpole_sb.png' ):

#     fig, ax = plt.subplots(figsize = (8,4))
#     no_rtg_dsa, rtg_dsa, rtg_na = data[0], data[1], data[2]

#     plt.plot(no_rtg_dsa, label = 'No RTG, DSA', marker='o', markersize='2')
#     plt.plot(rtg_dsa, label = 'RTG, DSA', marker='o', markersize='2')
#     plt.plot(rtg_na, label = 'RTG, NA', marker='o', markersize='2')


#     plt.xlabel('Iteration')
#     plt.ylabel('Average Return')
#     plt.title(title)
#     plt.legend();
#     fig.savefig(file)

#     return 

def get_tb_returns(list_of_path_patterns): 
    data = []
    for i, path_pattern in enumerate(list_of_path_patterns):
        base_dir = 'data'
        dir_path = base_dir + '/' + [el for el in os.listdir(base_dir) if path_pattern in el][0]
        file_path = dir_path + '/' + os.listdir(dir_path)[0]

        vals = []
        for e in summary_iterator(file_path): 
            for v in e.summary.value: 
                if v.tag == 'Eval_AverageReturn': 
                    vals.append(v.simple_value)
        data.append(vals)
    
    return data

def plot_tb(list_of_data, list_of_labels, title, output_file): 
    
    fig, ax = plt.subplots(figsize = (8,4))

    if len(list_of_data) == 1: 
        plt.plot(list_of_data[0], label = list_of_labels[0])

    else: 
        for data, label in zip(list_of_data, list_of_labels): 
            plt.plot(data, label = label)

    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel('Average Return')
    plt.legend(); 
    fig.savefig(output_file)


# Cartpole -- Small
# cartpole_sb_data = get_tb_returns(list_of_path_patterns = ['sb_no_rtg_dsa', 'sb_rtg_dsa', 'sb_rtg_na'])
# plot_tb(
#     cartpole_sb_data, 
#     ['no_rtg_dsa', 'rtg_dsa', 'rtg_na'], 
#     title = 'Cartpole Small Batch Returns', 
#     output_file = 'cartpole_sb_returns.png'
# )

# # Cartpole -- Large
# cartpole_lb_data = get_tb_returns(list_of_path_patterns = ['lb_no_rtg_dsa', 'lb_rtg_dsa', 'lb_rtg_na'])
# plot_tb(
#     cartpole_lb_data, 
#     ['no_rtg_dsa', 'rtg_dsa', 'rtg_na'], 
#     title = 'Cartpole Large Batch Returns', 
#     output_file = 'cartpole_lb_returns.png'
# )

# Lunar Lander 
lander_data = get_tb_returns(list_of_path_patterns = ['q3'])
plot_tb(
    list_of_data = lander_data, 
    list_of_labels = ['Lunar Lander'], 
    title = 'Lunar Lander Baseline Returns',
    output_file = 'lunar_lander_baseline_returns.png')
