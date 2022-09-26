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
    # plt.ylim(-50, 600)
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

# # Lunar Lander 
# lander_data = get_tb_returns(list_of_path_patterns = ['q3'])
# plot_tb(
#     list_of_data = lander_data, 
#     list_of_labels = ['Lunar Lander'], 
#     title = 'Lunar Lander Baseline Returns',
#     output_file = 'lunar_lander_baseline_returns.png')

# pendulum_data_all = get_tb_returns(list_of_path_patterns=[
#     'q2_pg_q2_b500_r1e-2', 'q2_pg_q2_b500_r2e-3', 'q2_pg_q2_b500_r5e-3', 
#     'q2_pg_q2_b1000_r1e-2', 'q2_pg_q2_b1000_r2e-3', 'q2_pg_q2_b1000_r5e-3', 
#     'q2_pg_q2_b5000_r1e-2', 'q2_pg_q2_b5000_r2e-3', 'q2_pg_q2_b5000_r5e-3'
# ])
# plot_tb(
#     list_of_data = pendulum_data_all, 
#     list_of_labels = ['b500_r1e-2', 'b500_r2e-3', 'b500_r5e-3', 
#                     'b1000_r1e-2', 'b1000_r2e-3', 'b1000_r5e-3', 
#                     'b5000_r1e-2', 'b5000_r2e-3', 'b5000_r5e-3'], 
#     title = 'Inverted Pendulum All Search Returns',
#     output_file = 'inverted_pendulum_all.png')

# pendulum_best = get_tb_returns(list_of_path_patterns=['q2_pg_q2_b500_r1e-2'])
# plot_tb(
#     list_of_data = pendulum_best, 
#     list_of_labels = ['b500_r1e-2'], 
#     title = 'Inverted Pendulum Best Returns',
#     output_file = 'inverted_pendulum_best.png')

hopper_data = get_tb_returns(
    list_of_path_patterns=[
        'q2_pg_q5_b2000_r0.001_lambda0.0_run2',
        'q2_pg_q5_b2000_r0.001_lambda0.95_run2', 
        'q2_pg_q5_b2000_r0.001_lambda0.98_run2', 
        'q2_pg_q5_b2000_r0.001_lambda0.99_run2', 
        'q2_pg_q5_b2000_r0.001_lambda1.0_run2' ])
plot_tb(
    list_of_data = hopper_data, 
    list_of_labels = ['Lambda = 0.0', 'Lambda = 0.95', 'Lambda = 0.98', 'Lambda = 0.99', 'Lambda = 1'], 
    title = 'GAE Hopper-v4 Returns',
    output_file = 'gae_hopper.png')


# cheetah_data = get_tb_returns(
#     list_of_path_patterns = [
#         'q2_pg_q4_search_b10000_lr0.005', #'q2_pg_q4_search_b10000_lr0.01', 'q2_pg_q4_search_b10000_lr0.005', 
#         'q2_pg_q4_search_b30000_lr0.005', #'q2_pg_q4_search_b30000_lr0.01', 'q2_pg_q4_search_b30000_lr0.005', 
#         'q2_pg_q4_search_b50000_lr0.005'], #'q2_pg_q4_search_b50000_lr0.01', 'q2_pg_q4_search_b50000_lr0.005']
# )
# plot_tb(
#     list_of_data = cheetah_data, 
#     list_of_labels = [
# 'b10000_lr0.005'#, 'b10000_lr0.01', 'b10000_lr0.005', 
#         'b30000_lr0.005'#, 'b30000_lr0.01', 'b30000_lr0.005', 
#         'b50000_lr0.005'],#, 'b50000_lr0.01', 'b50000_lr0.005'], 
#     title = 'Half Cheetah Search Returns',
#     output_file = 'half_cheetah_search_0.005.png')

# cheetah_data = get_tb_returns(
#     list_of_path_patterns=[
#         'q2_pg_q4_b50000_r0.02_rtg_nnbaseline_HalfCheetah-v4', 'q2_pg_q4_b50000_r0.02_nnbaseline_HalfCheetah-v4', 
#         'q2_pg_q4_b50000_r_0.02_rtg_HalfCheetah-v4', 'q2_pg_q4_b50000_r0.02_HalfCheetah'
#     ]
# )
# plot_tb(
#     list_of_data = cheetah_data, 
#     list_of_labels = ['rtg_nnbaseline', 'nnbaseline', 'rtg', 'neither'], 
#     title = 'Half Cheetah Returns for b = 50,000; lr = 0.02', 
#     output_file = 'half_cheetah_best.png'
# )