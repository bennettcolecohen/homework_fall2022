from tensorflow.python.summary.summary_iterator import summary_iterator
import numpy as np
import matplotlib.pyplot as plt
import os

def read_data(path): 

    full_path = 'data/' + [el for el in os.listdir('data') if path in el][0]
    
    full_path += '/' + [el for el in os.listdir(full_path) if 'event' in el][0]
    train_returns = []
    eval_returns = []
    print(full_path)
    for e in summary_iterator(full_path): 
        for v in e.summary.value: 
            if v.tag == 'Train_AverageReturn': 
                train_returns.append(v.simple_value)
            elif v.tag == 'Eval_AverageReturn': 
                eval_returns.append(v.simple_value)

    return train_returns, eval_returns


def plot_data(x, list_of_y, list_of_labels, title, xlabel, ylabel, legend, output_file):

    fig, ax = plt.subplots(figsize = (8,6))

    for idx, y in enumerate(list_of_y): 
        plt.plot(x, y, label = list_of_labels[idx])

    plt.title(title), plt.xlabel(xlabel), plt.ylabel(ylabel)
    if legend: 
        plt.legend()
    fig.savefig(output_file)



# y_data = [read_data('hw4_q4_reacher_horizon5')[1], 
#           read_data('hw4_q4_reacher_horizon15')[1], 
#           read_data('hw4_q4_reacher_horizon30')[1]]
# plot_data(np.arange(15), y_data, ['Horizon = 5', 'Horizon = 15', 'Horizon = 30'], 
# 'Effect of Horizon on Eval_AverageReturn', 'Iteration', 'Eval_AverageReturn', True, 'q4_horizon_plot.png')


# y_data_2 = [read_data('hw4_q4_reacher_numseq100')[1], 
#           read_data('hw4_q4_reacher_numseq1000')[1]
#           ]
# plot_data(np.arange(15), y_data_2, ['numseq = 100', 'numseq = 1000'], 
# 'Effect of Num. of Candidate Sequences on Eval_AverageReturn', 'Iteration', 'Eval_AverageReturn', True, 'q4_numseq_plot.png')


# y_data_3 = [read_data('q4_reacher_ensemble1')[1], 
#           read_data('q4_reacher_ensemble3')[1], 
#           read_data('q4_reacher_ensemble5')[1]]
# plot_data(np.arange(15), y_data_3, ['Ensemble = 1', 'Ensemble = 3', 'Ensemble = 5'], 
# 'Effect of Ensemble Size on Eval_AverageReturn', 'Iteration', 'Eval_AverageReturn', True, 'q4_ensemble_plot.png')


y_data_5 = [read_data('q5_cheetah_random')[1], 
          read_data('q5_cheetah_cem_2')[1], 
          read_data('q5_cheetah_cem_4')[1]]
plot_data(np.arange(len(y_data_5[0])), y_data_5, ['Random', 'CEM_2', 'CEM_4'], 
'CEM vs Random Shooting on Cheeteah', 'Iteration', 'Eval_AverageReturn', True, 'q5_plot.png')
