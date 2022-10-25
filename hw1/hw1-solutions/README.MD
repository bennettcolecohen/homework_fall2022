The following are instructions on how to run code to get outputs for both behavior cloning and DAgger. 
Note that these should be run from the directory hw1.


# Question 1.2: Behavior Cloning

### AntV4 Environment: 

Please run the following command:

```
python cs285/scripts/run_hw1.py --expert_policy_file cs285/policies/experts/Ant.pkl --env_name Ant-v4 --exp_name bc_ant --n_iter 1 --expert_data cs285/expert_data/expert_data_Ant-v4.pkl --video_log_freq -1 --eval_batch_size 100000 --batch_size 100000
```

### Walker2d Environment: 

Please run the following command: 
```
python cs285/scripts/run_hw1.py --expert_policy_file cs285/policies/experts/Walker2d.pkl --env_name Walker2d-v4 --exp_name bc_walker --n_iter 1 --expert_data cs285/expert_data/expert_data_Walker2d-v4.pkl --video_log_freq -1 --eval_batch_size 100000 --batch_size 100000
```

# Question 1.3: 

In order test my 1.3, simply run the command below. Further, in the train_agent method of rl_trainer you must
uncomment out lines 207 and 208 to log within a single iteration. Recall this log is not kept within the run_logs 
folder. 
```
python cs285/scripts/run_hw1.py --expert_policy_file cs285/policies/experts/Walker2d.pkl --env_name Walker2d-v4 --exp_name bc_walker_tuned --n_iter 1 --expert_data cs285/expert_data/expert_data_Walker2d-v4.pkl --video_log_freq -1 --eval_batch_size 100000 --batch_size 100000 --num_agent_train_steps_per_iter 50000 
```

# Question 2.2

### AntV4 Environment: 

Please run the following command: 
```
python cs285/scripts/run_hw1.py --expert_policy_file cs285/policies/experts/Ant.pkl --env_name Ant-v4 --exp_name dagger_ant --n_iter 20 --do_dagger --expert_data cs285/expert_data/expert_data_Ant-v4.pkl --video_log_freq -1 --eval_batch_size 10000 --batch_size 10000
```


### Walker2d Environment: 

Please run the following command: 
```
python cs285/scripts/run_hw1.py --expert_policy_file cs285/policies/experts/Walker2d.pkl --env_name Walker2d-v4 --exp_name dagger_walker --n_iter 20 --do_dagger --expert_data cs285/expert_data/expert_data_Walker2d-v4.pkl --video_log_freq -1 --eval_batch_size 10000 --batch_size 10000
```


# Other Notes: 

- I have added a script within the *scripts* directory called parse_tb.py which will create the plots for the 
DAgger performance as a function of iterations, along with error bars, etc. It will take the first file 
within the path given to the log directory and save a png to the root directory. 

- The only other function I have added is part of rl_trained called perform_intra_episode_logging which let 
me plot the EvalReturn as a function of steps in a single iterations (avoids training multiple times). It 
isn't called by default so to use it, uncomment out 207-208 in rl_trainer train_agent method. 