#!/usr/bin/env bash

####Train#############################################################

#softlearning run_example_local examples.development --universe=gym --domain=VA6dof --task=Energy0-v0 --exp-name=VA6dof_E0_r1  --checkpoint-frequency=1   --trial-gpus 1    --algorithm SAC  --epoch_length 500 --total_epoch 30 --actor_size 256 --critic_size 256

#softlearning run_example_local examples.development_TD3 --universe=gym --domain=VA6dof --task=Energy0-v0 --exp-name=VA6dof_E0_TD3_r5  --checkpoint-frequency=1   --trial-gpus 1    --algorithm TD3  --epoch_length 500 --total_epoch 30 --actor_size 256 --critic_size 256 --policy deterministicsPolicy

######################################################################


####Collect action########################################################
#python examples/development/collect_actions_SAC.py  --agent VA6dof --tr  _r10 --start 13 --final 390 --step 13

#python examples/development_TD3/collect_actions_TD3.py  --agent VA6dof --tr   _r8 _r9 --start 13 --final 390 --step 13

#OR

#python examples/development/collect_actions_SAC.py --path  /home/jzchai/PycharmProjects/synergy_analysis/experiments_results/gym/VA6dof/Energy0-v0/2020-03-03T14-21-27-VA6dof_E0_r5/ExperimentRunner_0_max_size=1000000,seed=69_2020-03-03_14-21-285cyt1ucs --start 1 --final 30 --step 1

#python examples/development_TD3/collect_actions_TD3.py --path /home/jzchai/PycharmProjects/synergy_analysis/experiments_results/gym/VA6dof/Energy0-v0/2020-03-03T13-47-31-VA6dof_E0_TD3_r1/ExperimentRunner_0_max_size=1000000,seed=2982_2020-03-03_13-47-3229_zgnu3 --start 1 --final 30 --step 1

#########################################################################


######### Main commands after training and collect actions###############
# Step 1 : Extract CSV files in raw_csv folder
#python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py --tr  _r6 _r7 _r8 _r9 _r10  --ee E0 --agentt VA

# Step 2 : Process the CSV files in raw_csv folder.
#python examples/plotting/AdaptiveW_process_SA.py --agentt VA
#python examples/plotting/AdaptiveW_SA_summary.py --agentt VA

#Step 3 :
#python examples/plotting/learning_progress_synergy.py
#python examples/plotting/AdaptiveW_plot_summary_histogram.py
#python examples/plotting/AdaptiveW_plot_summary_histogram_performance.py
#python examples/plotting/AdaptiveW_plot_summary_three_histograms.py
#python examples/plotting/AdaptiveW_plot_summary_three_histograms_performance.py
#python examples/plotting/learning_progress_compare_synergy.py
#python examples/plotting/compare_dof_synergy_lineplot.py

###########################################################################
