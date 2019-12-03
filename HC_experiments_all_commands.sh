#!/usr/bin/env bash


#####################Train#######################################
softlearning run_example_local examples.development --universe=gym --domain=HalfCheetah --task=Energy0-v0 --exp-name=HC_E0_r1    --checkpoint-frequency=100   --trial-gpus 1    --algorithm SAC

softlearning run_example_local examples.development --universe=gym --domain=HalfCheetah --task=Energy0-v0 --exp-name=HC_E0_r2    --checkpoint-frequency=100   --trial-gpus 1    --algorithm SAC

softlearning run_example_local examples.development --universe=gym --domain=HalfCheetah --task=Energy0-v0 --exp-name=HC_E0_r3    --checkpoint-frequency=100   --trial-gpus 1    --algorithm SAC

softlearning run_example_local examples.development --universe=gym --domain=HalfCheetah --task=Energy0-v0 --exp-name=HC_E0_r4    --checkpoint-frequency=100   --trial-gpus 1    --algorithm SAC

softlearning run_example_local examples.development --universe=gym --domain=HalfCheetah --task=Energy0-v0 --exp-name=HC_E0_r5    --checkpoint-frequency=100   --trial-gpus 1    --algorithm SAC


softlearning run_example_local examples.development_TD3 --universe=gym --domain=HalfCheetah --task=Energy0-v0 --exp-name=HC_E0_TD3_r1   --checkpoint-frequency=100   --trial-gpus 1    --algorithm TD3 --policy deterministicsPolicy

softlearning run_example_local examples.development_TD3 --universe=gym --domain=HalfCheetah --task=Energy0-v0 --exp-name=HC_E0_TD3_r2   --checkpoint-frequency=100   --trial-gpus 1    --algorithm TD3 --policy deterministicsPolicy

softlearning run_example_local examples.development_TD3 --universe=gym --domain=HalfCheetah --task=Energy0-v0 --exp-name=HC_E0_TD3_r3   --checkpoint-frequency=100   --trial-gpus 1    --algorithm TD3 --policy deterministicsPolicy

softlearning run_example_local examples.development_TD3 --universe=gym --domain=HalfCheetah --task=Energy0-v0 --exp-name=HC_E0_TD3_r4   --checkpoint-frequency=100   --trial-gpus 1    --algorithm TD3 --policy deterministicsPolicy

softlearning run_example_local examples.development_TD3 --universe=gym --domain=HalfCheetah --task=Energy0-v0 --exp-name=HC_E0_TD3_r5   --checkpoint-frequency=100   --trial-gpus 1    --algorithm TD3 --policy deterministicsPolicy
#####################Train#######################################

#####################Collect actions after training#######################################

python examples/development/collect_actions_SAC.py --agent HalfCheetah

python examples/development_TD3/collect_actions_TD3.py --agent HalfCheetah

#####################Collect actions after training#######################################


############### Synergy development graph ###################################

python examples/plotting/AdaptiveW_Extract_synergy_HC_compare_PI_spatiotemporal_evolution.py --tr   _r1 _r2 _r3 _r4 _r5 --ee E0 --agentt HC


python examples/plotting/AdaptiveW_Extract_synergy_HC_compare_PI_spatiotemporal_evolution.py --tr   _r1 _r2 _r3 _r4 _r5   --ee E0_TD3 --agentt HC


python examples/plotting/AdaptiveW_surface_area_spatiotemporal_evolution.py --tr    _r1 _r2 _r3 _r4 _r5   --ee E0 --agentt HC

python examples/plotting/AdaptiveW_surface_area_spatiotemporal_evolution.py --tr   _r1 _r2 _r3 _r4 _r5    --ee E0_TD3 --agentt HC


############### Synergy development graph ###################################

############### Histograms  ###################################

python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial.py  --tr   _r1 _r2 _r3 _r4 _r5  --ee E0 --agentt HC

python examples/plotting/AdaptiveW_Extract_SA_P_PI_corr_each_trial.py  --tr     _r1 _r2 _r3 _r4 _r5  --ee E0_TD3 --agentt HC


python examples/plotting/AdaptiveW_process_SA.py  --agentt HC

python examples/plotting/AdaptiveW_SA_summary.py  --agentt HC


python examples/plotting/AdaptiveW_plot_summary_histogram.py

python examples/plotting/AdaptiveW_plot_summary_histogram_performance.py

############### Histograms ###################################



