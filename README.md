# Synergy analysis source code

# Our special notes
This implementation uses Tensorflow and it is based on the library Softlearning  https://github.com/rail-berkeley/softlearning.
We customized the softlearning codebase to run our experiments.
Author of modification: Chai Jiazheng e-mail:chai.jiazheng.q1@dc.tohoku.ac.jp

# Getting Started

## Prerequisites

The environment can be run locally using conda. For conda installation, you need to have [Conda](https://www.anaconda.com/distribution/) installed. Also, our environments currently require a [MuJoCo](https://www.roboti.us/license.html) license.

## Conda Installation

1. [Download](https://www.roboti.us/index.html) and install MuJoCo 1.50/2.0 from the MuJoCo website. We assume that the MuJoCo files are extracted to the default location (`~/.mujoco/mjpro150`) or (`~/.mujoco/mujoco200`) .

2. Copy your MuJoCo license key (mjke
y.txt) to ~/.mujoco/mjkey.txt:

3. Git clone codebase `synergyDRL`

4. Create and activate conda environment, install softlearning to enable command line interface.
```
cd ${synergyDRL_PATH}
conda env create -f environment.yml
conda activate synergy_analysis
cd ..
pip install -e ${synergyDRL_PATH}
```

The environment should be ready to run experiments. 
## Important
We use customized mujoco xml files and python files that can be found in MUJOCO_FILES in synergyDRL codebase. 
Move the files inside this folder into the mujoco folder in 'synergy_analysis' virtual environment inside anaconda3.
Typically, the path is:
anaconda3/envs/synergy_analysis/lib/python3.6/site-packages/gym/envs/mujoco


Finally, to deactivate and remove the conda environment:
```
conda deactivate
conda remove --name synergy_analysis --all
```

## To run and reproduce our results:
While in the folder synergyDRL, with the virtual environment synergy_analysis activated, 
1) ./HC_experiments_all_commands.sh
2) ./HeavyHC_experiments_all_commands.sh
3) ./FC_experiments_all_commands.sh
4) ./summary_graphs_results_production.sh
5) ./extract_synergy_pattern.sh

Users must run 1),2),3) before 4) and 5). Users are also encouraged to further parallelize the command lines in 1),2),3) to speed up the training and action collection of the three agents. The whole experiments take tremendous of time.

The results after running 1),2),3),4),5) are in "experiments_results" in the synergy_analysis codebase.



## Details from the original softlearning codebase(Extra)
### Training an agent
1. To train the agent
```
synergy_analysis run_example_local examples.development \
    --universe=gym \
    --domain=HalfCheetah \
    --task=Energy0-v0 \
    --exp-name=HC_E0_r1 \
    --checkpoint-frequency=100  # Save the checkpoint to resume training later
```


`examples.development.main` contains several different environments and there are more example scripts available in the  `/examples` folder. For more information about the agents and configurations, run the scripts with `--help` flag: `python ./examples/development/main.py --help`
```
optional arguments:
  -h, --help            show this help message and exit
  --universe {gym}
  --domain {...}
  --task {...}
  --num-samples NUM_SAMPLES
  --resources RESOURCES
                        Resources to allocate to ray process. Passed to
                        `ray.init`.
  --cpus CPUS           Cpus to allocate to ray process. Passed to `ray.init`.
  --gpus GPUS           Gpus to allocate to ray process. Passed to `ray.init`.
  --trial-resources TRIAL_RESOURCES
                        Resources to allocate for each trial. Passed to
                        `tune.run_experiments`.
  --trial-cpus TRIAL_CPUS
                        Resources to allocate for each trial. Passed to
                        `tune.run_experiments`.
  --trial-gpus TRIAL_GPUS
                        Resources to allocate for each trial. Passed to
                        `tune.run_experiments`.
  --trial-extra-cpus TRIAL_EXTRA_CPUS
                        Extra CPUs to reserve in case the trials need to
                        launch additional Ray actors that use CPUs.
  --trial-extra-gpus TRIAL_EXTRA_GPUS
                        Extra GPUs to reserve in case the trials need to
                        launch additional Ray actors that use GPUs.
  --checkpoint-frequency CHECKPOINT_FREQUENCY
                        Save the training checkpoint every this many epochs.
                        If set, takes precedence over
                        variant['run_params']['checkpoint_frequency'].
  --checkpoint-at-end CHECKPOINT_AT_END
                        Whether a checkpoint should be saved at the end of
                        training. If set, takes precedence over
                        variant['run_params']['checkpoint_at_end'].
  --restore RESTORE     Path to checkpoint. Only makes sense to set if running
                        1 trial. Defaults to None.
  --policy {gaussian}
  --env ENV
  --exp-name EXP_NAME
  --log-dir LOG_DIR
  --upload-dir UPLOAD_DIR
                        Optional URI to sync training results to (e.g.
                        s3://<bucket> or gs://<bucket>).
  --confirm-remote [CONFIRM_REMOTE]
                        Whether or not to query yes/no on remote run.
```


# References
The algorithms are based on the following papers:

*Soft Actor-Critic Algorithms and Applications*.</br>
Tuomas Haarnoja*, Aurick Zhou*, Kristian Hartikainen*, George Tucker, Sehoon Ha, Jie Tan, Vikash Kumar, Henry Zhu, Abhishek Gupta, Pieter Abbeel, and Sergey Levine.
arXiv preprint, 2018.</br>
[paper](https://arxiv.org/abs/1812.05905)  |  [videos](https://sites.google.com/view/sac-and-applications)

*Latent Space Policies for Hierarchical Reinforcement Learning*.</br>
Tuomas Haarnoja*, Kristian Hartikainen*, Pieter Abbeel, and Sergey Levine.
International Conference on Machine Learning (ICML), 2018.</br>
[paper](https://arxiv.org/abs/1804.02808) | [videos](https://sites.google.com/view/latent-space-deep-rl)

*Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor*.</br>
Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine.
International Conference on Machine Learning (ICML), 2018.</br>
[paper](https://arxiv.org/abs/1801.01290) | [videos](https://sites.google.com/view/soft-actor-critic)

*Composable Deep Reinforcement Learning for Robotic Manipulation*.</br>
Tuomas Haarnoja, Vitchyr Pong, Aurick Zhou, Murtaza Dalal, Pieter Abbeel, Sergey Levine.
International Conference on Robotics and Automation (ICRA), 2018.</br>
[paper](https://arxiv.org/abs/1803.06773) | [videos](https://sites.google.com/view/composing-real-world-policies)

*Reinforcement Learning with Deep Energy-Based Policies*.</br>
Tuomas Haarnoja*, Haoran Tang*, Pieter Abbeel, Sergey Levine.
International Conference on Machine Learning (ICML), 2017.</br>
[paper](https://arxiv.org/abs/1702.08165) | [videos](https://sites.google.com/view/softqlearning/home)

If Softlearning helps you in your academic research, you are encouraged to cite their paper. Here is an example bibtex:
```
@techreport{haarnoja2018sacapps,
  title={Soft Actor-Critic Algorithms and Applications},
  author={Tuomas Haarnoja, Aurick Zhou, Kristian Hartikainen, George Tucker, Sehoon Ha, Jie Tan, Vikash Kumar, Henry Zhu, Abhishek Gupta, Pieter Abbeel, and Sergey Levine},
  journal={arXiv preprint arXiv:1812.05905},
  year={2018}
}
```
