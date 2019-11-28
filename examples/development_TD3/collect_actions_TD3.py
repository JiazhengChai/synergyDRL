import argparse
from distutils.util import strtobool
import json
import os
import pickle
from collections import OrderedDict
import tensorflow as tf
import numpy as np

from softlearning.policies.utils import get_policy_from_variant
from softlearning.samplers import rollouts,my_rollouts


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--agent',
                        type=str)

    parser.add_argument('--energy',
                        type=str, default='Energy0-v0')

    parser.add_argument('--start', '-s', type=int,default=100)
    parser.add_argument('--final', '-f', type=int,default=3000)
    parser.add_argument('--step', '-st', type=int,default=100)


    parser.add_argument('--max-path-length', '-l', type=int, default=1000)
    parser.add_argument('--num-rollouts', '-n', type=int, default=10)
    parser.add_argument('--render-mode', '-r',
                        type=str,
                        default=None,
                        choices=('human', 'rgb_array', None),
                        help="Mode to render the rollouts in.")
    parser.add_argument('--deterministic', '-d',
                        type=lambda x: bool(strtobool(x)),
                        nargs='?',
                        const=True,
                        default=True,
                        help="Evaluate policy deterministically.")
    parser.add_argument('--name',
                        type=str,
                        help='Experiment name')

    args = parser.parse_args()

    return args


def simulate_policy(args):
    gpu_options = tf.GPUOptions(allow_growth=True)
    session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    tf.keras.backend.set_session(session)

    session = tf.keras.backend.get_session()
    checkpoint_path = args.checkpoint_path.rstrip('/')
    experiment_path = os.path.dirname(checkpoint_path)

    variant_path = os.path.join(experiment_path, 'params.json')
    with open(variant_path, 'r') as f:
        variant = json.load(f)

    with session.as_default():
        pickle_path = os.path.join(checkpoint_path, 'checkpoint.pkl')
        with open(pickle_path, 'rb') as f:
            picklable = pickle.load(f)

    env = picklable['env']
    policy = (
        get_policy_from_variant(variant, env))
    policy.set_weights(picklable['policy_weights'])

    with policy.set_deterministic	(args.deterministic):
        paths = my_rollouts(env=env,
                         policy=policy,
                         path_length=args.max_path_length,
                         n_paths=args.num_rollouts,
                         render_mode=args.render_mode)


    return paths


if __name__ == '__main__':
    args = parse_args()

    agent=args.agent
    energy=args.energy
    top_path='./experiments_results/gym/'+agent+'/'+energy

    if 'Energy0' in energy:
        ene_sub='_E0_TD3'
    elif 'EnergyOne' in energy:
        ene_sub = '_E1_TD3'

    if agent=='HalfCheetah':
        abrv='HC'
    elif agent=='HalfCheetahHeavy':
        abrv = 'HCheavy'
    elif agent=='FullCheetah':
        abrv = 'FC'

    for experiment in os.listdir(top_path):
        exp_path=os.path.join(top_path,experiment)
        if 'TD3' in experiment:
            base_name=abrv+ene_sub


            trial='_'+experiment.split('_')[-1]

            for folder in os.listdir(exp_path):
                if 'ExperimentRunner' in folder:
                    base_path=os.path.join(exp_path,folder)

            start=args.start
            step=args.step
            final=args.final

            all_checkpoint = []
            all_name =[]
            for ch in range(start,final+1,step):
                specific='checkpoint_'+str(ch)
                all_checkpoint.append(os.path.join(base_path, specific))
                namee = base_name + '_C' + str(ch) + trial

                all_name.append(namee)

            for ind,chk in enumerate(all_checkpoint):
                args.checkpoint_path=chk
                args.name=all_name[ind]

                paths=simulate_policy(args)

                total_ori_reward = []
                total_energy = []

                action_list=[]
                states_list = []
                for path in paths:
                    try:
                        tmp = 0
                        tmpe=0
                        for i in range(len(path['infos'])):
                            tmp = tmp + path['infos'][i]['ori_reward']
                            tmpe = tmpe + path['infos'][i]['energy']
                        total_ori_reward.append(tmp)
                        total_energy.append(tmpe)
                    except:
                        pass

                    action_list.append(path['actions'])
                    states_list.append(path['states'])


                action_list=np.asarray(action_list)
                states_list = np.asarray(states_list)
                name = args.name
                print(name)

                total_energy = np.asarray(total_energy)
                if not os.path.exists('./experiments_results/collected_actions/trajectory_npy/reward_energy_dict'):
                    os.makedirs('./experiments_results/collected_actions/trajectory_npy/reward_energy_dict',exist_ok=True)

                if not os.path.exists('./experiments_results/collected_actions/trajectory_npy/actions_npy'):
                    os.makedirs('./experiments_results/collected_actions/trajectory_npy/actions_npy',exist_ok=True)

                if not os.path.exists('./experiments_results/collected_actions/trajectory_npy/states_npy'):
                    os.makedirs('./experiments_results/collected_actions/trajectory_npy/states_npy',exist_ok=True)

                try:
                    diagnostics = OrderedDict((
                        ('ori-return-average', np.mean(total_ori_reward)),
                        ('ori-return-min', np.min(total_ori_reward)),
                        ('ori-return-max', np.max(total_ori_reward)),
                        ('ori-return-std', np.std(total_ori_reward)),

                        ('total-energy-average', np.mean(total_energy)),
                        ('total-energy-min', np.min(total_energy)),
                        ('total-energy-max', np.max(total_energy)),
                        ('total-energy-std', np.std(total_energy)),
                    ))

                    np.save('./experiments_results/collected_actions/trajectory_npy/reward_energy_dict/' + name, diagnostics)

                except:
                    pass

                np.save('./experiments_results/collected_actions/trajectory_npy/actions_npy/' + name, action_list)
                np.save('./experiments_results/collected_actions/trajectory_npy/states_npy/' + name, states_list)



