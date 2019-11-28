import argparse
from distutils.util import strtobool
import json
import os
import pickle

import tensorflow as tf

from softlearning.policies.utils import get_policy_from_variant
from softlearning.samplers import rollouts

from gym import wrappers

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint_path',
                        type=str,
                        help='Path to the checkpoint.')
    parser.add_argument('--max-path-length', '-l', type=int, default=1000)
    parser.add_argument('--num-rollouts', '-n', type=int, default=10)
    parser.add_argument('--render-mode', '-r',
                        type=str,
                        default='human',
                        choices=('human', 'rgb_array', None,'No'),
                        help="Mode to render the rollouts in.")
    parser.add_argument('--deterministic', '-d',
                        type=lambda x: bool(strtobool(x)),
                        nargs='?',
                        const=True,
                        default=True,
                        help="Evaluate policy deterministically.")

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
    #env = wrappers.Monitor(env, '/home/jzchai/PycharmProjects/softlearning/examples/plotting/Synergy', force=True)

    with policy.set_deterministic	(args.deterministic):
        paths = rollouts(env=env,
                         policy=policy,
                         path_length=args.max_path_length,
                         n_paths=args.num_rollouts,
                         render_mode=args.render_mode)

    if args.render_mode != 'human':
        from pprint import pprint; import pdb; pdb.set_trace()
        pass

    return paths


if __name__ == '__main__':
    args = parse_args()
    simulate_policy(args)
