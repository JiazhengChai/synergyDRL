from ray import tune
import numpy as np

from softlearning.misc.utils import  deep_update

M = 256
N = 128#256
#N=45#human
REPARAMETERIZE = True

NUM_COUPLING_LAYERS = 2

GAUSSIAN_POLICY_PARAMS_BASE = {
    'type': 'GaussianPolicy',
    'kwargs': {
        'hidden_layer_sizes': (M, M),
        'squash': True,
    }
}

DETERMINISTICS_POLICY_PARAMS_BASE = {
    'type': 'DeterministicsPolicy',
    'kwargs': {
        'hidden_layer_sizes': (M, M),
        'squash': True
    }
}

RNN_GAUSSIAN_POLICY_PARAMS_BASE = {
    'type': 'RnnGaussianPolicy',
    'kwargs': {
        'hidden_layer_sizes': (N,N),
        'squash': True,
    }
}
GRU_GAUSSIAN_POLICY_PARAMS_BASE = {
    'type': 'GruGaussianPolicy',
    'kwargs': {
        'hidden_layer_sizes': (N,N),
        'squash': True,
    }
}
LSTM_GAUSSIAN_POLICY_PARAMS_BASE = {
    'type': 'LstmGaussianPolicy',
    'kwargs': {
        'hidden_layer_sizes': (N,N),
        'squash': True,
    }
}

GAUSSIAN_POLICY_PARAMS_FOR_DOMAIN = {
    'Bipedal2d':{
        'kwargs': {
        'hidden_layer_sizes': (64,64),
        'squash': True,
        }
    }

}

DETERMINISTICS_POLICY_PARAMS_FOR_DOMAIN = {}

RNN_GAUSSIAN_POLICY_PARAMS_FOR_DOMAIN={}



POLICY_PARAMS_BASE = {
    'GaussianPolicy': GAUSSIAN_POLICY_PARAMS_BASE,
    'DeterministicsPolicy': DETERMINISTICS_POLICY_PARAMS_BASE,
    'RnnGaussianPolicy':RNN_GAUSSIAN_POLICY_PARAMS_BASE,
    'GruGaussianPolicy':GRU_GAUSSIAN_POLICY_PARAMS_BASE,
    'LstmGaussianPolicy':LSTM_GAUSSIAN_POLICY_PARAMS_BASE

}

POLICY_PARAMS_BASE.update({
    'gaussian': POLICY_PARAMS_BASE['GaussianPolicy'],
    'deterministicsPolicy': POLICY_PARAMS_BASE['DeterministicsPolicy'],
    'RnnGaussian':POLICY_PARAMS_BASE['RnnGaussianPolicy'],
    'GruGaussian':POLICY_PARAMS_BASE['GruGaussianPolicy'],
    'LstmGaussian':POLICY_PARAMS_BASE['LstmGaussianPolicy'],

})

POLICY_PARAMS_FOR_DOMAIN = {
    'GaussianPolicy': GAUSSIAN_POLICY_PARAMS_FOR_DOMAIN,

    'DeterministicsPolicy': DETERMINISTICS_POLICY_PARAMS_FOR_DOMAIN,

    'RnnGaussianPolicy':RNN_GAUSSIAN_POLICY_PARAMS_FOR_DOMAIN,

    'GruGaussianPolicy': RNN_GAUSSIAN_POLICY_PARAMS_FOR_DOMAIN,

    'LstmGaussianPolicy': RNN_GAUSSIAN_POLICY_PARAMS_FOR_DOMAIN
}

POLICY_PARAMS_FOR_DOMAIN.update({
    'gaussian': POLICY_PARAMS_FOR_DOMAIN['GaussianPolicy'],

    'deterministicsPolicy': POLICY_PARAMS_FOR_DOMAIN['DeterministicsPolicy'],

    'RnnGaussian':POLICY_PARAMS_FOR_DOMAIN['RnnGaussianPolicy'],

    'GruGaussian': POLICY_PARAMS_FOR_DOMAIN['GruGaussianPolicy'],

    'LstmGaussian': POLICY_PARAMS_FOR_DOMAIN['LstmGaussianPolicy'],

})

DEFAULT_MAX_PATH_LENGTH = 1000
MAX_PATH_LENGTH_PER_DOMAIN = {
    'Point2DEnv': 50,
    'Pendulum': 200,
}

ALGORITHM_PARAMS_BASE = {
    'type': 'SAC',

    'kwargs': {
        'epoch_length': 1000,
        'train_every_n_steps': 1,
        'n_train_repeat': 1,
        'eval_render_mode': None,
        'eval_n_episodes': 3,
        'eval_deterministic': True,

        'discount': 0.99,
        'tau': 5e-3,
        'reward_scale': 1.0,
    }
}


ALGORITHM_PARAMS_ADDITIONAL = {
    'SAC': {
        'type': 'SAC',
        'kwargs': {
            'reparameterize': REPARAMETERIZE,
            'lr': 3e-4,
            'target_update_interval': 1,
            'tau': 5e-3,
            'target_entropy': 'auto',
            'store_extra_policy_info': False,
            'action_prior': 'uniform',
            'n_initial_exploration_steps': int(1e3),
        }
    },

    'TD3': {
        'type': 'TD3',
        'kwargs': {
            'reparameterize': REPARAMETERIZE,
            'lr': 1e-3,
            'target_update_interval': 2,
            'tau': 5e-3,
            'store_extra_policy_info': False,
            'action_prior': 'uniform',
            'n_initial_exploration_steps': int(1e4),
        }
    },

    'SAC_noise_v2': {
        'type': 'SAC_noise_v2',
        'kwargs': {
            'reparameterize': REPARAMETERIZE,
            'lr': 3e-4,
            'target_update_interval': 1,

            'Q_update_interval': 1,
            'S_noise': True,
            'A_noise': True,
            'SA_noise': True,
            'state_q_noise_std': 0.0005,
            'action_q_noise_std': 0.001,

            't_S_noise': False,
            't_A_noise': False,
            't_SA_noise': False,

            'tau': 5e-3,
            'target_entropy': 'auto',
            'store_extra_policy_info': False,
            'action_prior': 'uniform',
            'n_initial_exploration_steps': int(1e3),
        }
    },

}

DEFAULT_NUM_EPOCHS = 200

NUM_EPOCHS_PER_DOMAIN = {
    'Swimmer': int(3e2),
    'Hopper': int(1e3),
    'HalfCheetah': int(3e3),
    'Giraffe': int(2e3),
    'HalfCheetahHeavy':int(3e3),
    'HalfCheetah5dof': int(3e3),
    'HalfCheetah4dof':int(3e3),
    'HalfCheetah2dof':int(3e3),
    'HalfCheetah3doff': int(3e3),
    'HalfCheetah3dofb': int(3e3),
    'FullCheetah':int(3e3),
    'Centripede':int(2e3),
    'Walker2d': int(1e3),
    'Bipedal2d':int(300),
    'Ant': int(2e3),
    'VA': int(30),
    'VA4dof': int(30),
    'VA6dof': int(30),
    'VA8dof': int(100),
    'AntHeavy': int(2e3),
    'Humanoid': int(5e3),#int(1e4),
    'Humanoidrllab': int(3e3),#int(1e4),
    'Pusher2d': int(2e3),
    'HandManipulatePen': int(1e4),
    'HandManipulateEgg': int(1e4),
    'HandManipulateBlock': int(1e4),
    'HandReach': int(1e4),
    'Point2DEnv': int(200),
    'Reacher': int(200),
    'Pendulum': 10,
    'VMP': 50,
}

ALGORITHM_PARAMS_PER_DOMAIN = {
    **{
        domain: {
            'kwargs': {
                'n_epochs': NUM_EPOCHS_PER_DOMAIN.get(
                    domain, DEFAULT_NUM_EPOCHS),
                'n_initial_exploration_steps': (
                    MAX_PATH_LENGTH_PER_DOMAIN.get(
                        domain, DEFAULT_MAX_PATH_LENGTH
                    ) * 10),
            }
        } for domain in NUM_EPOCHS_PER_DOMAIN
    }
}

ENV_PARAMS = {
    'Bipedal2d': {  # 6 DoF
        'Energy0-v0': {
            'target_energy':3
        },

    },
    'VA': {  # 6 DoF
        'Energy0-v0': {
            'distance_reward_weight':5.0,
            'ctrl_cost_weight':0.05,
        },
    },
    'VA4dof': {  # 6 DoF
        'Energy0-v0': {
            'distance_reward_weight': 5.0,
            'ctrl_cost_weight': 0.05,
        },
    },
    'VA6dof': {  # 6 DoF
        'Energy0-v0': {
            'distance_reward_weight': 5.0,
            'ctrl_cost_weight': 0.05,
        },
    },
    'VA8dof': {  # 6 DoF
        'Energy0-v0': {
            'distance_reward_weight': 5.0,
            'ctrl_cost_weight': 0.05,#0.05
        },
    },
    'HalfCheetahHeavy': {  # 6 DoF
        'Energy0-v0': {
            'forward_reward_weight':1.0,
            'ctrl_cost_weight':0.1,
            'energy_weights':0,
        },

    },
    'HalfCheetah5dof': {  # 6 DoF
        'Energy0-v0': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
        },

    },
    'HalfCheetah4dof': {  # 6 DoF
        'Energy0-v0': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
        },

    },
    'HalfCheetah3doff': {  # 6 DoF
        'Energy0-v0': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
        },

    },
    'HalfCheetah3dofb': {  # 6 DoF
        'Energy0-v0': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
        },

    },
    'HalfCheetah2dof': {  # 6 DoF
        'Energy0-v0': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
        },

    },
    'HalfCheetah': {  # 6 DoF
        'EnergySix-v0': {
            'forward_reward_weight':1.0,
            'ctrl_cost_weight':0.1,
            'energy_weights':6.0,

        },
        'EnergyFour-v0': {
            'forward_reward_weight':1.0,
            'ctrl_cost_weight':0.1,
            'energy_weights':4.0,

        },
        'EnergyTwo-v0': {
            'forward_reward_weight':1.0,
            'ctrl_cost_weight':0.1,
            'energy_weights':2.0,
        },
        'EnergyOnePoint5-v0': {
            'forward_reward_weight':1.0,
            'ctrl_cost_weight':0.1,
            'energy_weights':1.5,
        },
        'EnergyOne-v0': {
            'forward_reward_weight':1.0,
            'ctrl_cost_weight':0.1,
            'energy_weights':1.,
        },
        'EnergyPoint5-v0': {
            'forward_reward_weight':1.0,
            'ctrl_cost_weight':0.1,
            'energy_weights':0.5,
        },
        'EnergyPoint1-v0': {
            'forward_reward_weight':1.0,
            'ctrl_cost_weight':0.1,
            'energy_weights':0.1,
        },
        'Energy0-v0': {
            'forward_reward_weight':1.0,
            'ctrl_cost_weight':0.1,
            'energy_weights':0,
        },

    },

    'FullCheetah': {  # 6 DoF
        'Energy0-v0': {
            'forward_reward_weight':1.0,
            'ctrl_cost_weight':0.1,
            'energy_weights':0,
        },

    },


}

NUM_CHECKPOINTS = 10


def get_variant_spec_base(universe, domain, task, policy, algorithm,epoch_length,num_epoch,fixed_latent_exploration=None,actor_size=256,critic_size=256):
    if num_epoch is not None:
        ALGORITHM_PARAMS_PER_DOMAIN[domain]['kwargs']['n_epochs']=num_epoch

    algorithm_params = deep_update(
        ALGORITHM_PARAMS_BASE,
        ALGORITHM_PARAMS_PER_DOMAIN.get(domain, {})
    )

    ALGORITHM_PARAMS_ADDITIONAL[algorithm]['kwargs']['epoch_length']=epoch_length

    POLICY_PARAMS_BASE[policy]['kwargs']['hidden_layer_sizes']=(actor_size,actor_size)

    algorithm_params = deep_update(
        algorithm_params,
        ALGORITHM_PARAMS_ADDITIONAL.get(algorithm, {})
    )
    variant_spec = {
        'domain': domain,
        'task': task,
        'universe': universe,

        'env_params': ENV_PARAMS.get(domain, {}).get(task, {}),
        'policy_params': deep_update(
            POLICY_PARAMS_BASE[policy],
            POLICY_PARAMS_FOR_DOMAIN[policy].get(domain, {})
        ),
        'Q_params': {
            'type': 'double_feedforward_Q_function',
            'kwargs': {
                'hidden_layer_sizes': (critic_size,critic_size),#256
            }
        },
        'algorithm_params': algorithm_params,
        'replay_pool_params': {
            'type': 'SimpleReplayPool',
            'kwargs': {
                'max_size': tune.sample_from(lambda spec: (
                    {
                        'SimpleReplayPool': int(1e6),
                        'TrajectoryReplayPool': int(1e4),
                    }.get(
                        spec.get('config', spec)
                        ['replay_pool_params']
                        ['type'],
                        int(1e6))
                )),
            }
        },
        'sampler_params': {
            'type': 'SimpleSampler',
            'kwargs': {
                'max_path_length': MAX_PATH_LENGTH_PER_DOMAIN.get(
                    domain, epoch_length),#DEFAULT_MAX_PATH_LENGTH
                'min_pool_size': MAX_PATH_LENGTH_PER_DOMAIN.get(
                    domain, DEFAULT_MAX_PATH_LENGTH),
                'batch_size': 256,
            }
        },
        'run_params': {
            'seed': tune.sample_from(
                lambda spec: np.random.randint(0, 10000)),
            'checkpoint_at_end': True,
            'checkpoint_frequency': NUM_EPOCHS_PER_DOMAIN.get(
                domain, DEFAULT_NUM_EPOCHS) // NUM_CHECKPOINTS,
            'checkpoint_replay_pool': False,
        },
    }

    return variant_spec


def get_variant_spec_image(universe,
                           domain,
                           task,
                           policy,
                           algorithm,
                           *args,
                           **kwargs):
    variant_spec = get_variant_spec_base(
        universe, domain, task, policy, algorithm, *args, **kwargs)

    if 'image' in task.lower() or 'image' in domain.lower():
        preprocessor_params = {
            'type': 'convnet_preprocessor',
            'kwargs': {
                'image_shape': variant_spec['env_params']['image_shape'],
                'output_size': M,
                'conv_filters': (4, 4),
                'conv_kernel_sizes': ((3, 3), (3, 3)),
                'pool_type': 'MaxPool2D',
                'pool_sizes': ((2, 2), (2, 2)),
                'pool_strides': (2, 2),
                'dense_hidden_layer_sizes': (),
            },
        }
        variant_spec['policy_params']['kwargs']['preprocessor_params'] = (
            preprocessor_params.copy())
        variant_spec['Q_params']['kwargs']['preprocessor_params'] = (
            preprocessor_params.copy())

    return variant_spec


def get_variant_spec(args):
    universe, domain, task = args.universe, args.domain, args.task

    if ('image' in task.lower()
        or 'blind' in task.lower()
        or 'image' in domain.lower()):
        variant_spec = get_variant_spec_image(
            universe, domain, task, args.policy, args.algorithm)
    else:
        variant_spec = get_variant_spec_base(
            universe, domain, task, args.policy, args.algorithm,args.epoch_length,args.total_epoch,args.fixed_latent_exploration,args.actor_size,args.critic_size)

    if args.checkpoint_replay_pool is not None:
        variant_spec['run_params']['checkpoint_replay_pool'] = (
            args.checkpoint_replay_pool)

    return variant_spec
