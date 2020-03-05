"""Custom Gym environments.

Every class inside this module should extend a gym.Env class. The file
structure should be similar to gym.envs file structure, e.g. if you're
implementing a mujoco env, you would implement it under gym.mujoco submodule.
"""

import gym


CUSTOM_GYM_ENVIRONMENTS_PATH = __package__
MUJOCO_ENVIRONMENTS_PATH = f'{CUSTOM_GYM_ENVIRONMENTS_PATH}.mujoco'

MUJOCO_ENVIRONMENT_SPECS = (
    {
        'id': 'VMPDv2-E0-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.vertical_mvt_pendulum_v2:VerticalMvtPendulumEnvDV2'),
    },
    {
        'id': 'VMPD-E0-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.vertical_mvt_pendulum:VerticalMvtPendulumDEnv'),
    },
    {
        'id': 'VMPv2-E0-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.vertical_mvt_pendulum_v2:VerticalMvtPendulumEnvV2'),
    },
    {
        'id': 'VMP-E0-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.vertical_mvt_pendulum:VerticalMvtPendulumEnv'),
    },
    {
        'id': 'Swimmer-Parameterizable-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.swimmer:SwimmerEnv'),
    },
    {
        'id': 'Swimmer-Energy0-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.swimmer:SwimmerEnv'),
    },
    {
        'id': 'Hopper-Energy0-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.hopper:HopperEnv'),
    },
    {
        'id': 'Hopper-EnergyPoint5-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.hopper:HopperEnv'),
    },
    {
        'id': 'Hopper-EnergyOne-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.hopper:HopperEnv'),
    },
    {
        'id': 'Hopper-EnergyOnePoint5-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.hopper:HopperEnv'),
    },
    {
        'id': 'Hopper-EnergyTwo-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.hopper:HopperEnv'),
    },
    {
        'id': 'Bipedal2d-Energy0-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.bipedal_2:Bipedal2Env'),
    },
    {
        'id': 'Walker2d-Energy0-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.walker2d:Walker2dEnv'),
    },
    {
        'id': 'Walker2d-EnergyPoint5-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.walker2d:Walker2dEnv'),
    },
    {
        'id': 'Walker2d-EnergyOne-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.walker2d:Walker2dEnv'),
    },
    {
        'id': 'Walker2d-EnergyOnePoint5-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.walker2d:Walker2dEnv'),
    },
    {
        'id': 'Walker2d-EnergyTwo-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.walker2d:Walker2dEnv'),
    },
    {
        'id': 'Walker2d-EnergyFour-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.walker2d:Walker2dEnv'),
    },
    {
        'id': 'Walker2d-EnergySix-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.walker2d:Walker2dEnv'),
    },
    {
        'id': 'HalfCheetah-EnergySix-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:HalfCheetahEnv'),
    },
    {
        'id': 'HalfCheetah-EnergyFour-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:HalfCheetahEnv'),
    },
    {
        'id': 'HalfCheetah-EnergyTwo-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:HalfCheetahEnv'),
    },
    {
        'id': 'HalfCheetah-EnergyOnePoint5-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:HalfCheetahEnv'),
    },
    {
        'id': 'HalfCheetah-EnergyOne-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:HalfCheetahEnv'),
    },
    {
        'id': 'HalfCheetah-EnergyPoint5-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:HalfCheetahEnv'),
    },
    {
        'id': 'HalfCheetah-EnergyPoint1-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:HalfCheetahEnv'),
    },
    {
        'id': 'HalfCheetah-Energy0-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:HalfCheetahEnv'),
    },
    {
        'id': 'HalfCheetah5dof-Energy0-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:HalfCheetahEnv_5dof'),
    },
    {
        'id': 'HalfCheetah4dof-Energy0-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:HalfCheetahEnv_4dof'),
    },
    {
        'id': 'HalfCheetah3doff-Energy0-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:HalfCheetahEnv_3doff'),
    },
    {
        'id': 'HalfCheetah3dofb-Energy0-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:HalfCheetahEnv_3dofb'),
    },
    {
        'id': 'HalfCheetah2dof-Energy0-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:HalfCheetahEnv_2dof'),
    },
    {
        'id': 'Giraffe-Energy0-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.giraffe:GiraffeEnv'),
    },
    {
        'id': 'HalfCheetahHeavy-Energy0-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:HalfCheetahHeavyEnv'),
    },
    {
        'id': 'VA-Energy0-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.vertical_arm:VA'),
    },
    {
        'id': 'VA4dof-Energy0-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.vertical_arm:VA4dof'),
    },
    {
        'id': 'VA6dof-Energy0-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.vertical_arm:VA6dof'),
    },
    {
        'id': 'VA8dof-Energy0-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.vertical_arm:VA8dof'),
    },
    {
    'id': 'Centripede-Energy0-v0',
    'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                    '.centripede:CentripedeEnv'),
    },
    {
        'id': 'FullCheetah-Energy0-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },
    {
        'id': 'HalfCheetah-PerfIndex-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
        '.half_cheetah:HalfCheetahEnv2'),
    },
    {
        'id': 'HalfCheetah-InvPerfIndex-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:HalfCheetahEnv3'),
    },

    {
        'id': 'Ant-Energy0-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.ant:AntEnv'),
    },
    {
        'id': 'AntHeavy-Energy0-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.ant:AntHeavyEnv'),
    },
    {
        'id': 'Ant-EnergyPoint5-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.ant:AntEnv'),
    },
    {
        'id': 'Ant-EnergyOne-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.ant:AntEnv'),
    },
    {
        'id': 'Ant-EnergyOnePoint5-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.ant:AntEnv'),
    },
    {
        'id': 'Ant-EnergyTwo-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.ant:AntEnv'),
    },
    {
        'id': 'Humanoid-Energy0-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.humanoid:HumanoidEnv'),
    },
    {
        'id': 'Humanoid-EnergyOne-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.humanoid:HumanoidEnv'),
    },
    {
        'id': 'Humanoid-EnergyPoint5-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.humanoid:HumanoidEnv'),
    },
    {
        'id': 'Humanoid-EnergyPoint1-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.humanoid:HumanoidEnv'),
    },
    {
        'id': 'Humanoid-EnergyPz5-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.humanoid:HumanoidEnv'),
    },
    {
        'id': 'Humanoidrllab-Energy0-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.humanoid_rllab:HumanoidEnv'),
    },
    {
        'id': 'Humanoidrllab-EnergyOne-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.humanoid_rllab:HumanoidEnv'),
    },
    {
        'id': 'Humanoidrllab-EnergyP5-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.humanoid_rllab:HumanoidEnv'),
    },
    {
        'id': 'Humanoidrllab-EnergyP1-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.humanoid_rllab:HumanoidEnv'),
    },
    {
        'id': 'Humanoidrllab-EnergyPz5-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.humanoid_rllab:HumanoidEnv'),
    },
    {
        'id': 'Pusher2d-Default-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.pusher_2d:Pusher2dEnv'),
    },
    {
        'id': 'Pusher2d-DefaultReach-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.pusher_2d:ForkReacherEnv'),
    },
    {
        'id': 'Pusher2d-ImageDefault-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.image_pusher_2d:ImagePusher2dEnv'),
    },
    {
        'id': 'Pusher2d-ImageReach-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.image_pusher_2d:ImageForkReacher2dEnv'),
    },
    {
        'id': 'Pusher2d-BlindReach-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.image_pusher_2d:BlindForkReacher2dEnv'),
    },
)

GENERAL_ENVIRONMENT_SPECS = (
    {
        'id': 'MultiGoal-Default-v0',
        'entry_point': (f'{CUSTOM_GYM_ENVIRONMENTS_PATH}'
                        '.multi_goal:MultiGoalEnv')
    },
)

MULTIWORLD_ENVIRONMENT_SPECS = (
    {
        'id': 'Point2DEnv-Default-v0',
        'entry_point': 'multiworld.envs.pygame.point2d:Point2DWallEnv'
    },
    {
        'id': 'Point2DEnv-Wall-v0',
        'entry_point': 'multiworld.envs.pygame.point2d:Point2DWallEnv'
    },
)

MUJOCO_ENVIRONMENTS = tuple(
    environment_spec['id']
    for environment_spec in MUJOCO_ENVIRONMENT_SPECS)


GENERAL_ENVIRONMENTS = tuple(
    environment_spec['id']
    for environment_spec in GENERAL_ENVIRONMENT_SPECS)


MULTIWORLD_ENVIRONMENTS = tuple(
    environment_spec['id']
    for environment_spec in MULTIWORLD_ENVIRONMENT_SPECS)

GYM_ENVIRONMENTS = (
    *MUJOCO_ENVIRONMENTS,
    *GENERAL_ENVIRONMENTS,
    *MULTIWORLD_ENVIRONMENTS,
)


def register_mujoco_environments():
    """Register softlearning mujoco environments."""
    for mujoco_environment in MUJOCO_ENVIRONMENT_SPECS:
        gym.register(**mujoco_environment)

    gym_ids = tuple(
        environment_spec['id']
        for environment_spec in  MUJOCO_ENVIRONMENT_SPECS)

    return gym_ids


def register_general_environments():
    """Register gym environments that don't fall under a specific category."""
    for general_environment in GENERAL_ENVIRONMENT_SPECS:
        gym.register(**general_environment)

    gym_ids = tuple(
        environment_spec['id']
        for environment_spec in  GENERAL_ENVIRONMENT_SPECS)

    return gym_ids


def register_multiworld_environments():
    """Register custom environments from multiworld package."""
    for multiworld_environment in MULTIWORLD_ENVIRONMENT_SPECS:
        gym.register(**multiworld_environment)

    gym_ids = tuple(
        environment_spec['id']
        for environment_spec in  MULTIWORLD_ENVIRONMENT_SPECS)

    return gym_ids


def register_environments():
    registered_mujoco_environments = register_mujoco_environments()
    registered_general_environments = register_general_environments()
    registered_multiworld_environments = register_multiworld_environments()

    return (
        *registered_mujoco_environments,
        *registered_general_environments,
        *registered_multiworld_environments,
    )
