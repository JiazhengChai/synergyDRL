import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import os
from . import path

DEFAULT_CAMERA_CONFIG = {
    'distance': 4.0,
}


class HalfCheetahEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                 xml_file='half_cheetah.xml',
                 forward_reward_weight=1.0,
                 ctrl_cost_weight=0.1,
                 reset_noise_scale=0.1,
                 exclude_current_positions_from_observation=True,
                 energy_weights=0.):

        utils.EzPickle.__init__(**locals())

        self._forward_reward_weight = forward_reward_weight

        self.joint_list=['bthigh','bshin','bfoot','fthigh','fshin','ffoot']

        self._ctrl_cost_weight = ctrl_cost_weight
        self.energy_weights=energy_weights
        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation)
        global path
        mujoco_env.MujocoEnv.__init__(self, os.path.join(path, xml_file), 5)
        #mujoco_env.MujocoEnv.__init__(self, xml_file, 5)

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    def step(self, action):
        states_angle = []
        for j in self.joint_list:
            states_angle.append(self.sim.data.get_joint_qpos(j))
        #states=self._get_obs()
        x_position_before = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.sim.data.qpos[0]

        x_velocity = ((x_position_after - x_position_before)
                      / self.dt)

        ctrl_cost = self.control_cost(action)

        forward_reward = self._forward_reward_weight * x_velocity

        observation = self._get_obs()
        #next_states=observation
        next_states_angle = []
        for j in self.joint_list:
            next_states_angle.append(self.sim.data.get_joint_qpos(j))

        reward = forward_reward - ctrl_cost
        done = False

        energy = 0
        for i in range(6):
            delta_theta = np.abs(next_states_angle[i] - states_angle[i])
            energy = energy + np.abs(action[i]) * delta_theta




        '''delta_theta_bt = np.abs(next_states[2] - states[2])
        delta_theta_bs = np.abs(next_states[3] - states[3])
        delta_theta_bf = np.abs(next_states[4] - states[4])
        delta_theta_ft = np.abs(next_states[5] - states[5])
        delta_theta_fs = np.abs(next_states[6] - states[6])
        delta_theta_ff = np.abs(next_states[7] - states[7])

        energy_bt = np.abs(action[0]) * delta_theta_bt
        energy_bs = np.abs(action[1]) * delta_theta_bs
        energy_bf = np.abs(action[2]) * delta_theta_bf
        energy_ft = np.abs(action[3]) * delta_theta_ft
        energy_fs = np.abs(action[4]) * delta_theta_fs
        energy_ff = np.abs(action[5]) * delta_theta_ff

        energy = energy_bt + energy_bs + energy_bf + energy_ft + energy_fs + energy_ff'''

        reward -= self.energy_weights*energy
        info = {
            'x_position': x_position_after,
            'x_velocity': x_velocity,
            'energy'    : energy,
            'reward_run': forward_reward,
            'reward_ctrl': -ctrl_cost,
            'ori_reward':forward_reward-ctrl_cost
        }

        return observation, reward, done, info

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observation = np.concatenate((position, velocity)).ravel()
        return observation

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.randn(
            self.model.nv)

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)

class HalfCheetahHeavyEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                 xml_file='half_cheetah_heavy.xml',
                 forward_reward_weight=1.0,
                 ctrl_cost_weight=0.1,
                 reset_noise_scale=0.1,
                 exclude_current_positions_from_observation=True,
                 energy_weights=0.):
        utils.EzPickle.__init__(**locals())

        self._forward_reward_weight = forward_reward_weight

        self.joint_list=['bthigh','bshin','bfoot','fthigh','fshin','ffoot']

        self._ctrl_cost_weight = ctrl_cost_weight
        self.energy_weights=energy_weights
        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation)
        global path
        mujoco_env.MujocoEnv.__init__(self, os.path.join(path, xml_file), 5)
        #mujoco_env.MujocoEnv.__init__(self, xml_file, 5)

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    def step(self, action):
        states_angle = []
        for j in self.joint_list:
            states_angle.append(self.sim.data.get_joint_qpos(j))
        #states=self._get_obs()
        x_position_before = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.sim.data.qpos[0]

        x_velocity = ((x_position_after - x_position_before)
                      / self.dt)

        ctrl_cost = self.control_cost(action)

        forward_reward = self._forward_reward_weight * x_velocity

        observation = self._get_obs()
        #next_states=observation
        next_states_angle = []
        for j in self.joint_list:
            next_states_angle.append(self.sim.data.get_joint_qpos(j))

        reward = forward_reward - ctrl_cost
        done = False

        energy = 0
        for i in range(6):
            delta_theta = np.abs(next_states_angle[i] - states_angle[i])
            energy = energy + np.abs(action[i]) * delta_theta

        '''delta_theta_bt = np.abs(next_states[2] - states[2])
        delta_theta_bs = np.abs(next_states[3] - states[3])
        delta_theta_bf = np.abs(next_states[4] - states[4])
        delta_theta_ft = np.abs(next_states[5] - states[5])
        delta_theta_fs = np.abs(next_states[6] - states[6])
        delta_theta_ff = np.abs(next_states[7] - states[7])

        energy_bt = np.abs(action[0]) * delta_theta_bt
        energy_bs = np.abs(action[1]) * delta_theta_bs
        energy_bf = np.abs(action[2]) * delta_theta_bf
        energy_ft = np.abs(action[3]) * delta_theta_ft
        energy_fs = np.abs(action[4]) * delta_theta_fs
        energy_ff = np.abs(action[5]) * delta_theta_ff

        energy = energy_bt + energy_bs + energy_bf + energy_ft + energy_fs + energy_ff'''

        reward -= self.energy_weights*energy
        info = {
            'x_position': x_position_after,
            'x_velocity': x_velocity,
            'energy'    : energy,
            'reward_run': forward_reward,
            'reward_ctrl': -ctrl_cost,
            'ori_reward':forward_reward-ctrl_cost
        }

        return observation, reward, done, info

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observation = np.concatenate((position, velocity)).ravel()
        return observation

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.randn(
            self.model.nv)

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)

class FullCheetahEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                 xml_file='full_cheetah.xml',
                 forward_reward_weight=1.0,
                 ctrl_cost_weight=0.1,
                 reset_noise_scale=0.1,
                 exclude_current_positions_from_observation=True,
                 energy_weights=0.):
        utils.EzPickle.__init__(**locals())

        self._forward_reward_weight = forward_reward_weight

        self.joint_list=['bthighL','bshinL','bfootL','fthighL','fshinL','ffootL',
                         'bthighR', 'bshinR', 'bfootR', 'fthighR', 'fshinR', 'ffootR']

        self._ctrl_cost_weight = ctrl_cost_weight
        self.energy_weights=energy_weights
        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation)
        global path
        mujoco_env.MujocoEnv.__init__(self, os.path.join(path, xml_file), 5)


    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    def step(self, action):
        states_angle = []
        for j in self.joint_list:
            states_angle.append(self.sim.data.get_joint_qpos(j))
        #states=self._get_obs()
        x_position_before = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.sim.data.qpos[0]

        x_velocity = ((x_position_after - x_position_before)
                      / self.dt)

        ctrl_cost = self.control_cost(action)

        forward_reward = self._forward_reward_weight * x_velocity

        observation = self._get_obs()
        #next_states=observation
        next_states_angle = []
        for j in self.joint_list:
            next_states_angle.append(self.sim.data.get_joint_qpos(j))

        reward = forward_reward - ctrl_cost
        done = False

        energy = 0
        for i in range(len(self.joint_list)):
            delta_theta = np.abs(next_states_angle[i] - states_angle[i])
            energy = energy + np.abs(action[i]) * delta_theta

        '''delta_theta_bt = np.abs(next_states[2] - states[2])
        delta_theta_bs = np.abs(next_states[3] - states[3])
        delta_theta_bf = np.abs(next_states[4] - states[4])
        delta_theta_ft = np.abs(next_states[5] - states[5])
        delta_theta_fs = np.abs(next_states[6] - states[6])
        delta_theta_ff = np.abs(next_states[7] - states[7])

        energy_bt = np.abs(action[0]) * delta_theta_bt
        energy_bs = np.abs(action[1]) * delta_theta_bs
        energy_bf = np.abs(action[2]) * delta_theta_bf
        energy_ft = np.abs(action[3]) * delta_theta_ft
        energy_fs = np.abs(action[4]) * delta_theta_fs
        energy_ff = np.abs(action[5]) * delta_theta_ff

        energy = energy_bt + energy_bs + energy_bf + energy_ft + energy_fs + energy_ff'''

        reward -= self.energy_weights*energy
        info = {
            'x_position': x_position_after,
            'x_velocity': x_velocity,
            'energy'    : energy,
            'reward_run': forward_reward,
            'reward_ctrl': -ctrl_cost,
            'ori_reward':forward_reward-ctrl_cost
        }

        return observation, reward, done, info

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observation = np.concatenate((position, velocity)).ravel()
        return observation

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.randn(
            self.model.nv)

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)

class HalfCheetahEnv_5dof(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                 xml_file='half_cheetah_5dof.xml',
                 forward_reward_weight=1.0,
                 ctrl_cost_weight=0.1,
                 reset_noise_scale=0.1,
                 exclude_current_positions_from_observation=True,
                 energy_weights=0.):
        utils.EzPickle.__init__(**locals())

        self._forward_reward_weight = forward_reward_weight

        self.joint_list=['bthigh','bshin','bfoot','fthigh','fshin']

        self._ctrl_cost_weight = ctrl_cost_weight
        self.energy_weights=energy_weights
        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation)

        global path
        mujoco_env.MujocoEnv.__init__(self, os.path.join(path, xml_file), 5)

        #mujoco_env.MujocoEnv.__init__(self, xml_file, 5)

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    def step(self, action):
        states_angle = []
        for j in self.joint_list:
            states_angle.append(self.sim.data.get_joint_qpos(j))
        #states=self._get_obs()
        x_position_before = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.sim.data.qpos[0]

        x_velocity = ((x_position_after - x_position_before)
                      / self.dt)

        ctrl_cost = self.control_cost(action)

        forward_reward = self._forward_reward_weight * x_velocity

        observation = self._get_obs()
        #next_states=observation
        next_states_angle = []
        for j in self.joint_list:
            next_states_angle.append(self.sim.data.get_joint_qpos(j))

        reward = forward_reward - ctrl_cost
        done = False

        energy = 0
        for i in range(len(self.joint_list)):
            delta_theta = np.abs(next_states_angle[i] - states_angle[i])
            energy = energy + np.abs(action[i]) * delta_theta

        '''delta_theta_bt = np.abs(next_states[2] - states[2])
        delta_theta_bs = np.abs(next_states[3] - states[3])
        delta_theta_bf = np.abs(next_states[4] - states[4])
        delta_theta_ft = np.abs(next_states[5] - states[5])
        delta_theta_fs = np.abs(next_states[6] - states[6])
        delta_theta_ff = np.abs(next_states[7] - states[7])

        energy_bt = np.abs(action[0]) * delta_theta_bt
        energy_bs = np.abs(action[1]) * delta_theta_bs
        energy_bf = np.abs(action[2]) * delta_theta_bf
        energy_ft = np.abs(action[3]) * delta_theta_ft
        energy_fs = np.abs(action[4]) * delta_theta_fs
        energy_ff = np.abs(action[5]) * delta_theta_ff

        energy = energy_bt + energy_bs + energy_bf + energy_ft + energy_fs + energy_ff'''

        reward -= self.energy_weights*energy
        info = {
            'x_position': x_position_after,
            'x_velocity': x_velocity,
            'energy'    : energy,
            'reward_run': forward_reward,
            'reward_ctrl': -ctrl_cost,
            'ori_reward':forward_reward-ctrl_cost
        }

        return observation, reward, done, info

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observation = np.concatenate((position, velocity)).ravel()
        return observation

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.randn(
            self.model.nv)

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)

class HalfCheetahEnv_4dof(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                 xml_file='half_cheetah_4dof.xml',
                 forward_reward_weight=1.0,
                 ctrl_cost_weight=0.1,
                 reset_noise_scale=0.1,
                 exclude_current_positions_from_observation=True,
                 energy_weights=0.):
        utils.EzPickle.__init__(**locals())

        self._forward_reward_weight = forward_reward_weight

        self.joint_list=['bthigh','bshin','fthigh','fshin']

        self._ctrl_cost_weight = ctrl_cost_weight
        self.energy_weights=energy_weights
        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation)

        global path
        mujoco_env.MujocoEnv.__init__(self, os.path.join(path, xml_file), 5)

        #mujoco_env.MujocoEnv.__init__(self, xml_file, 5)

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    def step(self, action):
        states_angle = []
        for j in self.joint_list:
            states_angle.append(self.sim.data.get_joint_qpos(j))
        #states=self._get_obs()
        x_position_before = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.sim.data.qpos[0]

        x_velocity = ((x_position_after - x_position_before)
                      / self.dt)

        ctrl_cost = self.control_cost(action)

        forward_reward = self._forward_reward_weight * x_velocity

        observation = self._get_obs()
        #next_states=observation
        next_states_angle = []
        for j in self.joint_list:
            next_states_angle.append(self.sim.data.get_joint_qpos(j))

        reward = forward_reward - ctrl_cost
        done = False

        energy = 0
        for i in range(len(self.joint_list)):
            delta_theta = np.abs(next_states_angle[i] - states_angle[i])
            energy = energy + np.abs(action[i]) * delta_theta

        '''delta_theta_bt = np.abs(next_states[2] - states[2])
        delta_theta_bs = np.abs(next_states[3] - states[3])
        delta_theta_bf = np.abs(next_states[4] - states[4])
        delta_theta_ft = np.abs(next_states[5] - states[5])
        delta_theta_fs = np.abs(next_states[6] - states[6])
        delta_theta_ff = np.abs(next_states[7] - states[7])

        energy_bt = np.abs(action[0]) * delta_theta_bt
        energy_bs = np.abs(action[1]) * delta_theta_bs
        energy_bf = np.abs(action[2]) * delta_theta_bf
        energy_ft = np.abs(action[3]) * delta_theta_ft
        energy_fs = np.abs(action[4]) * delta_theta_fs
        energy_ff = np.abs(action[5]) * delta_theta_ff

        energy = energy_bt + energy_bs + energy_bf + energy_ft + energy_fs + energy_ff'''

        reward -= self.energy_weights*energy
        info = {
            'x_position': x_position_after,
            'x_velocity': x_velocity,
            'energy'    : energy,
            'reward_run': forward_reward,
            'reward_ctrl': -ctrl_cost,
            'ori_reward':forward_reward-ctrl_cost
        }

        return observation, reward, done, info

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observation = np.concatenate((position, velocity)).ravel()
        return observation

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.randn(
            self.model.nv)

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)

class HalfCheetahEnv_3doff(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                 xml_file='half_cheetah_3dof_front.xml',
                 forward_reward_weight=1.0,
                 ctrl_cost_weight=0.1,
                 reset_noise_scale=0.1,
                 exclude_current_positions_from_observation=True,
                 energy_weights=0.):
        utils.EzPickle.__init__(**locals())

        self._forward_reward_weight = forward_reward_weight

        self.joint_list=['bthigh','fthigh','fshin']

        self._ctrl_cost_weight = ctrl_cost_weight
        self.energy_weights=energy_weights
        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation)

        global path
        mujoco_env.MujocoEnv.__init__(self, os.path.join(path, xml_file), 5)

        #mujoco_env.MujocoEnv.__init__(self, xml_file, 5)

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    def step(self, action):
        states_angle = []
        for j in self.joint_list:
            states_angle.append(self.sim.data.get_joint_qpos(j))
        #states=self._get_obs()
        x_position_before = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.sim.data.qpos[0]

        x_velocity = ((x_position_after - x_position_before)
                      / self.dt)

        ctrl_cost = self.control_cost(action)

        forward_reward = self._forward_reward_weight * x_velocity

        observation = self._get_obs()
        #next_states=observation
        next_states_angle = []
        for j in self.joint_list:
            next_states_angle.append(self.sim.data.get_joint_qpos(j))

        reward = forward_reward - ctrl_cost
        done = False

        energy = 0
        for i in range(len(self.joint_list)):
            delta_theta = np.abs(next_states_angle[i] - states_angle[i])
            energy = energy + np.abs(action[i]) * delta_theta

        '''delta_theta_bt = np.abs(next_states[2] - states[2])
        delta_theta_bs = np.abs(next_states[3] - states[3])
        delta_theta_bf = np.abs(next_states[4] - states[4])
        delta_theta_ft = np.abs(next_states[5] - states[5])
        delta_theta_fs = np.abs(next_states[6] - states[6])
        delta_theta_ff = np.abs(next_states[7] - states[7])

        energy_bt = np.abs(action[0]) * delta_theta_bt
        energy_bs = np.abs(action[1]) * delta_theta_bs
        energy_bf = np.abs(action[2]) * delta_theta_bf
        energy_ft = np.abs(action[3]) * delta_theta_ft
        energy_fs = np.abs(action[4]) * delta_theta_fs
        energy_ff = np.abs(action[5]) * delta_theta_ff

        energy = energy_bt + energy_bs + energy_bf + energy_ft + energy_fs + energy_ff'''

        reward -= self.energy_weights*energy
        info = {
            'x_position': x_position_after,
            'x_velocity': x_velocity,
            'energy'    : energy,
            'reward_run': forward_reward,
            'reward_ctrl': -ctrl_cost,
            'ori_reward':forward_reward-ctrl_cost
        }

        return observation, reward, done, info

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observation = np.concatenate((position, velocity)).ravel()
        return observation

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.randn(
            self.model.nv)

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)

class HalfCheetahEnv_3dofb(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                 xml_file='half_cheetah_3dof_back.xml',
                 forward_reward_weight=1.0,
                 ctrl_cost_weight=0.1,
                 reset_noise_scale=0.1,
                 exclude_current_positions_from_observation=True,
                 energy_weights=0.):
        utils.EzPickle.__init__(**locals())

        self._forward_reward_weight = forward_reward_weight

        self.joint_list=['bthigh','bshin','fthigh']

        self._ctrl_cost_weight = ctrl_cost_weight
        self.energy_weights=energy_weights
        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation)

        global path
        mujoco_env.MujocoEnv.__init__(self, os.path.join(path, xml_file), 5)

        #mujoco_env.MujocoEnv.__init__(self, xml_file, 5)

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    def step(self, action):
        states_angle = []
        for j in self.joint_list:
            states_angle.append(self.sim.data.get_joint_qpos(j))
        #states=self._get_obs()
        x_position_before = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.sim.data.qpos[0]

        x_velocity = ((x_position_after - x_position_before)
                      / self.dt)

        ctrl_cost = self.control_cost(action)

        forward_reward = self._forward_reward_weight * x_velocity

        observation = self._get_obs()
        #next_states=observation
        next_states_angle = []
        for j in self.joint_list:
            next_states_angle.append(self.sim.data.get_joint_qpos(j))

        reward = forward_reward - ctrl_cost
        done = False

        energy = 0
        for i in range(len(self.joint_list)):
            delta_theta = np.abs(next_states_angle[i] - states_angle[i])
            energy = energy + np.abs(action[i]) * delta_theta

        '''delta_theta_bt = np.abs(next_states[2] - states[2])
        delta_theta_bs = np.abs(next_states[3] - states[3])
        delta_theta_bf = np.abs(next_states[4] - states[4])
        delta_theta_ft = np.abs(next_states[5] - states[5])
        delta_theta_fs = np.abs(next_states[6] - states[6])
        delta_theta_ff = np.abs(next_states[7] - states[7])

        energy_bt = np.abs(action[0]) * delta_theta_bt
        energy_bs = np.abs(action[1]) * delta_theta_bs
        energy_bf = np.abs(action[2]) * delta_theta_bf
        energy_ft = np.abs(action[3]) * delta_theta_ft
        energy_fs = np.abs(action[4]) * delta_theta_fs
        energy_ff = np.abs(action[5]) * delta_theta_ff

        energy = energy_bt + energy_bs + energy_bf + energy_ft + energy_fs + energy_ff'''

        reward -= self.energy_weights*energy
        info = {
            'x_position': x_position_after,
            'x_velocity': x_velocity,
            'energy'    : energy,
            'reward_run': forward_reward,
            'reward_ctrl': -ctrl_cost,
            'ori_reward':forward_reward-ctrl_cost
        }

        return observation, reward, done, info

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observation = np.concatenate((position, velocity)).ravel()
        return observation

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.randn(
            self.model.nv)

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)

class HalfCheetahEnv_2dof(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                 xml_file='half_cheetah_2dof.xml',
                 forward_reward_weight=1.0,
                 ctrl_cost_weight=0.1,
                 reset_noise_scale=0.1,
                 exclude_current_positions_from_observation=True,
                 energy_weights=0.):
        utils.EzPickle.__init__(**locals())

        self._forward_reward_weight = forward_reward_weight

        self.joint_list=['bthigh','fthigh']

        self._ctrl_cost_weight = ctrl_cost_weight
        self.energy_weights=energy_weights
        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation)

        global path
        mujoco_env.MujocoEnv.__init__(self, os.path.join(path, xml_file), 5)

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    def step(self, action):
        states_angle = []
        for j in self.joint_list:
            states_angle.append(self.sim.data.get_joint_qpos(j))
        #states=self._get_obs()
        x_position_before = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.sim.data.qpos[0]

        x_velocity = ((x_position_after - x_position_before)
                      / self.dt)

        ctrl_cost = self.control_cost(action)

        forward_reward = self._forward_reward_weight * x_velocity

        observation = self._get_obs()
        #next_states=observation
        next_states_angle = []
        for j in self.joint_list:
            next_states_angle.append(self.sim.data.get_joint_qpos(j))

        reward = forward_reward - ctrl_cost
        done = False

        energy = 0
        for i in range(len(self.joint_list)):
            delta_theta = np.abs(next_states_angle[i] - states_angle[i])
            energy = energy + np.abs(action[i]) * delta_theta

        '''delta_theta_bt = np.abs(next_states[2] - states[2])
        delta_theta_bs = np.abs(next_states[3] - states[3])
        delta_theta_bf = np.abs(next_states[4] - states[4])
        delta_theta_ft = np.abs(next_states[5] - states[5])
        delta_theta_fs = np.abs(next_states[6] - states[6])
        delta_theta_ff = np.abs(next_states[7] - states[7])

        energy_bt = np.abs(action[0]) * delta_theta_bt
        energy_bs = np.abs(action[1]) * delta_theta_bs
        energy_bf = np.abs(action[2]) * delta_theta_bf
        energy_ft = np.abs(action[3]) * delta_theta_ft
        energy_fs = np.abs(action[4]) * delta_theta_fs
        energy_ff = np.abs(action[5]) * delta_theta_ff

        energy = energy_bt + energy_bs + energy_bf + energy_ft + energy_fs + energy_ff'''

        reward -= self.energy_weights*energy
        info = {
            'x_position': x_position_after,
            'x_velocity': x_velocity,
            'energy'    : energy,
            'reward_run': forward_reward,
            'reward_ctrl': -ctrl_cost,
            'ori_reward':forward_reward-ctrl_cost
        }

        return observation, reward, done, info

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observation = np.concatenate((position, velocity)).ravel()
        return observation

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.randn(
            self.model.nv)

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)