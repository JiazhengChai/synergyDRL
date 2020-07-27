import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from gym import utils
from . import path
import os

DEFAULT_CAMERA_CONFIG = {
    'trackbodyid': 2,
    'distance': 4.0,
    'lookat': np.array((0.0, 0.0, 1.15)),
    'elevation': -20.0,
}


class Walker2dEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                 xml_file='walker2d.xml',
                 forward_reward_weight=1.0,
                 ctrl_cost_weight=1e-3,
                 healthy_reward=1.0,
                 terminate_when_unhealthy=True,
                 healthy_z_range=(0.8, 2.0),
                 healthy_angle_range=(-1.0, 1.0),
                 reset_noise_scale=5e-3,
                 exclude_current_positions_from_observation=True,
                 energy_weights=0):
        utils.EzPickle.__init__(**locals())

        self.joint_list=['thigh_joint','leg_joint','foot_joint','thigh_left_joint','leg_left_joint','foot_left_joint']
        self.energy_weights=energy_weights
        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight

        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy

        self._healthy_z_range = healthy_z_range
        self._healthy_angle_range = healthy_angle_range

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation)

        global path
        mujoco_env.MujocoEnv.__init__(self, os.path.join(path, xml_file), 4)
        #mujoco_env.MujocoEnv.__init__(self, xml_file, 4)
    @property
    def healthy_reward(self):
        return float(
            self.is_healthy
            or self._terminate_when_unhealthy
        ) * self._healthy_reward

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    @property
    def is_healthy(self):
        z, angle = self.sim.data.qpos[1:3]

        min_z, max_z = self._healthy_z_range
        min_angle, max_angle = self._healthy_angle_range

        healthy_z = min_z < z < max_z
        healthy_angle = min_angle < angle < max_angle
        is_healthy = healthy_z and healthy_angle

        return is_healthy

    @property
    def done(self):
        done = (not self.is_healthy
                if self._terminate_when_unhealthy
                else False)
        return done

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = np.clip(
            self.sim.data.qvel.flat.copy(), -10, 10)

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observation = np.concatenate((position, velocity)).ravel()
        return observation

    def step(self, action):
        states_angle = []
        for j in self.joint_list:
            states_angle.append(self.sim.data.get_joint_qpos(j))
        x_position_before = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.sim.data.qpos[0]
        x_velocity = ((x_position_after - x_position_before)
                      / self.dt)
        next_states_angle = []
        for j in self.joint_list:
            next_states_angle.append(self.sim.data.get_joint_qpos(j))
        ctrl_cost = self.control_cost(action)

        forward_reward = self._forward_reward_weight * x_velocity
        healthy_reward = self.healthy_reward

        energy = 0
        for i in range(len(self.joint_list)):
            delta_theta = np.abs(next_states_angle[i] - states_angle[i])
            energy = energy + np.abs(action[i]) * delta_theta


        rewards = forward_reward + healthy_reward
        costs = ctrl_cost

        #energy = 0
        #for i in range(17):
        #    delta_theta = np.abs(next_states_speed[i + 6] - states_speed[i + 6])
        #    energy = energy + np.abs(action[i]) * delta_theta

        observation = self._get_obs()
        ori_reward = rewards - costs
        reward= ori_reward-self.energy_weights*energy
        done = self.done
        info = {
            'x_position': x_position_after,
            'x_velocity': x_velocity,
            'ori_reward': ori_reward,
            'energy': energy,
        }

        return observation, reward, done, info

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv)

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)
