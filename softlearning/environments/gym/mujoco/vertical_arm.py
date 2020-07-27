import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import os
from . import path

DEFAULT_CAMERA_CONFIG = {
    'trackbodyid': 0,
    'distance': 1.0,
    'lookat': np.array((0.0, 0.0, 0)),
    'elevation': 0,
}
def sin(t,omega=1.5,phi=0.):#1
    return np.sin(omega*t+phi)
def cos(t,omega=1.5,phi=0.):#1
    return np.cos(omega*t+phi)

class VA(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,xml_file='vertical_arm.xml',
                 distance_reward_weight=5.0,
                 ctrl_cost_weight=0.05
                 ):
        #utils.EzPickle.__init__(self)
        utils.EzPickle.__init__(**locals())
        self.joint_list = ['shoulder', 'elbow']
        self.real_time=0.01
        self.frame_skip=2
        self.t=0
        self.target_pos = np.asarray([0, 0, 0])
        self.distance_reward_weight= distance_reward_weight
        self.ctrl_cost_weight= ctrl_cost_weight
        global path
        mujoco_env.MujocoEnv.__init__(self, os.path.join(path, xml_file), self.frame_skip)
        '''if path is not None:
            mujoco_env.MujocoEnv.__init__(self, os.path.join(path, xml_file), self.frame_skip)
        else:
            mujoco_env.MujocoEnv.__init__(self, xml_file, self.frame_skip)'''

    def step(self, a):
        #print(a)
        states_angle = []
        for j in self.joint_list:
            states_angle.append(self.sim.data.get_joint_qpos(j))

        vec = self.get_body_com("fingertip")- self.target_pos
        reward_dist = - np.linalg.norm(vec)
        reward_ctrl =- np.square(a).sum()
        reward = self.distance_reward_weight*reward_dist +self.ctrl_cost_weight*reward_ctrl
        self.do_simulation(a, self.frame_skip)

        self.t += self.frame_skip * self.real_time
        self.sim.data.site_xpos[0] = self.sim.data.site_xpos[0] + [-0.15*sin(self.t,phi=0)*np.sin(-25* np.pi / 180.), 0,
                                                                   0.15*sin(self.t,phi=0)*np.cos(-25* np.pi / 180.)+0.01]#,phi=1


        self.target_pos = self.sim.data.site_xpos[0]


        ob = self._get_obs()

        next_states_angle = []
        for j in self.joint_list:
            next_states_angle.append(self.sim.data.get_joint_qpos(j))

        energy = 0
        for i in range(len(self.joint_list)):
            delta_theta = np.abs(next_states_angle[i] - states_angle[i])
            energy = energy + np.abs(a[i]) * delta_theta


        done = False

        info = {
            'energy': energy,
            'reward_dist': self.distance_reward_weight*reward_dist,
            'reward_ctrl': self.ctrl_cost_weight*reward_ctrl,
            'ori_reward': reward
        }

        return ob, reward, done, info#,reward_ctrl=reward_ctrl

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)

    def reset_model(self):
        self.t=0
        #self.data.site_xpos[0] = [1, 1, 1] -.15 0.01 -.1
        qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos

        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)

        self.set_state(qpos, qvel)

        self.target_pos = self.data.site_xpos[0]

        return self._get_obs()

    def _get_obs(self):
        theta = self.sim.data.qpos.flat[:2]

        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat[:2],
            [self.get_body_com("fingertip")[0] - self.target_pos[0]],
            [self.get_body_com("fingertip")[2] - self.target_pos[2]]
        ]).ravel()

class VA4dof(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,xml_file='vertical_arm4dof.xml',
                 distance_reward_weight=5.0,
                 ctrl_cost_weight=0.05
                 ):
        #utils.EzPickle.__init__(self)
        utils.EzPickle.__init__(**locals())
        self.joint_list = ['shoulder','shoulder2', 'elbow', 'elbow2']
        self.real_time=0.01
        self.frame_skip=2
        self.t=0
        self.target_pos = np.asarray([0, 0, 0])
        self.distance_reward_weight= distance_reward_weight
        self.ctrl_cost_weight= ctrl_cost_weight
        global path
        mujoco_env.MujocoEnv.__init__(self, os.path.join(path, xml_file), self.frame_skip)
        '''if path is not None:
            mujoco_env.MujocoEnv.__init__(self, os.path.join(path, xml_file), self.frame_skip)
        else:
            mujoco_env.MujocoEnv.__init__(self, xml_file, self.frame_skip)'''

    def step(self, a):
        #print(a)
        states_angle = []
        for j in self.joint_list:
            states_angle.append(self.sim.data.get_joint_qpos(j))

        vec = self.get_body_com("fingertip")- self.target_pos
        reward_dist = - np.linalg.norm(vec)
        reward_ctrl =- np.square(a).sum()
        reward = self.distance_reward_weight*reward_dist +self.ctrl_cost_weight*reward_ctrl
        self.do_simulation(a, self.frame_skip)

        self.t += self.frame_skip * self.real_time
        self.sim.data.site_xpos[0] = self.sim.data.site_xpos[0] + [-0.15*sin(self.t,phi=0)*np.sin(-25* np.pi / 180.), 0,
                                                                   0.15*sin(self.t,phi=0)*np.cos(-25* np.pi / 180.)+0.01]#,phi=1


        self.target_pos = self.sim.data.site_xpos[0]


        ob = self._get_obs()

        next_states_angle = []
        for j in self.joint_list:
            next_states_angle.append(self.sim.data.get_joint_qpos(j))

        energy = 0
        for i in range(len(self.joint_list)):
            delta_theta = np.abs(next_states_angle[i] - states_angle[i])
            energy = energy + np.abs(a[i]) * delta_theta


        done = False

        info = {
            'energy': energy,
            'reward_dist': self.distance_reward_weight*reward_dist,
            'reward_ctrl': self.ctrl_cost_weight*reward_ctrl,
            'ori_reward': reward
        }

        return ob, reward, done, info#,reward_ctrl=reward_ctrl

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)

    def reset_model(self):
        self.t=0
        #self.data.site_xpos[0] = [1, 1, 1] -.15 0.01 -.1
        qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos

        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)

        self.set_state(qpos, qvel)

        self.target_pos = self.data.site_xpos[0]

        return self._get_obs()

    def _get_obs(self):
        theta = self.sim.data.qpos.flat[:4]

        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.sim.data.qpos.flat[4:],
            self.sim.data.qvel.flat[:4],
            [self.get_body_com("fingertip")[0] - self.target_pos[0]],
            [self.get_body_com("fingertip")[2] - self.target_pos[2]]
        ]).ravel()

class VA6dof(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,xml_file='vertical_arm6dof.xml',
                 distance_reward_weight=5.0,
                 ctrl_cost_weight=0.05
                 ):
        #utils.EzPickle.__init__(self)
        utils.EzPickle.__init__(**locals())
        self.joint_list = ['shoulder','shoulder2', 'elbow', 'elbow2','elbow3', 'elbow4']
        self.real_time=0.01
        self.frame_skip=2
        self.t=0
        self.target_pos = np.asarray([0, 0, 0])
        self.distance_reward_weight= distance_reward_weight
        self.ctrl_cost_weight= ctrl_cost_weight
        global path
        mujoco_env.MujocoEnv.__init__(self, os.path.join(path, xml_file), self.frame_skip)
        '''if path is not None:
            mujoco_env.MujocoEnv.__init__(self, os.path.join(path, xml_file), self.frame_skip)
        else:
            mujoco_env.MujocoEnv.__init__(self, xml_file, self.frame_skip)'''

    def step(self, a):
        #print(a)
        states_angle = []
        for j in self.joint_list:
            states_angle.append(self.sim.data.get_joint_qpos(j))

        vec = self.get_body_com("fingertip")- self.target_pos
        reward_dist = - np.linalg.norm(vec)
        reward_ctrl =- np.square(a).sum()
        reward = self.distance_reward_weight*reward_dist +self.ctrl_cost_weight*reward_ctrl
        self.do_simulation(a, self.frame_skip)

        self.t += self.frame_skip * self.real_time
        self.sim.data.site_xpos[0] = self.sim.data.site_xpos[0] + [-0.15*sin(self.t,phi=0)*np.sin(-25* np.pi / 180.), 0,
                                                                   0.15*sin(self.t,phi=0)*np.cos(-25* np.pi / 180.)+0.01]#,phi=1


        self.target_pos = self.sim.data.site_xpos[0]


        ob = self._get_obs()

        next_states_angle = []
        for j in self.joint_list:
            next_states_angle.append(self.sim.data.get_joint_qpos(j))

        energy = 0
        for i in range(len(self.joint_list)):
            delta_theta = np.abs(next_states_angle[i] - states_angle[i])
            energy = energy + np.abs(a[i]) * delta_theta


        done = False

        info = {
            'energy': energy,
            'reward_dist': self.distance_reward_weight*reward_dist,
            'reward_ctrl': self.ctrl_cost_weight*reward_ctrl,
            'ori_reward': reward
        }

        return ob, reward, done, info#,reward_ctrl=reward_ctrl

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)

    def reset_model(self):
        self.t=0
        #self.data.site_xpos[0] = [1, 1, 1] -.15 0.01 -.1
        qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos

        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)

        self.set_state(qpos, qvel)

        self.target_pos = self.data.site_xpos[0]

        return self._get_obs()

    def _get_obs(self):
        theta = self.sim.data.qpos.flat[:6]

        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.sim.data.qpos.flat[6:],
            self.sim.data.qvel.flat[:6],
            [self.get_body_com("fingertip")[0] - self.target_pos[0]],
            [self.get_body_com("fingertip")[2] - self.target_pos[2]]
        ]).ravel()

class VA8dof(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,xml_file='vertical_arm8dof.xml',
                 distance_reward_weight=5.0,
                 ctrl_cost_weight=0.05
                 ):
        #utils.EzPickle.__init__(self)
        utils.EzPickle.__init__(**locals())
        self.joint_list = ['shoulder','shoulder2', 'shoulder3','shoulder4','elbow', 'elbow2','elbow3', 'elbow4']
        self.real_time=0.01
        self.frame_skip=2
        self.t=0
        self.target_pos = np.asarray([0, 0, 0])
        self.distance_reward_weight= distance_reward_weight
        self.ctrl_cost_weight= ctrl_cost_weight
        global path
        mujoco_env.MujocoEnv.__init__(self, os.path.join(path, xml_file), self.frame_skip)
        '''if path is not None:
            mujoco_env.MujocoEnv.__init__(self, os.path.join(path, xml_file), self.frame_skip)
        else:
            mujoco_env.MujocoEnv.__init__(self, xml_file, self.frame_skip)'''

    def step(self, a):
        #print(a)
        states_angle = []
        for j in self.joint_list:
            states_angle.append(self.sim.data.get_joint_qpos(j))

        vec = self.get_body_com("fingertip")- self.target_pos
        reward_dist = - np.linalg.norm(vec)
        reward_ctrl =- np.square(a).sum()
        reward = self.distance_reward_weight*reward_dist +self.ctrl_cost_weight*reward_ctrl
        self.do_simulation(a, self.frame_skip)

        self.t += self.frame_skip * self.real_time
        self.sim.data.site_xpos[0] = self.sim.data.site_xpos[0] + [-0.15*sin(self.t,phi=0)*np.sin(-25* np.pi / 180.), 0,
                                                                   0.15*sin(self.t,phi=0)*np.cos(-25* np.pi / 180.)+0.01]#,phi=1


        self.target_pos = self.sim.data.site_xpos[0]


        ob = self._get_obs()

        next_states_angle = []
        for j in self.joint_list:
            next_states_angle.append(self.sim.data.get_joint_qpos(j))

        energy = 0
        for i in range(len(self.joint_list)):
            delta_theta = np.abs(next_states_angle[i] - states_angle[i])
            energy = energy + np.abs(a[i]) * delta_theta


        done = False

        info = {
            'energy': energy,
            'reward_dist': self.distance_reward_weight*reward_dist,
            'reward_ctrl': self.ctrl_cost_weight*reward_ctrl,
            'ori_reward': reward
        }

        return ob, reward, done, info#,reward_ctrl=reward_ctrl

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)

    def reset_model(self):
        self.t=0
        #self.data.site_xpos[0] = [1, 1, 1] -.15 0.01 -.1
        qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos

        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)

        self.set_state(qpos, qvel)

        self.target_pos = self.data.site_xpos[0]

        return self._get_obs()

    def _get_obs(self):
        theta = self.sim.data.qpos.flat[:8]

        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.sim.data.qpos.flat[8:],
            self.sim.data.qvel.flat[:8],
            [self.get_body_com("fingertip")[0] - self.target_pos[0]],
            [self.get_body_com("fingertip")[2] - self.target_pos[2]]
        ]).ravel()
