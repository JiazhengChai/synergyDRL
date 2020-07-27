import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import os

TARGET_ENERGY=3
class Bipedal2Env(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self,type='assets/bipedal.xml',
                 path=None,
                 target_energy=TARGET_ENERGY,
                 alpha=1,#speed
                 beta=0,#contact
                 gamma=0,#middle
                 delta=0,#alive
                 zeta=1 #complementary
                 ):

        self.target_energy=target_energy
        self.before_init=True

        self.alpha=alpha
        self.beta=beta
        self.gamma=gamma
        self.delta=delta
        self.zeta=zeta
        if path:
            mujoco_env.MujocoEnv.__init__(self, os.path.join(path,type), 4)
        else:
            try:
                mujoco_env.MujocoEnv.__init__(self, 'bipedal.xml', 4)
            except:
                print('Error.')
                print('Please specify the folder in which the '+type+ ' can be found.')
                exit()

        utils.EzPickle.__init__(self)
        self.before_init =False

    def step(self, action):
        #action=[0,0]
        pos_root_before = self.sim.data.qpos[0]

        pos_legr_before = self.sim.data.body_xpos[2][0]
        pos_footr_before = self.sim.data.body_xpos[3][0]
        pos_legl_before = self.sim.data.body_xpos[4][0]
        pos_footl_before = self.sim.data.body_xpos[5][0]

        ang_legr_before = self.sim.data.qpos[3]
        ang_footr_before = self.sim.data.qpos[4]
        ang_legl_before = self.sim.data.qpos[5]
        ang_footl_before = self.sim.data.qpos[6]

        self.do_simulation(action, self.frame_skip)

        pos_root_after = self.sim.data.qpos[0]
        height = self.sim.data.qpos[1]

        pos_legr_after = self.sim.data.body_xpos[2][0]
        pos_footr_after = self.sim.data.body_xpos[3][0]
        pos_legl_after = self.sim.data.body_xpos[4][0]
        pos_footl_after = self.sim.data.body_xpos[5][0]

        ang_legr_after = self.sim.data.qpos[3]
        ang_footr_after = self.sim.data.qpos[4]
        ang_legl_after = self.sim.data.qpos[5]
        ang_footl_after = self.sim.data.qpos[6]

        #reward_big_angle=np.abs(ang_legr_after)+np.abs(ang_legl_after)
        # 前に進んだら報酬を与える(後ろに進んだらマイナスの報酬)

        raw_contact_forces = self.sim.data.cfrc_ext
        contact_forces_right_leg=(np.linalg.norm(raw_contact_forces[3]))
        contact_forces_left_leg=(np.linalg.norm(raw_contact_forces[5]))

        if (contact_forces_right_leg!=0 and contact_forces_left_leg==0) or \
                (contact_forces_right_leg==0 and contact_forces_left_leg!=0):
            reward_contact=1
        else:
            reward_contact=0

        reward_speed = ((pos_root_after - pos_root_before) / self.dt)

        if pos_legr_after>pos_legl_after:

            if pos_root_after>=pos_legl_after and pos_root_after<=pos_legr_after:

                reward_middle=2
            else:

                reward_middle = -1
        elif pos_legr_after <= pos_legl_after:

            if pos_root_after>=pos_legr_after and pos_root_after<=pos_legl_after:
                reward_middle=2

            else:

                reward_middle = -1

        #reward_middle=pos_root_after
        # 足を交互に降り出したら報酬を与える
        # 支持脚と遊脚の情報がわかればうまく行きそう
        '''diff_xr = pos_footr_after[0] - pos_footr_before[0]
        diff_xl = pos_footl_after[0] - pos_footl_before[0]

        if diff_xr == 0 and diff_xl != 0:
            reward += diff_xl
        elif diff_xr != 0 and diff_xl == 0:
            reward += diff_xr'''

        alive_bonus = 1.0



        reward = self.alpha*reward_speed + \
                 self.beta*reward_contact+\
                 self.gamma*reward_middle + \
                 self.delta*alive_bonus -\
                 self.zeta*np.abs(action[0])*np.abs(action[1])
        #+self.zeta*reward_big_angle



        # reward -= 1e-3 * np.square(action).sum()
        # 床を押して進んだら報酬（足をプッシュする方向）
        # 支持脚の角度で表現したい
        # reward += 

        # 床から離れたらマイナスの報酬（or 試行終了？）
        # 必ず片足はついているはず（両足が浮くことはない）
        # diff_zr = pos_footr_after[3] - pos_footr_before[3]
        # diff_zl = pos_footl_after[3] - pos_footl_before[3]
        # if pos_footr_after[2] > 0.4 and pos_footl_after[2] > 0.4:
        #     reward -= 10

        # 倒れたらその思考を終了する (倒れている状態で height 0.3)
        #done = not (height > 0.7 and height < 1.3)
        #done = not (height > 1 and height < 1.3)
        done = not (height > 0.7 and height < 1.5)
        ob = self._get_obs()

        info = {
            'reward_speed': reward_speed,
            #'reward_big_angle':reward_big_angle
            'reward_middle': reward_middle,
        }
        return ob, reward, done, info

    # 足の情報を取れるようにしたい
    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        return np.concatenate([qpos[1:], np.clip(qvel, -10, 10)]).ravel()


    def reset_model(self):
        reset_p=self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        reset_v=self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)

        #reset_p[1]=1.11
        reset_p[3]=-1.+ self.np_random.uniform(low=-.005, high=.005)
        reset_p[5] = -0+ self.np_random.uniform(low=-.005, high=.005)

        self.set_state(
            reset_p,
            reset_v
        )
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20
