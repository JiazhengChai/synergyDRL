import os
import numpy as np
from sklearn.preprocessing import StandardScaler
class exp_variant():
    def __init__(self, name,
                 action_path='./experiments_results/collected_actions/trajectory_npy/actions_npy',
                 reward_path='./experiments_results/collected_actions/trajectory_npy/reward_energy_dict',
                 state_path='./experiments_results/collected_actions/trajectory_npy/states_npy',
                 short_name=False):
        if not short_name:
            if '_C' in name:
                self.name = name
            else:
                if '_r' in name:
                    name_list=name.split('_')
                    self.name=name.replace(name_list[-1],'')+'C3000'+'_'+name_list[-1]
                else:
                    self.name = name + '_C3000'
        else:
            if 'TD3' not in name:
                if '_r' in name:
                    name_list = name.split('_')
                    self.name = name_list[0]  +'_'+ name_list[-1]
                else:
                    name_list = name.split('_')
                    self.name = name_list[0]+ '_r1'
            else:
                if '_r' in name:
                    name_list = name.split('_')
                    self.name = name_list[0] + '_'+name_list[2]+ '_'  + name_list[-1]
                else:
                    name_list = name.split('_')
                    self.name = name_list[0]+ '_'+name_list[2] + '_r1'

        if 'npy' not in name:
            self.npy = name + '.npy'
        else:
            self.npy = name

        self.action_npy = os.path.join(action_path, self.npy)

        self.reward_dict_npy = os.path.join(reward_path, self.npy)

        self.state_npy = os.path.join(state_path, self.npy)

        if 'E2' in self.name:
            self.alpha = 2
        elif 'E1p5' in self.name:
            self.alpha = 1.5
        elif 'E1' in self.name:
            self.alpha = 1
        elif 'Ep5' in self.name:
            self.alpha = 0.5
        elif 'E0' in self.name:
            self.alpha = 0

        self.__calc_P_PI()

    def __calc_P_PI(self):
        dictionary = np.load(self.reward_dict_npy).item()

        P_mean = dictionary.get('ori-return-average')
        P_std = dictionary.get('ori-return-std')
        P_mean_ene = dictionary.get('total-energy-average')
        P_std_ene = dictionary.get('total-energy-std')

        self.P=P_mean
        self.P_std=P_std
        self.E_mean=P_mean_ene
        self.E_std=P_std_ene
        self.PI=P_mean/P_mean_ene


    def eval(self,exp):
        if 'P' == exp:
            r=self.P
        elif 'PI'==exp:
            r=self.PI
        elif 'E'==exp:
            r=self.E_mean
        return r

    def check_complete_data(self):
        if len(np.shape(np.load(self.action_npy)))==3:
            return True
        else:
            return False


def PCA(X):
    SS = StandardScaler()
    X = SS.fit_transform(X)

    cov_mat = np.cov(X.T)


    eig_vals, eig_vecs = np.linalg.eig(cov_mat)


    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]


    eig_pairs.sort(key=lambda x: x[0], reverse=True)

    return eig_pairs,eig_vals, eig_vecs,SS