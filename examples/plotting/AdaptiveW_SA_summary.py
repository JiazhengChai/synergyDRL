from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt
import os
from matplotlib.lines import Line2D
from exp_variant_class import exp_variant#,PCA
from sklearn.decomposition import PCA
import argparse
from scipy import integrate
import csv
from scipy.stats.stats import pearsonr
import pandas as pd
def gauss(x, mu, a = 1, sigma = 1/6):
    return a * np.exp(-(x - mu)**2 / (2*sigma**2))

def R2():

    return r'R^{{{e:d}}}'.format(e=int(2))

cmap = plt.cm.viridis
cmaplist = [cmap(i) for i in range(cmap.N)]
cmaplen=len(cmaplist)

color_list=['b','r','g','c','m','y','k','#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf']

plt.rcParams["figure.figsize"] = (10,8)

parser = argparse.ArgumentParser()

parser.add_argument('--agentt',
                    type=str,choices=['HCheavy','HC','A','Antheavy','FC','Ctp','G','HC_E1','HC4dof','HC5dof','HC3doff','HC3dofb','HC2dof','VA','VA4dof','VA6dof','VA8dof'])

args = parser.parse_args()

tif=False
sortt=False
standscale=True
temporal=True
manual_pca=False
recon_num=8

ori_total_vec_rsq=9
truncated_start=200
dll=50

std=True

agentt=args.agentt

precheck=False

if 'HC' in agentt and 'dof' not in agentt:
    total_vec = 6
    total_chk=30

    ori_final = 3000
    ori_begin = 100
    ori_step = 100
    x_speed_index=8
    desired_dist=500


elif 'FC' in agentt:
    total_vec = 12
    total_chk=30
    #total_chk=2
    ori_final = 3000
    ori_begin = 100#100
    #ori_step = 2900
    ori_step = 100
    x_speed_index=14
    desired_dist=500
    #ori_final = 1100
    #ori_begin = 500#1000
    #ori_step = 100
elif agentt=='HC4dof':
    total_vec = 4
    total_chk=30
    ori_final = 3000
    ori_begin = 100
    ori_step = 100
    x_speed_index=6
    desired_dist=500
elif agentt=='HC2dof':
    total_vec = 2
    total_chk=30
    ori_final = 3000
    ori_begin = 100
    ori_step = 100
    x_speed_index=4
    desired_dist=500

elif agentt == 'HC3doff':
    total_vec = 3
    total_chk = 30
    ori_final = 3000
    ori_begin = 100
    ori_step = 100
    x_speed_index = 5
    desired_dist = 500
    action_path = './experiments_results/collected_actions/trajectory_npy/actions_npy'
    reward_path = './experiments_results/collected_actions/trajectory_npy/reward_energy_dict'
    state_path = './experiments_results/collected_actions/trajectory_npy/states_npy'
    agentt_folder = 'HC3doff'

elif agentt == 'HC3dofb':
    total_vec = 3
    total_chk = 30
    ori_final = 3000
    ori_begin = 100
    ori_step = 100
    x_speed_index = 5
    desired_dist = 500
    action_path = './experiments_results/collected_actions/trajectory_npy/actions_npy'
    reward_path = './experiments_results/collected_actions/trajectory_npy/reward_energy_dict'
    state_path = './experiments_results/collected_actions/trajectory_npy/states_npy'
    agentt_folder = 'HC3dofb'

elif agentt == 'HC5dof':
    total_vec = 5
    total_chk = 30
    ori_final = 3000
    ori_begin = 100
    ori_step = 100
    x_speed_index = 7
    desired_dist = 500
    action_path = './experiments_results/collected_actions/trajectory_npy/actions_npy'
    reward_path = './experiments_results/collected_actions/trajectory_npy/reward_energy_dict'
    state_path = './experiments_results/collected_actions/trajectory_npy/states_npy'
    agentt_folder = 'HC5dof'

elif agentt == 'VA':
    total_vec = 2
    total_chk = 30
    ori_final = 390
    ori_begin = 13
    ori_step = 13
    x_speed_index = None
    desired_dist = 500
    action_path = './experiments_results/collected_actions/trajectory_npy/actions_npy'
    reward_path = './experiments_results/collected_actions/trajectory_npy/reward_energy_dict'
    state_path = './experiments_results/collected_actions/trajectory_npy/states_npy'
    agentt_folder = 'VA'
    dll = 400
    truncated_start= 300
    #standscale = False
elif agentt == 'VA4dof':
    total_vec = 4
    total_chk = 30
    ori_final = 390
    ori_begin = 13
    ori_step = 13
    x_speed_index = None
    desired_dist = 500
    action_path = './experiments_results/collected_actions/trajectory_npy/actions_npy'
    reward_path = './experiments_results/collected_actions/trajectory_npy/reward_energy_dict'
    state_path = './experiments_results/collected_actions/trajectory_npy/states_npy'
    agentt_folder = 'VA4dof'
    dll = 400
    truncated_start= 300
    #standscale = False
elif agentt == 'VA6dof':
    total_vec = 6
    total_chk = 30
    ori_final = 390
    ori_begin = 13
    ori_step = 13
    x_speed_index = None
    desired_dist = 500
    joint_list = ['shoulder','shoulder2', 'elbow','elbow2','elbow3','elbow4']
    action_path = './experiments_results/collected_actions/trajectory_npy/actions_npy'
    reward_path = './experiments_results/collected_actions/trajectory_npy/reward_energy_dict'
    state_path = './experiments_results/collected_actions/trajectory_npy/states_npy'
    agentt_folder = 'VA6dof'
    dll = 400
    truncated_start= 300

elif agentt == 'VA8dof':
    total_vec = 8
    total_chk = 30
    ori_final = 390
    ori_begin = 13
    ori_step = 13
    x_speed_index = None
    desired_dist = 500
    joint_list = ['shoulder','shoulder2','shoulder3','shoulder4', 'elbow','elbow2','elbow3','elbow4']
    action_path = './experiments_results/collected_actions/trajectory_npy/actions_npy'
    reward_path = './experiments_results/collected_actions/trajectory_npy/reward_energy_dict'
    state_path = './experiments_results/collected_actions/trajectory_npy/states_npy'
    agentt_folder = 'VA8dof'
    dll = 400
    truncated_start= 300
top_folder=agentt

cwd=os.getcwd()
path_to_folder=cwd+'/experiments_results/Synergy/all_csv/process_SA_intermediate'

final = ori_final
begin = ori_begin
step = ori_step

path_to_csv=path_to_folder+'/'+agentt

output_folder=cwd+'/experiments_results/Synergy/all_csv/process_SA_final_summary/'+agentt
if not os.path.exists(output_folder):
    os.makedirs(output_folder, exist_ok=True)

process_csv = open(output_folder+ '/' + agentt +'_final_summary.csv', 'w')

writer = csv.writer(process_csv, lineterminator='\n')

writer.writerow(['Algorithms',

                 'corr SA_P mean', 'corr SA_P std',
                 'corr SA_PI mean', 'corr_SA_PI std',
                 'corr SA_E mean', 'corr SA_E std',

                'FP mean','FP std',
                'FPI mean','FPI std',
                'FE mean','FE std',

                'FSA mean','FSA std',
                'DSA mean','DSA std',
                'ASA mean','ASA std',


                'Corr FSA_FP','Corr FSA_FPI', 'Corr FSA_FE',
                'Corr DSA_FP','Corr DSA_FPI', 'Corr DSA_FE',
                'Corr ASA_FP','Corr ASA_FPI', 'Corr ASA_FE',

                 ])

TD3_data=[]
for csv_ in os.listdir(path_to_csv):
    current_csv = pd.read_csv(path_to_csv + '/' + csv_)

    trial_name_list=current_csv['Trials']
    counter=0
    for name in trial_name_list:
        if 'TD3' not in name:
            counter+=1


    corr_SAP_list = current_csv['Corr SA_P'][0:counter]
    corr_SAPI_list = current_csv['Corr SA_PI'][0:counter]
    corr_SAE_list = current_csv['Corr SA_E'][0:counter]

    FSA_list=current_csv['FSA'][0:counter]
    DSA_list = current_csv['DSA'][0:counter]
    ASA_list = current_csv['ASA'][0:counter]

    FP_list=current_csv['FP'][0:counter]
    FPI_list = current_csv['FPI'][0:counter]
    FE_list = current_csv['FE'][0:counter]

    corr_SAP_list = np.asarray(corr_SAP_list)
    corr_SAPI_list = np.asarray(corr_SAPI_list)
    corr_SAE_list = np.asarray(corr_SAE_list)

    FSA_list = np.asarray(FSA_list)
    DSA_list = np.asarray(DSA_list)
    ASA_list = np.asarray(ASA_list)

    FP_list = np.asarray(FP_list)
    FPI_list = np.asarray(FPI_list)
    FE_list = np.asarray(FE_list)

    trial_name_list= np.asarray(trial_name_list)

    corr_SAP_mean = np.mean(corr_SAP_list)
    corr_SAP_std = np.mean(corr_SAP_list)
    corr_SAPI_mean = np.mean(corr_SAPI_list)
    corr_SAPI_std = np.mean(corr_SAPI_list)
    corr_SAE_mean = np.mean(corr_SAE_list)
    corr_SAE_std = np.mean(corr_SAE_list)

    FP_mean=np.mean(FP_list)
    FP_std=np.std(FP_list)
    FPI_mean=np.mean(FPI_list)
    FPI_std=np.std(FPI_list)
    FE_mean=np.mean(FE_list)
    FE_std=np.std(FE_list)

    FSA_mean = np.mean(FSA_list)
    FSA_std = np.std(FSA_list)
    DSA_mean = np.mean(DSA_list)
    DSA_std = np.std(DSA_list)
    ASA_mean = np.mean(ASA_list)
    ASA_std = np.std(ASA_list)



    corr_FSA_FP = np.corrcoef(FSA_list, FP_list)[0, 1]
    corr_FSA_FPI = np.corrcoef(FSA_list, FPI_list)[0, 1]
    corr_FSA_FE = np.corrcoef(FSA_list, FE_list)[0, 1]
    corr_DSA_FP = np.corrcoef(DSA_list, FP_list)[0, 1]
    corr_DSA_FPI = np.corrcoef(DSA_list, FPI_list)[0, 1]
    corr_DSA_FE = np.corrcoef(DSA_list, FE_list)[0, 1]
    corr_ASA_FP = np.corrcoef(ASA_list, FP_list)[0, 1]
    corr_ASA_FPI = np.corrcoef(ASA_list, FPI_list)[0, 1]
    corr_ASA_FE = np.corrcoef(ASA_list, FE_list)[0, 1]

    writer.writerow(['SAC',corr_SAP_mean,corr_SAP_std,corr_SAPI_mean,corr_SAPI_std,corr_SAE_mean,corr_SAE_std,
                     FP_mean,FP_std,FPI_mean,FPI_std,FE_mean,FE_std,
                     FSA_mean,FSA_std,DSA_mean,DSA_std,ASA_mean,ASA_std,
                     corr_FSA_FP,corr_FSA_FPI,corr_FSA_FE,
                     corr_DSA_FP,corr_DSA_FPI,corr_DSA_FE,
                     corr_ASA_FP,corr_ASA_FPI,corr_ASA_FE
                     ])

    corr_SAP_list = current_csv['Corr SA_P'][counter::]
    corr_SAPI_list = current_csv['Corr SA_PI'][counter::]
    corr_SAE_list = current_csv['Corr SA_E'][counter::]

    FSA_list = current_csv['FSA'][counter::]
    DSA_list = current_csv['DSA'][counter::]
    ASA_list = current_csv['ASA'][counter::]

    FP_list = current_csv['FP'][counter::]
    FPI_list = current_csv['FPI'][counter::]
    FE_list = current_csv['FE'][counter::]

    corr_SAP_list = np.asarray(corr_SAP_list)
    corr_SAPI_list = np.asarray(corr_SAPI_list)
    corr_SAE_list = np.asarray(corr_SAE_list)

    FSA_list = np.asarray(FSA_list)
    DSA_list = np.asarray(DSA_list)
    ASA_list = np.asarray(ASA_list)

    FP_list = np.asarray(FP_list)
    FPI_list = np.asarray(FPI_list)
    FE_list = np.asarray(FE_list)

    trial_name_list = np.asarray(trial_name_list)

    corr_SAP_mean = np.mean(corr_SAP_list)
    corr_SAP_std = np.mean(corr_SAP_list)
    corr_SAPI_mean = np.mean(corr_SAPI_list)
    corr_SAPI_std = np.mean(corr_SAPI_list)
    corr_SAE_mean = np.mean(corr_SAE_list)
    corr_SAE_std = np.mean(corr_SAE_list)

    FP_mean = np.mean(FP_list)
    FP_std = np.std(FP_list)
    FPI_mean = np.mean(FPI_list)
    FPI_std = np.std(FPI_list)
    FE_mean = np.mean(FE_list)
    FE_std = np.std(FE_list)

    FSA_mean = np.mean(FSA_list)
    FSA_std = np.std(FSA_list)
    DSA_mean = np.mean(DSA_list)
    DSA_std = np.std(DSA_list)
    ASA_mean = np.mean(ASA_list)
    ASA_std = np.std(ASA_list)

    corr_FSA_FP = np.corrcoef(FSA_list, FP_list)[0, 1]
    corr_FSA_FPI = np.corrcoef(FSA_list, FPI_list)[0, 1]
    corr_FSA_FE = np.corrcoef(FSA_list, FE_list)[0, 1]
    corr_DSA_FP = np.corrcoef(DSA_list, FP_list)[0, 1]
    corr_DSA_FPI = np.corrcoef(DSA_list, FPI_list)[0, 1]
    corr_DSA_FE = np.corrcoef(DSA_list, FE_list)[0, 1]
    corr_ASA_FP = np.corrcoef(ASA_list, FP_list)[0, 1]
    corr_ASA_FPI = np.corrcoef(ASA_list, FPI_list)[0, 1]
    corr_ASA_FE = np.corrcoef(ASA_list, FE_list)[0, 1]

    writer.writerow(['TD3', corr_SAP_mean, corr_SAP_std, corr_SAPI_mean, corr_SAPI_std, corr_SAE_mean, corr_SAE_std,
                     FP_mean, FP_std, FPI_mean, FPI_std, FE_mean, FE_std,
                     FSA_mean, FSA_std, DSA_mean, DSA_std, ASA_mean, ASA_std,
                     corr_FSA_FP, corr_FSA_FPI, corr_FSA_FE,
                     corr_DSA_FP, corr_DSA_FPI, corr_DSA_FE,
                     corr_ASA_FP, corr_ASA_FPI, corr_ASA_FE
                     ])

process_csv.close()


