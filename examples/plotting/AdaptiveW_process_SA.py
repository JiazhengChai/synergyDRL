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
                    type=str,choices=['HCheavy','HC','A','Antheavy','FC','Ctp','G','HC_E1'])

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

if 'HC' in agentt:
    total_vec = 6
    total_chk=30
    ori_final = 3000
    ori_begin = 100
    ori_step = 100
    x_speed_index=8
    desired_dist=500
    if 'E1' in agentt:
        agentt_folder='HC_E1'
    elif 'heavy' in agentt:
        agentt_folder = 'HCheavy'
    else:
        agentt_folder = 'HC'

elif 'FC' in agentt:
    total_vec = 12
    total_chk=30
    ori_final = 3000
    ori_begin = 100
    ori_step = 100
    x_speed_index=14
    desired_dist=500
    agentt_folder='FC'

top_folder=agentt


file_path=os.path.abspath(os.getcwd())
path_list=file_path.split('/')
while path_list[-1] !="synergyDRL":
    path_list.pop(-1)

cwd="/".join(path_list)

path_to_folder=cwd+'/experiments_results/Synergy/all_csv/raw_csv'

final = ori_final
begin = ori_begin
step = ori_step

path_to_csv=path_to_folder+'/'+agentt_folder

output_folder=cwd+'/experiments_results/Synergy/all_csv/process_SA_intermediate/'+agentt
if not os.path.exists(output_folder):
    os.makedirs(output_folder, exist_ok=True)

process_csv = open(output_folder+ '/' + agentt +'_process_all_surface.csv', 'w')

writer = csv.writer(process_csv, lineterminator='\n')

writer.writerow(['Trials', 'Corr SA_P', 'Corr SA_PI', 'Corr SA_E','FSA', 'DSA', 'ASA','FP', 'FPI', 'FE'])

TD3_data=[]
for csv_ in os.listdir(path_to_csv):
    current_csv = pd.read_csv(path_to_csv + '/' + csv_)

    current_name_list=csv_.split('_')
    current_name_list=current_name_list[0:-2]
    name=''
    for cn in current_name_list:
        name=name+cn+'_'
    name=name[0:-1]

    P_list = current_csv['P']
    PI_list = current_csv['PI']
    E_list = current_csv['E']
    SA_list = current_csv['Surface Area']
    Checkpoint_list = current_csv['Checkpoint']
    P_list = np.asarray(P_list)
    PI_list = np.asarray(PI_list)
    E_list = np.asarray(E_list)
    SA_list = np.asarray(SA_list)
    Checkpoint_list = np.asarray(Checkpoint_list)

    corr_SA_P = np.corrcoef(SA_list, P_list)[0, 1]
    corr_SA_PI = np.corrcoef(SA_list, PI_list)[0, 1]
    corr_SA_E = np.corrcoef(SA_list, E_list)[0, 1]

    FP = P_list[0]
    FPI = PI_list[0]
    FE = E_list[0]

    FSA = SA_list[0]
    DSA = SA_list[0] - SA_list[-1]

    SA_list2 = np.copy(SA_list)
    ASA = 0
    neg_ASA = 0
    for sa in SA_list:
        for sa2 in SA_list2:
            diff = sa - sa2
            if diff >= 0 and diff > ASA:
                ASA = diff
            elif diff < 0:
                if diff < neg_ASA:
                    neg_ASA = diff
    if np.abs(neg_ASA) > ASA:
        ASA = neg_ASA

    if 'TD3' not in name:

        writer.writerow([name,corr_SA_P,corr_SA_PI,corr_SA_E,FSA,DSA,ASA,FP,FPI,FE])

    else:
        TD3_data.append([name,corr_SA_P,corr_SA_PI,corr_SA_E,FSA,DSA,ASA,FP,FPI,FE])

for row in TD3_data:
    writer.writerow(row)

process_csv.close()


