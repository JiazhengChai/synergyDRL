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


plt.rcParams["figure.figsize"] = (37,10)
plt.rcParams["font.size"] = 30
cwd=os.getcwd()

path_to_folder=cwd+'/experiments_results/Synergy/all_csv/process_SA_final_summary'


for agent in os.listdir(path_to_folder):
    path_to_agent=os.path.join(path_to_folder,agent)
    for csv in os.listdir(path_to_agent):
        path_to_csv = os.path.join(path_to_agent, csv)
        csv_file = pd.read_csv(path_to_csv)
        if agent=='HC':
            FSA_mean_HC=csv_file['FP mean']
            FSA_std_HC= csv_file['FP std']

            DSA_mean_HC = csv_file['FPI mean']
            DSA_std_HC = csv_file['FPI std']

            ASA_mean_HC = csv_file['FE mean']
            ASA_std_HC = csv_file['FE std']

        elif agent=='HCheavy':
            FSA_mean_HCheavy = csv_file['FP mean']
            FSA_std_HCheavy = csv_file['FP std']

            DSA_mean_HCheavy = csv_file['FPI mean']
            DSA_std_HCheavy = csv_file['FPI std']

            ASA_mean_HCheavy = csv_file['FE mean']
            ASA_std_HCheavy = csv_file['FE std']
        elif agent == 'FC':
            FSA_mean_FC = csv_file['FP mean']
            FSA_std_FC = csv_file['FP std']

            DSA_mean_FC = csv_file['FPI mean']
            DSA_std_FC = csv_file['FPI std']

            ASA_mean_FC = csv_file['FE mean']
            ASA_std_FC = csv_file['FE std']

FSA_mean_zip=[]
for HC_val,HCheavy_val,FC_val in zip(FSA_mean_HC,FSA_mean_HCheavy,FSA_mean_FC):
    FSA_mean_zip.append([HC_val,HCheavy_val,FC_val])

FSA_std_zip = []
for HC_val,HCheavy_val,FC_val in zip(FSA_std_HC,FSA_std_HCheavy,FSA_std_FC):
    FSA_std_zip.append([HC_val,HCheavy_val,FC_val])

DSA_mean_zip=[]
for HC_val,HCheavy_val,FC_val in zip(DSA_mean_HC,DSA_mean_HCheavy,DSA_mean_FC):
    DSA_mean_zip.append([HC_val,HCheavy_val,FC_val])

DSA_std_zip = []
for HC_val,HCheavy_val,FC_val in zip(DSA_std_HC,DSA_std_HCheavy,DSA_std_FC):
    DSA_std_zip.append([HC_val,HCheavy_val,FC_val])

ASA_mean_zip=[]
for HC_val,HCheavy_val,FC_val in zip(ASA_mean_HC,ASA_mean_HCheavy,ASA_mean_FC):
    ASA_mean_zip.append([HC_val,HCheavy_val,FC_val])

ASA_std_zip = []
for HC_val,HCheavy_val,FC_val in zip(ASA_std_HC,ASA_std_HCheavy,ASA_std_FC):
    ASA_std_zip.append([HC_val,HCheavy_val,FC_val])

fig = plt.figure()
ax_FSA = fig.add_subplot(1,3,1)
ax_ASA = fig.add_subplot(1,3,2)
ax_DSA = fig.add_subplot(1,3,3)


# set width of bar
barWidth = 0.25

# set height of bar
FSAHC = FSA_mean_zip[0]
FSATD3 = FSA_mean_zip[1]
yerrFSAHC =FSA_std_zip[0]
yerrFSATD3 =FSA_std_zip[1]

DSAHC = DSA_mean_zip[0]
DSATD3 = DSA_mean_zip[1]
yerrDSAHC =DSA_std_zip[0]
yerrDSATD3 =DSA_std_zip[1]

ASAHC = ASA_mean_zip[0]
ASATD3 = ASA_mean_zip[1]
yerrASAHC =ASA_std_zip[0]
yerrASATD3 =ASA_std_zip[1]


r1 = np.arange(len(FSAHC))
r2 = [x + barWidth for x in r1]

ax_FSA.bar(r1, FSAHC,yerr=FSA_std_zip[0], color='#7f6d5f', width=barWidth, edgecolor='white', label='SAC')
ax_FSA.bar(r2, FSATD3,yerr=FSA_std_zip[1], color='#557f2d', width=barWidth, edgecolor='white', label='TD3')

ax_FSA.set_ylabel('Rewards')
ax_FSA.set_xticks([r + barWidth for r in range(len(FSAHC))])
ax_FSA.set_xticklabels(['HC', 'HeavyHC', 'FC'])

ax_FSA.set_title('Performance')

r1 = np.arange(len(DSAHC))
r2 = [x + barWidth for x in r1]

ax_DSA.bar(r1, DSAHC,yerr=DSA_std_zip[0], color='#7f6d5f', width=barWidth, edgecolor='white', label='SAC')
ax_DSA.bar(r2, DSATD3,yerr=DSA_std_zip[1], color='#557f2d', width=barWidth, edgecolor='white', label='TD3')

ax_DSA.set_ylabel('Rewards per energy consumed')

ax_DSA.set_xticks([r + barWidth for r in range(len(DSAHC))])
ax_DSA.set_xticklabels(['HC', 'HeavyHC', 'FC'])

ax_DSA.set_title('Performance Index')
ax_DSA.legend(loc=(1,0.9))


r1 = np.arange(len(ASAHC))
r2 = [x + barWidth for x in r1]

ax_ASA.bar(r1, ASAHC,yerr=ASA_std_zip[0], color='#7f6d5f', width=barWidth, edgecolor='white', label='SAC')
ax_ASA.bar(r2, ASATD3,yerr=ASA_std_zip[1], color='#557f2d', width=barWidth, edgecolor='white', label='TD3')
ax_ASA.set_ylabel('Energy consumed')

ax_ASA.set_xlabel('Agent', fontweight='bold')
ax_ASA.set_xticks([r + barWidth for r in range(len(ASAHC))])
ax_ASA.set_xticklabels(['HC', 'HeavyHC', 'FC'])

ax_ASA.set_title('Energy consumed')


if not os.path.exists(cwd+'/experiments_results/Synergy/histograms'):
    os.makedirs(cwd+'/experiments_results/Synergy/histograms',exist_ok=True)
plt.savefig(cwd+'/experiments_results/Synergy/histograms/P_hitsogram.png')


