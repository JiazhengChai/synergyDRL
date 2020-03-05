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


plt.rcParams["figure.figsize"] = (25,12)
plt.rcParams["font.size"] = 20

cwd=os.getcwd()

path_to_folder=cwd+'/experiments_results/Synergy/all_csv/process_SA_final_summary'


fixed_scale=True
for agent in os.listdir(path_to_folder):
    path_to_agent=os.path.join(path_to_folder,agent)
    for csv in os.listdir(path_to_agent):
        path_to_csv = os.path.join(path_to_agent, csv)
        csv_file = pd.read_csv(path_to_csv)
        FP_mean = csv_file['FP mean']
        FP_std = csv_file['FP std']

        FPI_mean = csv_file['FPI mean']
        FPI_std = csv_file['FPI std']

        FE_mean = csv_file['FE mean']
        FE_std = csv_file['FE std']
        '''if agent=='HC':
            FP_mean=csv_file['FP mean']
            FP_std= csv_file['FP std']

            FPI_mean = csv_file['FPI mean']
            FPI_std= csv_file['FPI std']

            FE_mean = csv_file['FE mean']
            FE_std = csv_file['FE std']

        elif agent=='HCheavy':
            FP_mean = csv_file['FP mean']
            FP_std = csv_file['FP std']

            FPI_mean = csv_file['FPI mean']
            FPI_std = csv_file['FPI std']

            FE_mean = csv_file['FE mean']
            FE_std = csv_file['FE std']
        elif agent == 'FC':
            FP_mean = csv_file['FP mean']
            FP_std = csv_file['FP std']

            FPI_mean = csv_file['FPI mean']
            FPI_std = csv_file['FPI std']

            FE_mean = csv_file['FE mean']
            FE_std = csv_file['FE std']'''




    fig = plt.figure()
    ax_FP = fig.add_subplot(1,3,1)
    ax_FPI = fig.add_subplot(1,3,2)
    ax_FE = fig.add_subplot(1,3,3)


    barWidth = 0.25

    FP_SAC = FP_mean[0]
    FP_TD3 = FP_mean[1]
    FP_std_SAC =FP_std[0]
    FP_std_TD3 =FP_std[1]

    FPI_SAC = FPI_mean[0]
    FPI_TD3 = FPI_mean[1]
    FPI_std_SAC =FPI_std[0]
    FPI_std_TD3 =FPI_std[1]

    FE_SAC = FE_mean[0]
    FE_TD3 = FE_mean[1]
    FE_std_SAC =FE_std[0]
    FE_std_TD3 =FE_std[1]


    r1 = np.arange(len([FP_SAC]))
    r2 = [x + barWidth for x in r1]

    ax_FP.bar(r1, FP_SAC,yerr=FP_std_SAC, color='#7f6d5f', width=barWidth, edgecolor='white', label='SAC')
    ax_FP.bar(r2, FP_TD3,yerr=FP_std_TD3, color='#557f2d', width=barWidth, edgecolor='white', label='TD3')

    ax_FP.set_ylabel('Rewards')
    ax_FP.set_xticks([r + barWidth for r in range(len([FP_SAC]))])

    ax_FP.set_xticklabels([agent])

    ax_FP.set_title('Performance')
    if fixed_scale:
        if 'VA' in agent:
            ax_FP.set_ylim([-600, 50])
        else:
            ax_FP.set_ylim([0, 16000])


    r1 = np.arange(len([FPI_SAC]))
    r2 = [x + barWidth for x in r1]

    ax_FPI.bar(r1, FPI_SAC,yerr=FPI_std_SAC, color='#7f6d5f', width=barWidth, edgecolor='white', label='SAC')
    ax_FPI.bar(r2, FPI_TD3,yerr=FPI_std_TD3, color='#557f2d', width=barWidth, edgecolor='white', label='TD3')

    ax_FPI.set_ylabel('Rewards per energy consumed')

    ax_FPI.set_xlabel('Agent', fontweight='bold')

    ax_FPI.set_xticks([r + barWidth for r in range(len([FPI_SAC]))])

    ax_FPI.set_xticklabels([agent])

    ax_FPI.set_title('Performance Index')
    if fixed_scale:
        if 'VA' in agent:
            ax_FPI.set_ylim([-100, 50])
        else:
            ax_FPI.set_ylim([0, 13])
    r1 = np.arange(len([FE_SAC]))
    r2 = [x + barWidth for x in r1]

    ax_FE.bar(r1, FE_SAC,yerr=FE_std_SAC, color='#7f6d5f', width=barWidth, edgecolor='white', label='SAC')
    ax_FE.bar(r2, FE_TD3,yerr=FE_std_TD3, color='#557f2d', width=barWidth, edgecolor='white', label='TD3')
    ax_FE.set_ylabel('Energy consumed')


    ax_FE.set_xticks([r + barWidth for r in range(len([FE_SAC]))])
    #ax_FE.set_xticklabels(['HC', 'HeavyHC', 'FC'])
    ax_FE.set_xticklabels([agent])

    ax_FE.set_title('Energy consumed')
    if fixed_scale:
        if 'VA' in agent:
            ax_FE.set_ylim([0, 20])
        else:
            ax_FE.set_ylim([0, 2500])

    ax_FE.legend()
    if not os.path.exists(cwd+'/experiments_results/Synergy/histograms'):
        os.makedirs(cwd+'/experiments_results/Synergy/histograms',exist_ok=True)

    plt.savefig(cwd+'/experiments_results/Synergy/histograms/P_one_hitsogram_'+agent+'.png')


