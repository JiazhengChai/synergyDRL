import numpy as np
from matplotlib import pyplot as plt
import os

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

for agent in os.listdir(path_to_folder):
    path_to_agent=os.path.join(path_to_folder,agent)
    for csv in os.listdir(path_to_agent):
        path_to_csv = os.path.join(path_to_agent, csv)
        csv_file = pd.read_csv(path_to_csv)
        FSA_mean = csv_file['FSA mean']
        FSA_std = csv_file['FSA std']

        DSA_mean = csv_file['DSA mean']
        DSA_std = csv_file['DSA std']

        ASA_mean = csv_file['ASA mean']
        ASA_std = csv_file['ASA std']
        '''if agent=='HC':
            FSA_mean=csv_file['FSA mean']
            FSA_std= csv_file['FSA std']

            DSA_mean = csv_file['DSA mean']
            DSA_std = csv_file['DSA std']

            ASA_mean = csv_file['ASA mean']
            ASA_std = csv_file['ASA std']

        elif agent=='HCheavy':
            FSA_mean = csv_file['FSA mean']
            FSA_std = csv_file['FSA std']

            DSA_mean = csv_file['DSA mean']
            DSA_std = csv_file['DSA std']

            ASA_mean = csv_file['ASA mean']
            ASA_std = csv_file['ASA std']
        elif agent == 'FC':
            FSA_mean = csv_file['FSA mean']
            FSA_std = csv_file['FSA std']

            DSA_mean = csv_file['DSA mean']
            DSA_std = csv_file['DSA std']

            ASA_mean = csv_file['ASA mean']
            ASA_std = csv_file['ASA std']'''




    FSA_SAC = FSA_mean[0]
    FSA_TD3 = FSA_mean[1]
    FSA_std_SAC =FSA_std[0]
    FSA_std_TD3 =FSA_std[1]

    DSA_SAC = DSA_mean[0]
    DSA_TD3 = DSA_mean[1]
    DSA_std_SAC =DSA_std[0]
    DSA_std_TD3 =DSA_std[1]

    ASA_SAC = ASA_mean[0]
    ASA_TD3 = ASA_mean[1]
    ASA_std_SAC =ASA_std[0]
    ASA_std_TD3 =ASA_std[1]

    fig = plt.figure()
    ax_FSA = fig.add_subplot(1,3,1)
    ax_DSA = fig.add_subplot(1,3,2)
    ax_ASA = fig.add_subplot(1,3,3)

    barWidth = 0.25

    r1 = np.arange(len([FSA_SAC]))
    r2 = [x + barWidth for x in r1]

    ax_FSA.bar(r1, FSA_SAC,yerr=FSA_std_SAC, color='#7f6d5f', width=barWidth, edgecolor='white', label='SAC')
    ax_FSA.bar(r2, FSA_TD3,yerr=FSA_std_TD3, color='#557f2d', width=barWidth, edgecolor='white', label='TD3')

    ax_FSA.set_ylabel('Area')
    ax_FSA.set_xticks([r + barWidth for r in range(len([FSA_SAC]))])

    ax_FSA.set_xticklabels([agent])

    ax_FSA.set_title('FSA')



    r1 = np.arange(len([DSA_SAC]))
    r2 = [x + barWidth for x in r1]

    ax_DSA.bar(r1, DSA_SAC,yerr=DSA_std_SAC, color='#7f6d5f', width=barWidth, edgecolor='white', label='SAC')
    ax_DSA.bar(r2, DSA_TD3,yerr=DSA_std_TD3, color='#557f2d', width=barWidth, edgecolor='white', label='TD3')

    ax_DSA.set_ylabel('Area')

    ax_DSA.set_xlabel('Agent', fontweight='bold')

    ax_DSA.set_xticks([r + barWidth for r in range(len([DSA_SAC]))])

    ax_DSA.set_xticklabels([agent])

    ax_DSA.set_title('DSA')



    r1 = np.arange(len([ASA_SAC]))
    r2 = [x + barWidth for x in r1]

    ax_ASA.bar(r1, ASA_SAC,yerr=ASA_std_SAC, color='#7f6d5f', width=barWidth, edgecolor='white', label='SAC')
    ax_ASA.bar(r2, ASA_TD3,yerr=ASA_std_TD3, color='#557f2d', width=barWidth, edgecolor='white', label='TD3')
    ax_ASA.set_ylabel('Area')

    ax_ASA.set_xticks([r + barWidth for r in range(len([ASA_SAC]))])
    ax_ASA.set_xticklabels(['HC', 'HeavyHC', 'FC'])

    ax_ASA.set_title('ASA')

    ax_ASA.legend()
    if not os.path.exists(cwd + '/experiments_results/Synergy/histograms'):
        os.makedirs(cwd + '/experiments_results/Synergy/histograms', exist_ok=True)

    plt.savefig(cwd + '/experiments_results/Synergy/histograms/SA_hitsogram_' + agent + '.png')

