import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA

plt.rcParams['figure.figsize'] = [15, 12]
plt.rcParams['axes.linewidth'] = 2.
plt.rcParams['font.size'] = 35
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['font.family'] = 'Times New Roman'

LW=5
side=160
LGS=30
DPI=350
trans_rate=0.3

cwd=os.getcwd()
HC_folder=cwd+'/experiments_results/Synergy/all_csv/raw_csv/HC'
HeavyHC_folder=cwd+'/experiments_results/Synergy/all_csv/raw_csv/HCheavy'
FC_folder=cwd+'/experiments_results/Synergy/all_csv/raw_csv/FC'

available_list=[]
for folder in [HC_folder,HeavyHC_folder,FC_folder]:
    if os.path.exists(folder):
        available_list.append(folder)

print(available_list)

def my_as_si(x, ndp):
    s = '{x:0.{ndp:d}e}'.format(x=x, ndp=ndp)
    m, e = s.split('e')
    return r'\times 10^{{{e:d}}}'.format(e=int(e))

c_SA='C4'
c_P='C2'
c_PI='C1'


fixed_scale=True
agentt_type='HC'
algo='SAC'
save_path=cwd+'/experiments_results/Synergy/learning_progress_graphs'
if not os._exists(save_path):
    os.makedirs(save_path,exist_ok=True)
get_rid_div=False
div_rate = 0.2
choice=FC_folder

for choice in available_list:
    counter_sac=0
    counter_TD3=0
    for f in os.listdir(choice):
        file_path=os.path.join(choice,f)
        current_file = pd.read_csv(file_path)
        if 'TD3' not in f:
            counter_sac=counter_sac+1
            if counter_sac==1:
                current_SAC_surface_list=np.asarray(current_file['Surface Area'])
                current_SAC_P_list = np.asarray(current_file['P'])
                current_SAC_PI_list = np.asarray(current_file['PI'])
                current_SAC_E_list =np.asarray( current_file['E'])
            else:
                current_SAC_surface_list=np.vstack((current_SAC_surface_list,np.asarray(current_file['Surface Area'])))
                current_SAC_P_list=np.vstack((current_SAC_P_list,np.asarray(current_file['P'])))
                current_SAC_PI_list=np.vstack((current_SAC_PI_list,np.asarray(current_file['PI'])))
                current_SAC_E_list=np.vstack((current_SAC_E_list,np.asarray(current_file['E'])))
        else:
            counter_TD3 = counter_TD3 + 1
            if counter_TD3 == 1:
                current_TD3_surface_list = np.asarray(current_file['Surface Area'])
                current_TD3_P_list = np.asarray(current_file['P'])
                current_TD3_PI_list = np.asarray(current_file['PI'])
                current_TD3_E_list = np.asarray(current_file['E'])
            else:
                current_TD3_surface_list = np.vstack((current_TD3_surface_list, np.asarray(current_file['Surface Area'])))
                current_TD3_P_list = np.vstack((current_TD3_P_list, np.asarray(current_file['P'])))
                current_TD3_PI_list = np.vstack((current_TD3_PI_list, np.asarray(current_file['PI'])))
                current_TD3_E_list = np.vstack((current_TD3_E_list, np.asarray(current_file['E'])))

    mean_P_SAC=np.flip(np.mean(current_SAC_P_list,axis=0),axis=0)
    mean_PI_SAC=np.flip(np.mean(current_SAC_PI_list,axis=0),axis=0)
    mean_surface_SAC=np.flip(np.mean(current_SAC_surface_list,axis=0),axis=0)
    std_surface_SAC=np.flip(np.std(current_SAC_surface_list,axis=0),axis=0)
    std_P_SAC=np.flip(np.std(current_SAC_P_list,axis=0),axis=0)
    std_PI_SAC=np.flip(np.std(current_SAC_PI_list,axis=0),axis=0)

    mean_P_TD3=np.flip(np.mean(current_TD3_P_list,axis=0),axis=0)
    mean_PI_TD3=np.flip(np.mean(current_TD3_PI_list,axis=0),axis=0)
    mean_surface_TD3=np.flip(np.mean(current_TD3_surface_list,axis=0),axis=0)
    std_P_TD3=np.flip(np.std(current_TD3_P_list,axis=0),axis=0)
    std_PI_TD3=np.flip(np.std(current_TD3_PI_list,axis=0),axis=0)
    std_surface_TD3=np.flip(np.std(current_TD3_surface_list,axis=0),axis=0)

    P_all=np.vstack((current_SAC_P_list,current_TD3_P_list))
    PI_all=np.vstack((current_SAC_PI_list,current_TD3_PI_list))
    SA_all=np.vstack((current_SAC_surface_list,current_TD3_surface_list))

    mean_P_all=np.flip(np.mean(P_all,axis=0),axis=0)
    mean_PI_all=np.flip(np.mean(PI_all,axis=0),axis=0)
    mean_surface_all=np.flip(np.mean(SA_all,axis=0),axis=0)
    std_P_all=np.flip(np.std(P_all,axis=0),axis=0)
    std_PI_all=np.flip(np.std(PI_all,axis=0),axis=0)
    std_surface_all=np.flip(np.std(SA_all,axis=0),axis=0)

    if choice==HC_folder:
        llist=[ ('HC', 'SAC'),
        ('HC', 'TD3')]
    elif choice==HeavyHC_folder:
        llist=[ ('HeavyHC', 'SAC'),
        ('HeavyHC', 'TD3')]
    elif choice==FC_folder:
        llist=[ ('FC', 'SAC'),
        ('FC', 'TD3')]


    for agentt_type, algo in llist:
        if algo=='SAC':
            surface_area_w=mean_surface_SAC
            P_list=mean_P_SAC
            PI_list=mean_PI_SAC

            std_surface_area_w=std_surface_SAC
            std_P_list=std_P_SAC
            std_PI_list=std_PI_SAC

        elif algo=='TD3':
            surface_area_w=mean_surface_TD3
            P_list=mean_P_TD3
            PI_list=mean_PI_TD3

            std_surface_area_w=std_surface_TD3
            std_P_list=std_P_TD3
            std_PI_list=std_PI_TD3

        surface_plot_w_ax = host_subplot(111, axes_class=AA.Axes)
        plt.subplots_adjust(right=0.75)

        ax2_w = surface_plot_w_ax.twinx()
        ax3_w = surface_plot_w_ax.twinx()

        new_fixed_axis = ax2_w.get_grid_helper().new_fixed_axis
        ax2_w.axis["right"] = new_fixed_axis(loc="right",
                                             axes=ax2_w,
                                             offset=(0, 0))

        ax2_w.axis["right"].toggle(all=True)

        new_fixed_axis = ax3_w.get_grid_helper().new_fixed_axis
        ax3_w.axis["right"] = new_fixed_axis(loc="right",
                                             axes=ax3_w,
                                             offset=(side, 0))

        ax3_w.axis["right"].toggle(all=True)

        surface_plot_w_ax.set_ylabel('Surface Area', color=c_SA)
        surface_plot_w_ax.set_xlabel(r"${0:s}$ timesteps".format(my_as_si(1e5, 2)))
        ax2_w.set_ylabel('Performance', color=c_P)
        ax3_w.set_ylabel('Performance-energy', color=c_PI)


        if fixed_scale:
            if agentt_type == 'HC':
                ax2_w.set_ylim([0, 20000])
            elif agentt_type == 'HeavyHC':
                ax2_w.set_ylim([0, 8000])
            elif agentt_type == 'FC':
                ax2_w.set_ylim([0, 23000])

            if agentt_type == 'HC':
                ax3_w.set_ylim([0, 8])
            elif agentt_type == 'HeavyHC':
                ax3_w.set_ylim([0, 5])
            elif agentt_type == 'FC':
                ax3_w.set_ylim([0, 6])

            if agentt_type == 'HC':
                surface_plot_w_ax.set_ylim([3, 4.5])
            elif agentt_type == 'HeavyHC':
               surface_plot_w_ax.set_ylim([3, 4.5])
            elif agentt_type == 'FC':
                surface_plot_w_ax.set_ylim([6, 11])

        if get_rid_div:
            bad_ind_list = []
            for ind, p in enumerate(PI_list):
                if ind > 0 and ind < (len(PI_list) - 1):
                    if abs(p - PI_list[ind - 1]) / abs(p) > div_rate and abs(p - PI_list[ind + 1]) / abs(
                            p) > div_rate:
                        bad_ind_list.append(ind)


        if get_rid_div and len(bad_ind_list) > 0:
            print('DIV')

            P_list = np.delete(P_list, bad_ind_list, 0)
            PI_list = np.delete(PI_list, bad_ind_list, 0)


            surface_area_w = np.delete(surface_area_w, bad_ind_list, 0)


        surface_plot_w_ax.plot(range(1, len(surface_area_w) + 1), surface_area_w, color=c_SA,
                                                       label='Surface area',linewidth=LW)
        surface_plot_w_ax.fill_between(range(1, len(surface_area_w) + 1),surface_area_w+std_surface_area_w,
                                       surface_area_w-std_surface_area_w, facecolor=c_SA, alpha=trans_rate)

        ax2_w.plot(range(1, len(surface_area_w) + 1), P_list, color=c_P,
                   label='Performance',linewidth=LW)
        ax2_w.fill_between(range(1, len(surface_area_w) + 1),P_list+std_P_list,
                           P_list-std_P_list, facecolor=c_P, alpha=trans_rate)

        ax3_w.plot(range(1, len(surface_area_w) + 1), PI_list, color=c_PI,
                   label='Performance-energy',linewidth=LW)
        ax3_w.fill_between(range(1, len(surface_area_w) + 1),PI_list+std_PI_list,
                           PI_list-std_PI_list, facecolor=c_PI, alpha=trans_rate)

        surface_plot_w_ax.legend(loc=2,prop={'size': LGS})

        plt.tight_layout()

        '''if not fixed_scale:
            plt.savefig(
                os.path.join(path, 'Surface_weigthed_all_' + type_ + ex + '_' + str(min_rsq)) + '.jpg')
        else:
            plt.savefig(os.path.join(path, 'Fixed_scale_surface_weigthed_all_' + type_ + ex + '_' + str(
                min_rsq) + '.jpg'))'''

        plt.savefig(os.path.join(save_path, 'Learning_progress_' + agentt_type  + '_' +algo  + '.png'),format = 'png', dpi=DPI)

        plt.close('all')








