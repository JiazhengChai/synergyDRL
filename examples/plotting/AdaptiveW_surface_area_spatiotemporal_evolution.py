from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt
import os
from matplotlib.lines import Line2D
from exp_variant_class import exp_variant
from sklearn.decomposition import PCA
import argparse
from scipy import integrate
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA

def my_as_si(x, ndp):
    s = '{x:0.{ndp:d}e}'.format(x=x, ndp=ndp)
    m, e = s.split('e')
    return r'\times 10^{{{e:d}}}'.format(e=int(e))

cmap = plt.cm.viridis
cmaplist = [cmap(i) for i in range(cmap.N)]
cmaplen=len(cmaplist)

color_list=['b','r','g','c','m','y','k','#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf']

plt.rcParams["figure.figsize"] = (10,8)

parser = argparse.ArgumentParser()

parser.add_argument('--tr', nargs='+', required=True)

parser.add_argument('--ee',type=str, nargs='+',choices=['E0','E0_TD3','E1','E1_TD3'])

parser.add_argument('--agentt',
                    type=str,choices=['HCheavy','HC','A','Antheavy','FC','Ctp','G'])

args = parser.parse_args()

surface_weighting=True

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

elif 'FC' in agentt:
    total_vec = 12
    total_chk=30
    ori_final = 3000
    ori_begin = 100
    ori_step = 100
    x_speed_index=14
    desired_dist=500


get_rid_div=True
top_folder=agentt+'_surface_evolution'
if get_rid_div:
    div_rate=0.2
    top_folder=top_folder+'_no_div'


if args.agentt=='A' and 'E0' in args.ee:
    args.tr=['']+args.tr
for fixed_scale in [
                    True, False
                    ]:
    for tr in args.tr:

        for ee in args.ee:
            final = ori_final
            begin = ori_begin
            step = ori_step
            trial = tr
            subfolder=ee

            base= agentt + '_' + ee
            all_names=[]
            tmp=[]
            for cc in range(final, begin - step, -step):
                tmp.append(base + '_C' + str(cc) + trial)

            all_names.append(tmp)
            all_names.reverse()
            print(all_names)
            if precheck:
                top_tmp=[]
                for all_name in all_names:
                    tmpp = []
                    for n in all_name:
                        complete=exp_variant(n).check_complete_data()
                        if complete:
                            tmpp.append(n)
                    top_tmp.append(tmpp)

                all_names=top_tmp

            for dl, nl, mr, minr, t in [ (dll, True, True, 0, 'PI'),

                                       ]:

                type_ = t
                total_vec_rsq = ori_total_vec_rsq
                min_rsq = minr
                desired_length = dl

                named_label = nl
                manual_rsq = mr

                rsq_all_list = []
                for all_name in all_names:
                    agent = all_name[0] + '_spatiotemporal_evolution'

                    folder_name=agent+'_Compare_'+type_+'_'+str(desired_length)
                    if mr:
                        folder_name=folder_name+'_manual_rsq'
                    if standscale:
                        folder_name = folder_name + '_SS'

                    step=cmaplen//len(all_name)

                    color_list=[]
                    c=0
                    for l in range(len(all_name)):
                        color_list.append(cmaplist[c])
                        c+=step

                    all_label=[]
                    exp_variant_list=[]
                    for ind,n in enumerate(all_name):
                        exp_variant_list.append(exp_variant(n))
                        all_label.append(exp_variant_list[ind].eval(type_))
                    ############################################################################################################
                    if sortt:
                        new_index=sorted(range(len(all_label)), key=lambda k: all_label[k],reverse=True)
                        all_label=sorted(all_label,reverse=True)

                        tmp=[]
                        tmp_2=[]
                        for ni in new_index:
                            tmp.append(exp_variant_list[ni])
                            tmp_2.append(all_name[ni])
                        exp_variant_list=tmp
                        all_name=tmp_2

                    if named_label:
                        all_label=[]
                        for ind,n in enumerate(all_name):
                            prefix_list=exp_variant_list[ind].name.split('_')
                            for pp in prefix_list:
                                if 'C' in pp and 'H' not in pp:
                                    prefix=pp
                                    break
                            all_label.append(prefix + ': ' + '{:.2f}'.format(exp_variant_list[ind].eval(type_),2))
                    else:
                        all_label = []
                        for ind, n in enumerate(all_name):
                            all_label.append('{:.2f}'.format(exp_variant_list[ind].eval(type_), 2))
                    ############################################################################################################

                    combinations=[
                                  (True, '2_components_truncated', 2)
                                    ]
                    plot_r_sq=True
                    save=True

                    num_epi=10


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
                                                       offset=(60, 0))

                    ax3_w.axis["right"].toggle(all=True)

                    surface_plot_w_ax.set_ylabel('Surface Area', color='g')
                    surface_plot_w_ax.set_xlabel(r"${0:s}$ timesteps".format(my_as_si(1e5, 2)))
                    ax2_w.set_ylabel('Performance', color='b')
                    ax3_w.set_ylabel('Performance Index', color='c')

                    if fixed_scale:
                        if agentt == 'A' or agentt=='Antheavy':
                            ax2_w.set_ylim([0, 7000])
                        elif agentt == 'HC':
                            ax2_w.set_ylim([0, 17000])
                        elif agentt == 'HCheavy':
                            ax2_w.set_ylim([0, 10000])

                        if agentt == 'A' or agentt=='Antheavy':
                            ax3_w.set_ylim([0, 23])
                        elif agentt == 'HC':
                            ax3_w.set_ylim([2, 8])
                        elif agentt == 'HCheavy':
                            ax3_w.set_ylim([0, 6])

                    s = 0

                    surface_area=[]
                    surface_area_w=[]

                    P_list=[]
                    PI_list=[]
                    for n_ind,name in enumerate(all_name):

                        exp_variant_obj=exp_variant_list[n_ind]
                        truncated, version, ori_num_vec=(True, '2_components_truncated', 2)
                        P_list.append(exp_variant_obj.eval('P'))
                        PI_list.append(exp_variant_obj.eval('PI'))

                        if plot_r_sq:

                            rsq_label = []

                            X=np.load(exp_variant_obj.action_npy)
                            state_ = np.load(exp_variant_obj.state_npy)

                            mini = 1000
                            if X.shape == (10,):
                                # print('a')
                                for i in range(10):
                                    amin = np.asarray(X[i]).shape[0]
                                    if amin < mini:
                                        mini = amin
                                print(mini)

                                tmp = np.expand_dims(np.asarray(X[0])[0:mini, :], 0)
                                for i in range(9):
                                    tmp = np.vstack((tmp, np.expand_dims(np.asarray(X[i + 1])[0:mini, :], 0)))
                                print(tmp.shape)
                                X = tmp

                                tmp2 = np.expand_dims(np.asarray(state_[0])[0:mini, :], 0)
                                for i in range(9):
                                    tmp2 = np.vstack((tmp2, np.expand_dims(np.asarray(state_[i + 1])[0:mini, :], 0)))

                                state_ = tmp2
                            X=X[0:num_epi,:,:]
                            state_ = state_[0:num_epi, :, :]
                            distance = []
                            speed_record = state_[0, :, x_speed_index]
                            for i in range(len(speed_record)):
                                if i == 0:
                                    distance.append(speed_record[0])
                                else:
                                    distance.append(np.sum(speed_record[0:i]))

                            distance = np.asarray(distance)
                            if truncated:
                                total_vec_rsq = ori_total_vec_rsq

                                if mini == 1000 or mini > 300:
                                    current_dist = distance[truncated_start]
                                    end_dist_index = truncated_start
                                    tmp_dist = 0

                                    while tmp_dist < desired_dist and end_dist_index < len(distance) - 1:
                                        end_dist_index += 1
                                        tmp_dist = distance[end_dist_index] - current_dist

                                    remaining_index = end_dist_index - truncated_start

                                    desired_length=remaining_index
                                elif mini - desired_length >= 0:
                                    remaining_index = desired_length
                                    desired_length = remaining_index
                                if mini == 1000:
                                    X_truncated = X[:, truncated_start:truncated_start + desired_length, :]
                                else:
                                    if mini >= (truncated_start + desired_length):
                                        X_truncated = X[:, truncated_start:truncated_start + desired_length, :]

                                    elif mini - desired_length >= 0:

                                        X_truncated = X[:, mini - desired_length:mini, :]

                                    else:
                                        X_truncated = X[:, 0:mini, :]

                                        if mini>=ori_total_vec_rsq:
                                            total_vec_rsq=ori_total_vec_rsq
                                        else:
                                            total_vec_rsq=mini

                                X = X_truncated

                            X = X.reshape([X.shape[0], -1])

                            s_by_a = X
                            a_by_s = X.T

                            rsq_single_list=[]

                            if standscale:
                                mx = np.mean(X, axis=0)
                                X = X - mx

                            for num_vec_to_keep_ in range(1,total_vec_rsq+1):
                                if manual_pca:

                                    cov_mat = np.cov(X.T)

                                    # Compute the eigen values and vectors using numpy
                                    eig_vals, eig_vecs = np.linalg.eig(cov_mat)

                                    # Make a list of (eigenvalue, eigenvector) tuples
                                    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]

                                    # Sort the (eigenvalue, eigenvector) tuples from high to low
                                    eig_pairs.sort(key=lambda x: x[0], reverse=True)
                                else:
                                    pca = PCA(n_components=num_vec_to_keep_)
                                    pca.fit(X)
                                    eig_vecs = pca.components_
                                    eig_vals = pca.singular_values_
                                    eig_pairs = [(eig_vals[i], eig_vecs[i, :]) for i in range(len(eig_vals))]

                                num_features = X.shape[1]
                                percentage = sum(pca.explained_variance_ratio_)
                                proj_mat = eig_pairs[0][1].reshape(num_features, 1)

                                for eig_vec_idx in range(1, num_vec_to_keep_):
                                    proj_mat = np.hstack((proj_mat, eig_pairs[eig_vec_idx][1].reshape(num_features, 1)))

                                # Project the data
                                W = proj_mat

                                C = X.dot(W)
                                X_prime = C.dot(W.T)

                                if manual_rsq:

                                    Vm = np.mean(X, axis=0, keepdims=True)
                                    resid = X - np.dot(Vm, np.ones((X.shape[1],1 )))
                                    SST = np.linalg.norm(resid)
                                    SSE=np.linalg.norm(X-X_prime)

                                    rsq=1-SSE/SST
                                else:
                                    rsq = r2_score(X, X_prime)

                                rsq_single_list.append(rsq)

                            rsq_label.append('Rsq_' + exp_variant_obj.name)
                            rsq_all_list.append(rsq_single_list)

                            surface_area.append(integrate.simps(rsq_single_list,range(1,total_vec_rsq+1)))

                            rsq_single_list_modified=rsq_single_list*(np.asarray(range(9,0,-1))/10)
                            surface_area_w.append(integrate.simps(rsq_single_list_modified,range(1,total_vec_rsq+1)))

                            s += 1

                            if n_ind==0:

                                pass

                    path = 'experiments_results/Synergy/synergy_development_'+agentt+'/' + top_folder + '/' + subfolder + '/' + folder_name
                    os.makedirs(path, exist_ok=True)
                    ex = ''
                    if named_label:
                        ex = '_named'
                    P_list=list(reversed(P_list))
                    PI_list=list(reversed(PI_list))
                    surface_area=list(reversed(surface_area))
                    surface_area_w = list(reversed(surface_area_w))

                    P_list=np.asarray(P_list)
                    PI_list=np.asarray(PI_list)
                    surface_area=np.asarray(surface_area)
                    surface_area_w=np.asarray(surface_area_w)

                    if get_rid_div:
                        bad_ind_list = []
                        for ind, p in enumerate(P_list):
                            if ind > 0 and ind < (len(P_list) - 1):
                                if abs(p - P_list[ind - 1]) / abs(p) > div_rate and abs(p - P_list[ind + 1]) / abs(
                                        p) > div_rate:
                                    bad_ind_list.append(ind)

                    if get_rid_div and len(bad_ind_list) > 0:

                        P_list = np.delete(P_list, bad_ind_list, 0)
                        PI_list = np.delete(PI_list, bad_ind_list, 0)

                        surface_area = np.delete(surface_area, bad_ind_list, 0)
                        surface_area_w = np.delete(surface_area_w, bad_ind_list, 0)



                    if fixed_scale:
                        if agentt == 'A' or agentt=='Antheavy':
                            surface_plot_w_ax.set_ylim([1, 2])
                        else:
                            surface_plot_w_ax.set_ylim([1, 4.05])

                    if agentt!='A' and agentt!='Antheavy':
                        surface_plot_w_ax.plot(range(1,len(surface_area_w)+1),surface_area_w,color='g',label='Weighted surface area')#,color=color_list[s])

                        ax2_w.plot(range(1,len(surface_area_w)+1), P_list, color='b',
                                               label='P')

                        ax3_w.plot(range(1,len(surface_area_w)+1), PI_list, color='c',
                                               label='PI')
                    else:
                        x_range=np.asarray([i for i in range(1, len(surface_area_w) + 1)])
                        x_range=x_range/2
                        surface_plot_w_ax.plot(x_range, surface_area_w, color='g',
                                               label='Weighted surface area')

                        ax2_w.plot(x_range, P_list, color='b',
                                   label='P')

                        ax3_w.plot(x_range, PI_list, color='c',
                                   label='PI')

                    surface_plot_w_ax.legend(loc=2)

                    plt.tight_layout()

                    if tif:
                        if not fixed_scale:
                            plt.savefig(os.path.join(path, 'Surface_weigthed_all_'+type_+ex+'_'+str(min_rsq))+'.tif',dpi=300,format='tif')
                        else:
                            plt.savefig(os.path.join(path, 'Fixed_scale_surface_weigthed_all_'+type_+ex+'_'+str(min_rsq))+'.tif',dpi=300,format='tif')
                    else:
                        if not fixed_scale:
                            plt.savefig(os.path.join(path, 'Surface_weigthed_all_' + type_ + ex + '_' + str(min_rsq))+'.jpg')
                        else:
                            plt.savefig(os.path.join(path, 'Fixed_scale_surface_weigthed_all_' + type_ + ex + '_' + str(
                                min_rsq)+'.jpg'))

                    plt.close('all')


                    custom_lines=[]
                    s=0
                    for kkk in range(len(all_label)):
                        custom_lines.append(Line2D([0], [0], color=color_list[s], lw=4))
                        s+=1



