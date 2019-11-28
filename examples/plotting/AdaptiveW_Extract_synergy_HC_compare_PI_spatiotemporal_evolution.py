import numpy as np
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt
import os
from matplotlib.lines import Line2D
from exp_variant_class import exp_variant
from sklearn.decomposition import PCA
import argparse
cmap = plt.cm.viridis
cmaplist = [cmap(i) for i in range(cmap.N)]
cmaplen=len(cmaplist)

def gauss(x, mu, a = 1, sigma = 1/6):
    return a * np.exp(-(x - mu)**2 / (2*sigma**2))

def R2():

    return r'R^{{{e:d}}}'.format(e=int(2))
color_list=['b','r','g','c','m','y','k','#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf']

plt.rcParams["figure.figsize"] = (10,8)

parser = argparse.ArgumentParser()


parser.add_argument('--tr', nargs='+', required=True)

parser.add_argument('--ee',type=str, nargs='+',choices=['E0','E0_TD3','E1','E1_TD3'])

parser.add_argument('--agentt',
                    type=str,choices=['HCheavy','HC','A','Antheavy','FC','Ctp','G'])

args = parser.parse_args()

energy_penalty=False
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
    ori_begin = 100#
    ori_step = 100
    x_speed_index=8
    desired_dist=500

elif 'FC' in agentt:
    total_vec = 12
    total_chk=30
    ori_final = 3000
    ori_begin = 100#
    ori_step = 100
    x_speed_index=14
    desired_dist=500

top_folder=agentt+'_spatiotemporal_evolution'
if energy_penalty:
    top_folder=top_folder+'_EP'

if args.agentt=='A' and 'E0' in args.ee:
    args.tr=['']+args.tr

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
            r_sq_all_TOP, r_sq_all_TOP_ax = plt.subplots(1, 1)
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
                    #color_list.append(cmaplist[c])
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
                              (True, '4_components_truncated', 4)
                                ]
                plot_r_sq=True
                save=True

                num_epi=10

                r_sq_all_combare, r_sq_all_combare_ax = plt.subplots(1, 1)
                s = 0
                if energy_penalty:
                    energy_all_list = []
                    best_perf_energy_ind_tuple=(0,0,0)

                    for n_ind, name in enumerate(all_name):
                        exp_variant_obj = exp_variant_list[n_ind]
                        current_energy=exp_variant_obj.eval('E')
                        current_P = exp_variant_obj.eval('P')
                        energy_all_list.append(current_energy)

                        if current_P>best_perf_energy_ind_tuple[0]:
                            best_perf_energy_ind_tuple=(current_P,current_energy,n_ind)

                    energy_all_list = np.asarray(energy_all_list)
                    E_max = np.max(energy_all_list)
                    energy_all_list = energy_all_list / E_max
                    best_perf_energy_ind_tuple = (best_perf_energy_ind_tuple[0], energy_all_list[best_perf_energy_ind_tuple[2]], best_perf_energy_ind_tuple[2])
                    energy_all_list=gauss(energy_all_list,best_perf_energy_ind_tuple[1])

                for n_ind,name in enumerate(all_name):

                    exp_variant_obj=exp_variant_list[n_ind]

                    for comb in combinations:
                        truncated,version,ori_num_vec=comb

                        exp_var_percentage = 0.95

                        num_vec_to_keep=ori_num_vec
                        X=np.load(exp_variant_obj.action_npy)

                        state_ = np.load(exp_variant_obj.state_npy)


                        mini = 1000
                        if X.shape == (10,):
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

                            if mini == 1000 or mini>300:
                                current_dist = distance[truncated_start]
                                end_dist_index = truncated_start
                                tmp_dist = 0

                                while tmp_dist < desired_dist and end_dist_index < len(distance) - 1:
                                    end_dist_index += 1
                                    tmp_dist = distance[end_dist_index] - current_dist

                                remaining_index = end_dist_index - truncated_start
                                desired_length=remaining_index
                            elif mini - desired_length >= 0:
                                remaining_index=desired_length
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

                        ori_shape = X.shape

                        a_list=[]

                        if standscale:
                            ori_shape = X.shape
                            X = X.reshape([X.shape[0], -1])
                            mx = np.mean(X, axis=0)

                            X = X - mx

                            X = X.reshape(ori_shape)

                        for iterrr in range(total_vec):
                            a_list.append(X[recon_num,:,iterrr])

                        gg, ax = plt.subplots(total_vec, 1)
                        for ii in range(len(a_list)):
                            if len(a_list[ii])>=desired_length:
                                ax[ii].plot(range(desired_length), a_list[ii][0:desired_length])
                            else:
                                ax[ii].plot(range(mini), a_list[ii][0:mini])

                            ax[ii].set_ylabel('Joint '+str(ii+1))
                            ax[ii].get_xaxis().set_visible(False)
                            if ii==len(a_list)-1:
                                ax[ii].set_xlabel('Timesteps')
                                ax[ii].get_xaxis().set_visible(True)

                        X = X.reshape([X.shape[0], -1])


                        s_by_a=X
                        a_by_s=X.T
                        if manual_pca:
                            cov_mat = np.cov(X.T)

                            eig_vals, eig_vecs = np.linalg.eig(cov_mat)

                            eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]

                            eig_pairs.sort(key=lambda x: x[0], reverse=True)

                            tot = sum(eig_vals)
                            var_exp = [(i / tot) for i in sorted(eig_vals, reverse=True)]
                            cum_var_exp = np.cumsum(var_exp)

                            if num_vec_to_keep == 0:
                                ori_num_vec = num_vec_to_keep
                                for index, percentage in enumerate(cum_var_exp):
                                    if percentage > exp_var_percentage:
                                        num_vec_to_keep = index + 1
                                        break
                            else:
                                percentage = cum_var_exp[num_vec_to_keep - 1]
                        else:

                            pca=PCA(n_components=num_vec_to_keep)

                            pca.fit(X)
                            eig_vecs=pca.components_
                            eig_vals=pca.singular_values_

                            eig_pairs = [(eig_vals[i], eig_vecs[i, :]) for i in range(len(eig_vals))]

                            percentage=sum(pca.explained_variance_ratio_)


                        print(num_vec_to_keep)
                        print(percentage)

                        num_features = X.shape[1]
                        proj_mat = eig_pairs[0][1].reshape(num_features,1)
                        for eig_vec_idx in range(1, num_vec_to_keep):
                          proj_mat = np.hstack((proj_mat, eig_pairs[eig_vec_idx][1].reshape(num_features,1)))

                        W=proj_mat

                        C = X.dot(W)

                        X_prime=C.dot(W.T)

                        X_prime=X_prime.reshape(ori_shape)

                        C = C.reshape([num_vec_to_keep,-1])

                        a_list=[]
                        for iterr in range(total_vec):
                            a_list.append(X_prime[recon_num,:,iterr])

                        for ii in range(len(a_list)):

                            if len(a_list[ii])>=desired_length:
                                ax[ii].plot(range(desired_length), a_list[ii][0:desired_length], color='r')
                            else:
                                ax[ii].plot(range(mini), a_list[ii][0:mini], color='r')

                        if save==False:
                            plt.show()
                        else:
                            path='experiments_results/Synergy/synergy_development_'+agentt+'/'+top_folder+'/'+subfolder+'/'+folder_name+'/'+exp_variant_obj.name+'_synergy'+'/Synergy_plot_'+version
                            os.makedirs(path, exist_ok=True)
                            if tif:
                                gg.savefig(os.path.join(path, 'Reconstructions' + exp_variant_obj.name),dpi=300,format='tif')
                            else:
                                gg.savefig(os.path.join(path, 'Reconstructions' + exp_variant_obj.name+'.jpg'))

                        gg, ax = plt.subplots(num_vec_to_keep, 1)


                        c_list = []
                        for iterr in range(num_vec_to_keep):
                            c_list.append(C[iterr, 0:X.shape[0]])

                        for ii in range(len(c_list)):
                            ax[ii].bar(range(X.shape[0]), c_list[ii][0:X.shape[0]], 0.8)
                            ax[ii].set_ylabel('C '+str(ii))
                            if ii==len(c_list)-1:
                                ax[ii].set_xlabel('Number of trials')

                        if save == False:
                            plt.show()
                        else:
                            path = 'experiments_results/Synergy/synergy_development_'+agentt+'/'+top_folder+'/'+subfolder+'/'+folder_name+'/'+ exp_variant_obj.name+'_synergy' + '/Synergy_plot_' + version
                            os.makedirs(path, exist_ok=True)
                            gg.savefig(os.path.join(path, 'C-matrix-' + exp_variant_obj.name ),dpi=300,format='tif')

                        gg, ax = plt.subplots(num_vec_to_keep, 1)
                        gb, bx = plt.subplots(num_vec_to_keep, 1)


                        for ii in range(num_vec_to_keep):
                            ax[ii].bar(range(W[:,ii].shape[0]), W[:,ii], 0.8)
                            bx[ii].plot(range(W[:, ii].shape[0]), W[:, ii], color='r')

                            ax[ii].set_ylabel('W-'+str(ii))
                            ax[ii].get_xaxis().set_visible(False)

                            bx[ii].set_ylabel('W-' + str(ii))
                            bx[ii].get_xaxis().set_visible(False)

                            if ii==num_vec_to_keep-1:
                                ax[ii].get_xaxis().set_visible(True)
                                bx[ii].get_xaxis().set_visible(True)

                        if save==False:
                            plt.show()
                        else:
                            path = 'experiments_results/Synergy/synergy_development_'+agentt+'/'+top_folder+'/'+subfolder+'/'+folder_name+'/'+  exp_variant_obj.name+'_synergy' + '/PCA_components_'+version
                            os.makedirs(path, exist_ok=True)
                            if tif:
                                gg.savefig(path+'/PCA_components_' + exp_variant_obj.name,dpi=300,format='tif')
                                gb.savefig(path+'/Line_PCA_components_' + exp_variant_obj.name,dpi=300,format='tif')
                            else:
                                gg.savefig(path + '/PCA_components_' + exp_variant_obj.name+'.jpg')
                                gb.savefig(path + '/Line_PCA_components_' + exp_variant_obj.name+'.jpg')

                    if plot_r_sq:


                        rsq_label = []

                        X=np.load(exp_variant_obj.action_npy)

                        state_ = np.load(exp_variant_obj.state_npy)

                        mini = 1000
                        if X.shape == (10,):

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
                            if mini == 1000 or mini>300:
                                current_dist = distance[truncated_start]
                                end_dist_index = truncated_start
                                tmp_dist = 0

                                while tmp_dist < desired_dist and end_dist_index < len(distance) - 1:
                                    end_dist_index += 1
                                    tmp_dist = distance[end_dist_index] - current_dist

                                remaining_index = end_dist_index - truncated_start
                                desired_length = remaining_index

                            elif mini - desired_length >= 0:
                                remaining_index=desired_length
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

                                eig_vals, eig_vecs = np.linalg.eig(cov_mat)

                                eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]

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
                        r_sq_single, r_sq_single_ax = plt.subplots(1, 1)


                        r_sq_single_ax.plot(range(1,total_vec_rsq+1),rsq_single_list)

                        r_sq_single_ax.set_ylabel(r"${0:s}$".format(R2()))
                        r_sq_single_ax.set_xlabel('Number of PCA components')


                        path = 'experiments_results/Synergy/synergy_development_'+agentt+'/'+top_folder+'/'+subfolder+'/'+folder_name+'/'+ exp_variant_obj.name+'_synergy' + '/Rsq'
                        os.makedirs(path, exist_ok=True)
                        if tif:
                            r_sq_single.savefig(os.path.join(path, 'Rsq_' + exp_variant_obj.name),dpi=300,format='tif')
                        else:
                            r_sq_single.savefig(os.path.join(path, 'Rsq_' + exp_variant_obj.name+'.jpg'))


                        if energy_penalty:
                            rsq_single_list=np.asarray(rsq_single_list)
                            A=0.5
                            B=0.5
                            rsq_single_list=(A*rsq_single_list+B*energy_all_list[n_ind])/(A+B)
                            r_sq_all_combare_ax.plot(range(1,total_vec_rsq+1),rsq_single_list,color=color_list[s])

                        else:
                            r_sq_all_combare_ax.plot(range(1,total_vec_rsq+1),rsq_single_list,color=color_list[s])

                        r_sq_all_combare_ax.set_ylim([min_rsq, 1])

                        s += 1

                        if n_ind==0:
                            r_sq_all_combare_ax.set_ylabel(r"${0:s}$".format(R2()))
                            r_sq_all_combare_ax.set_xlabel('Number of PCA components')
                        plt.close('all')
                custom_lines=[]

                s=0
                for kkk in range(len(all_label)):
                    custom_lines.append(Line2D([0], [0], color=color_list[s], lw=4))
                    s+=1


                path = 'experiments_results/Synergy/synergy_development_'+agentt+'/'+top_folder+'/'+subfolder+'/'+folder_name
                os.makedirs(path, exist_ok=True)
                ex=''
                if named_label:
                    ex='_named'

                r_sq_all_combare.tight_layout()
                r_sq_all_combare_ax.set_ylim([0, 1.05])
                if energy_penalty:
                    if tif:
                        r_sq_all_combare.savefig(os.path.join(path, 'Rsq_all_'+type_+ex+'_'+str(min_rsq)+'_EP'),dpi=300,format='tif')
                    else:
                        r_sq_all_combare.savefig(os.path.join(path, 'Rsq_all_'+type_+ex+'_'+str(min_rsq)+'_EP.jpg'))

                else:
                    if tif:
                        r_sq_all_combare.savefig(os.path.join(path, 'Rsq_all_'+type_+ex+'_'+str(min_rsq)),dpi=300,format='tif')
                    else:
                        r_sq_all_combare.savefig(os.path.join(path, 'Rsq_all_'+type_+ex+'_'+str(min_rsq)+'.jpg'))





