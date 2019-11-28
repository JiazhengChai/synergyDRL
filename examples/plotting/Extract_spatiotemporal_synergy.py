import numpy as np
from matplotlib import pyplot as plt
import os
from matplotlib.lines import Line2D
from exp_variant_class import exp_variant
from sklearn.decomposition import TruncatedSVD
import argparse

cmap = plt.cm.viridis
cmaplist = [cmap(i) for i in range(cmap.N)]
cmaplen = len(cmaplist)

def gauss(x, mu, a=1, sigma=1 / 6):
    return a * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

def R2():
    return r'R^{{{e:d}}}'.format(e=int(2))


color_list = ['b', 'r', 'g', 'c', 'm', 'y', 'k', '#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33',
              '#a65628', '#f781bf']

plt.rcParams["figure.figsize"] = (10, 8)
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['font.size'] = 15
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['font.family'] = 'Times New Roman'
LW = 3

parser = argparse.ArgumentParser()

parser.add_argument('--tr', nargs='+', required=True)

parser.add_argument('--ee', type=str, nargs='+', choices=['E0', 'E0_TD3', 'E1', 'E1_TD3'])

parser.add_argument('--agentt',
                    type=str, choices=['HCheavy', 'HC', 'A', 'Antheavy', 'FC', 'Ctp', 'G'])

parser.add_argument('--high_res',
                    type=bool, default=False)
parser.add_argument('--adaptive_windows',
                    type=bool, default=False)

args = parser.parse_args()


tif = False
ori_total_vec_rsq = 9
truncated_start = 200
dll = 7
type_ = 'PI'
num_synergy=3
save_figure = True

agentt = args.agentt
high_res = args.high_res
adaptive_windows = args.adaptive_windows
if 'HC' in agentt:
    total_vec = 6
    total_chk = 30
    ori_final = 3000
    ori_begin = 100
    ori_step = 100
    x_speed_index=8
    desired_dist=150
    joint_list = ['back thigh', 'back shin', 'back foot', 'front thigh', 'front shin',
                  'front foot']

elif 'FC' in agentt:
    plt.rcParams['figure.figsize'] = [15, 12]
    total_vec = 12
    total_chk = 30
    ori_final = 3000
    ori_begin = 100
    ori_step = 100
    desired_dist=500
    x_speed_index=14

    joint_list = ['lbt', 'lbs', 'lbf', 'lft',
                  'lfs',
                  'lff',
                  'rbt', 'rbs', 'rbf', 'rft',
                  'rfs',
                  'rff'
                  ]

top_folder = agentt + '_spatiotemporal_evolution'

for tr in args.tr:
    for ee in args.ee:
        final = ori_final
        begin = ori_begin
        step = ori_step
        trial = tr
        subfolder = ee

        base = agentt + '_' + ee
        all_names = []
        tmp = []
        for cc in range(final, begin - step, -step):
            if cc == 3000 and 'HC' in agentt:
                tmp.append(base + trial)
                try:
                    dummy = exp_variant(tmp[-1])
                except:
                    tmp.pop()
                    tmp.append(base + '_C' + str(cc) + trial)
            else:
                tmp.append(base + '_C' + str(cc) + trial)
        all_names.append(tmp)
        all_names.reverse()
        print(all_names)

        total_vec_rsq = ori_total_vec_rsq
        desired_length = dll

        rsq_all_list = []
        r_sq_all_TOP, r_sq_all_TOP_ax = plt.subplots(1, 1)
        for all_name in all_names:
            agent = all_name[0] + '_spatiotemporal_evolution'

            folder_name = agent +'_'+type_ +'_'+ str(desired_length)

            step = cmaplen // len(all_name)
            color_list = []
            c = 0
            for l in range(len(all_name)):
                color_list.append(cmaplist[c])
                c += step
            all_label = []
            exp_variant_list = []
            for ind, n in enumerate(all_name):
                exp_variant_list.append(exp_variant(n))
                all_label.append(exp_variant_list[ind].eval(type_))

            all_label = []
            for ind, n in enumerate(all_name):
                prefix_list = exp_variant_list[ind].name.split('_')
                for pp in prefix_list:
                    if 'C' in pp and 'H' not in pp:
                        prefix = pp
                        break
                all_label.append(prefix + ': ' + '{:.2f}'.format(exp_variant_list[ind].eval(type_), 2))


            plot_r_sq = True

            num_epi = 10

            r_sq_all_combare, r_sq_all_combare_ax = plt.subplots(1, 1)
            s = 0

            for n_ind, name in enumerate(all_name):

                exp_variant_obj = exp_variant_list[n_ind]
                exp_var_percentage = 0.95

                X = np.load(exp_variant_obj.action_npy)
                state_ = np.load(exp_variant_obj.state_npy)

                mini = 1000
                if X.shape == (10,):
                    for i in range(10):
                        amin = np.asarray(X[i]).shape[0]
                        if amin < mini:
                            mini = amin

                    tmp = np.expand_dims(np.asarray(X[0])[0:mini, :], 0)
                    for i in range(9):
                        tmp = np.vstack((tmp, np.expand_dims(np.asarray(X[i + 1])[0:mini, :], 0)))
                    X = tmp

                    tmp2 = np.expand_dims(np.asarray(state_[0])[0:mini, :], 0)
                    for i in range(9):
                        tmp2 = np.vstack((tmp2, np.expand_dims(np.asarray(state_[i + 1])[0:mini, :], 0)))
                    state_ = tmp2

                X = X[0:num_epi, :, :]
                state_ = state_[0:num_epi, :, :]

                total_vec_rsq = ori_total_vec_rsq

                distance = []
                speed_record = state_[0, :, x_speed_index]
                for i in range(len(speed_record)):
                    if i == 0:
                        distance.append(speed_record[0])
                    else:
                        distance.append(np.sum(speed_record[0:i]))

                distance = np.asarray(distance)

                if adaptive_windows:
                    if mini == 1000 or mini > 300:
                        current_dist = distance[truncated_start]
                        end_dist_index = truncated_start
                        tmp_dist = 0

                        while tmp_dist < desired_dist and end_dist_index < len(distance) - 1:
                            end_dist_index += 1
                            tmp_dist = distance[end_dist_index] - current_dist

                        remaining_index = end_dist_index - truncated_start
                        desired_length = remaining_index
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

                        if mini >= ori_total_vec_rsq:
                            total_vec_rsq = ori_total_vec_rsq
                        else:
                            total_vec_rsq = mini
                X = X_truncated

                ori_shape = X.shape
                X_unnorm = np.copy(X)


                def reshape_into_spt_shape(X_input):
                    assert len(X_input.shape) == 3
                    trial_num, sample_l, action_dim = X_input.shape

                    X_loc = np.expand_dims(X_input[0, :, 0], 0)

                    for i in range(1, action_dim, 1):
                        X_loc = np.concatenate((X_loc, np.expand_dims(X_input[0, :, i], 0)), axis=1)

                    for l in range(1, trial_num, 1):
                        X_inner = np.expand_dims(X_input[l, :, 0], 0)
                        for i in range(1, action_dim, 1):
                            X_inner = np.concatenate((X_inner, np.expand_dims(X_input[l, :, i], 0)), axis=1)

                        X_loc = np.concatenate((X_loc, X_inner), axis=0)

                    return X_loc


                def reshape_into_ori_shape(X_input):
                    assert len(X_input.shape) == 2
                    trial_num = X_input.shape[0]
                    sample_l = int(X_input.shape[1] / total_vec)
                    action_dim = total_vec

                    X_loc = np.expand_dims(X_input[0, 0:sample_l], 1)
                    for i in range(1, action_dim, 1):
                        X_loc = np.concatenate(
                            (X_loc, np.expand_dims(X_input[0, sample_l * i:sample_l * (i + 1)], 1)), axis=1)
                    X_loc = np.expand_dims(X_loc, 0)

                    for l in range(1, trial_num, 1):
                        X_inner = np.expand_dims(X_input[l, 0:sample_l], 1)
                        for i in range(1, action_dim, 1):
                            X_inner = np.concatenate(
                                (X_inner, np.expand_dims(X_input[l, sample_l * i:sample_l * (i + 1)], 1)),
                                axis=1)
                        X_inner = np.expand_dims(X_inner, 0)

                        X_loc = np.concatenate((X_loc, X_inner), axis=0)

                    assert X_loc.shape[0] == trial_num
                    assert X_loc.shape[1] == sample_l
                    assert X_loc.shape[2] == action_dim

                    return X_loc


                max_list = []
                max_ind = np.argmax(X[0, :, 0])
                max_list.append(max_ind)
                X_temp = np.concatenate(
                    (np.expand_dims(X[0, max_ind::, :], 0), np.expand_dims(X[0, 0:max_ind, :], 0)), axis=1)
                for l in range(1, X.shape[0], 1):
                    max_ind = np.argmax(X[l, :, 0])
                    max_list.append(max_ind)
                    X_temp = np.concatenate((X_temp, np.concatenate(
                        (np.expand_dims(X[l, max_ind::, :], 0), np.expand_dims(X[l, 0:max_ind, :], 0)),
                        axis=1)), axis=0)
                X = X_temp

                mx = np.mean(X, axis=1)

                for k in range(X.shape[1]):
                    X[:, k, :] = X[:, k, :] - mx

                X_mean = np.mean(X, axis=0)
                X_std = np.std(X, axis=0)

                a_list = []
                a_list_std = []
                for iterrr in range(total_vec):
                    a_list.append(X_mean[:, iterrr])
                    a_list_std.append(X_std[:, iterrr])

                if agentt == 'FC':
                    gg, ax = plt.subplots(total_vec, 1, figsize=(15, 30))
                else:
                    gg, ax = plt.subplots(total_vec, 1)

                for ii in range(len(a_list)):
                    if len(a_list[ii]) >= desired_length:
                        ax[ii].plot(range(desired_length), a_list[ii][0:desired_length], linewidth=LW)
                        ax[ii].fill_between(range(desired_length),
                                            a_list[ii] + a_list_std[ii],
                                            a_list[ii] - a_list_std[ii],
                                            alpha=0.2)
                    else:
                        ax[ii].plot(range(mini), a_list[ii][0:mini])
                        ax[ii].fill_between(range(mini),
                                            a_list[ii] + a_list_std[ii],
                                            a_list[ii] - a_list_std[ii],
                                            alpha=0.2)

                    if agentt == 'HC' or agentt == 'FC':
                        ax[ii].set_ylabel(joint_list[ii])
                    else:
                        ax[ii].set_ylabel('Joint ' + str(ii + 1))
                    ax[ii].get_xaxis().set_visible(False)
                    if ii == len(a_list) - 1:
                        ax[ii].set_xlabel('Time steps')
                        ax[ii].get_xaxis().set_visible(True)

                X = reshape_into_spt_shape(X)
                s_by_a = X
                a_by_s = X.T

                pca = TruncatedSVD(n_components=num_synergy)
                pca.fit(X)
                eig_vecs = pca.components_
                eig_vals = pca.singular_values_
                eig_pairs = [(eig_vals[i], eig_vecs[i, :]) for i in range(len(eig_vals))]
                percentage = sum(pca.explained_variance_ratio_)

                num_features = X.shape[1]
                proj_mat = eig_pairs[0][1].reshape(num_features, 1)
                for eig_vec_idx in range(1, num_synergy):
                    proj_mat = np.hstack((proj_mat, eig_pairs[eig_vec_idx][1].reshape(num_features, 1)))

                W = proj_mat
                C = X.dot(W)
                X_prime = C.dot(W.T)

                X_prime_norm = np.copy(X_prime)
                X_prime_norm = reshape_into_ori_shape(X_prime_norm)

                X_prime = reshape_into_ori_shape(X_prime)

                Xp_mean = np.mean(X_prime, axis=0)
                Xp_std = np.std(X_prime, axis=0)

                a_list = []
                a_list_std = []

                for iterrr in range(total_vec):
                    a_list.append(Xp_mean[:, iterrr])
                    a_list_std.append(Xp_std[:, iterrr])


                for ii in range(len(a_list)):
                    ax[ii].axhline(0, color='black', alpha=0.5, linestyle='--')

                    if len(a_list[ii]) >= desired_length:
                        ax[ii].plot(range(desired_length), a_list[ii][0:desired_length], color='r')

                        ax[ii].fill_between(range(desired_length),
                                            a_list[ii] + a_list_std[ii],
                                            a_list[ii] - a_list_std[ii],
                                            alpha=0.2, color='r')
                    else:
                        ax[ii].plot(range(mini), a_list[ii][0:mini], color='r')

                        ax[ii].fill_between(range(desired_length),
                                            a_list[ii] + a_list_std[ii],
                                            a_list[ii] - a_list_std[ii],
                                            alpha=0.2, color='r')

                    ax[ii].set_ylim([-1.2, 1.2])
                if save_figure == False:
                    plt.show()
                else:
                    path = 'experiments_results/Synergy/pattern/' + top_folder + '/' + subfolder + '/' + folder_name + '/' + 'Reconstructions'
                    os.makedirs(path, exist_ok=True)
                    gg.tight_layout()
                    if tif:
                        gg.savefig(os.path.join(path, 'Reconstructions' + exp_variant_obj.name), dpi=300,
                                   format='tif')
                    else:
                        if not high_res:
                            gg.savefig(os.path.join(path, 'Reconstructions' + exp_variant_obj.name + '.jpg'))
                        else:
                            gg.savefig(os.path.join(path, 'Reconstructions' + exp_variant_obj.name + '.png'),
                                       dpi=300)

                gg, ax = plt.subplots(num_synergy, 1)

                c_list = []
                for iterr in range(num_synergy):
                    c_list.append(C[0:X.shape[0], iterr])
                c_list = np.asarray(c_list)
                for ii in range(num_synergy):
                    ax[ii].bar(range(c_list.shape[1]), c_list[ii, :], 0.8)

                    if ii == len(c_list) - 1:
                        ax[ii].set_xlabel('Number of trials')

                    if 'HC' in agentt:
                        ax[ii].set_ylim([-5, 5])
                    elif 'FC' in agentt:
                        ax[ii].set_ylim([-3, 6.5])

                if save_figure == False:
                    plt.show()
                else:
                    path = 'experiments_results/Synergy/pattern/' + top_folder + '/' + subfolder + '/' + folder_name + '/' + 'C_matrix'

                    os.makedirs(path, exist_ok=True)
                    gg.tight_layout()
                    if tif:
                        gg.savefig(os.path.join(path, 'C-matrix-' + exp_variant_obj.name), dpi=300,
                                   format='tif')
                    else:
                        if not high_res:
                            gg.savefig(os.path.join(path, 'C-matrix-' + exp_variant_obj.name + '.jpg'),
                                       format='jpg')
                        else:
                            gg.savefig(os.path.join(path, 'C-matrix-' + exp_variant_obj.name + '.png'),
                                       format='png', dpi=300)

                gb, bx = plt.subplots(total_vec, num_synergy)
                sample_length = int(W.shape[0] / total_vec)
                for ii in range(num_synergy):
                    for j in range(num_synergy):
                        for k in range(total_vec):
                            bx[k, j].plot(W[sample_length * k:sample_length * (k + 1), j],
                                          linewidth=LW)
                            bx[k, j].axhline(0, color='black', alpha=0.5, linestyle='--')

                            bx[k, j].get_xaxis().set_visible(False)

                            if 'HC' in agentt:
                                bx[k, j].set_ylim([-0.6, 0.6])
                            elif 'FC' in agentt:
                                bx[k, j].set_ylim([-0.4, 0.4])

                            if k == 0:
                                bx[k, j].set_title('W' + '$_{}$'.format(j + 1))

                            if k == total_vec - 1:
                                bx[k, j].get_xaxis().set_visible(True)
                                bx[k, j].set_xlabel('Time steps')
                            if j == 0:
                                bx[k, j].set_ylabel(joint_list[k])

                if save_figure == False:
                    plt.show()
                else:
                    path = 'experiments_results/Synergy/pattern/' + top_folder + '/' + subfolder + '/' + folder_name + '/' + 'W_matrix'
                    os.makedirs(path, exist_ok=True)
                    gb.tight_layout()
                    if tif:
                        gb.savefig(path + '/Line_PCA_components_' + exp_variant_obj.name, dpi=300, format='tif')
                    else:
                        gb.savefig(path + '/Line_PCA_components_' + exp_variant_obj.name + '.jpg')


                if plot_r_sq:
                    rsq_label = []

                    X = np.load(exp_variant_obj.action_npy)

                    mini = 1000
                    if X.shape == (10,):
                        for i in range(10):
                            amin = np.asarray(X[i]).shape[0]
                            if amin < mini:
                                mini = amin

                        tmp = np.expand_dims(np.asarray(X[0])[0:mini, :], 0)
                        for i in range(9):
                            tmp = np.vstack((tmp, np.expand_dims(np.asarray(X[i + 1])[0:mini, :], 0)))
                        print(tmp.shape)
                        X = tmp
                    X = X[0:num_epi, :, :]

                    total_vec_rsq = ori_total_vec_rsq
                    if mini == 1000:
                        X_truncated = X[:, truncated_start:truncated_start + desired_length, :]
                    else:
                        if mini >= (truncated_start + desired_length):
                            X_truncated = X[:, truncated_start:truncated_start + desired_length, :]

                        elif mini - desired_length >= 0:

                            X_truncated = X[:, mini - desired_length:mini, :]

                        else:
                            X_truncated = X[:, 0:mini, :]

                            if mini >= ori_total_vec_rsq:
                                total_vec_rsq = ori_total_vec_rsq
                            else:
                                total_vec_rsq = mini

                    X = X_truncated

                    rsq_single_list = []

                    max_list = []
                    max_ind = np.argmax(X[0, :, 0])
                    max_list.append(max_ind)
                    X_temp = np.concatenate(
                        (np.expand_dims(X[0, max_ind::, :], 0), np.expand_dims(X[0, 0:max_ind, :], 0)), axis=1)
                    for l in range(1, X.shape[0], 1):
                        max_ind = np.argmax(X[l, :, 0])
                        max_list.append(max_ind)
                        X_temp = np.concatenate((X_temp, np.concatenate(
                            (np.expand_dims(X[l, max_ind::, :], 0), np.expand_dims(X[l, 0:max_ind, :], 0)),
                            axis=1)), axis=0)
                    X = X_temp

                    mx = np.mean(X, axis=1)

                    for k in range(X.shape[1]):
                        X[:, k, :] = X[:, k, :] - mx

                    X = reshape_into_spt_shape(X)

                    for num_vec_to_keep_ in range(1, total_vec_rsq + 1):
                        pca = TruncatedSVD(n_components=num_vec_to_keep_)
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

                        Vm = np.mean(X)
                        resid = X - Vm
                        resid2 = X - X_prime
                        SSE = np.sum(np.square(resid2))
                        SST = np.sum(np.square(resid))
                        rsq = 1 - SSE / SST

                        rsq_single_list.append(rsq)

                    rsq_label.append('Rsq_' + exp_variant_obj.name)

                    rsq_all_list.append(rsq_single_list)
                    r_sq_single, r_sq_single_ax = plt.subplots(1, 1)

                    r_sq_single_ax.plot(range(1, total_vec_rsq + 1), rsq_single_list)

                    r_sq_single_ax.set_ylabel(r"${0:s}$".format(R2()))
                    r_sq_single_ax.set_xlabel('Number of PCA components')

                    path = 'experiments_results/Synergy/pattern/' + top_folder + '/' + subfolder + '/' + folder_name + '/' + 'Rsq_single'

                    os.makedirs(path, exist_ok=True)
                    r_sq_single.tight_layout()

                    if tif:
                        r_sq_single.savefig(os.path.join(path, 'Rsq_' + exp_variant_obj.name), dpi=300,
                                            format='tif')
                    else:
                        if not high_res:
                            r_sq_single.savefig(os.path.join(path,
                                                             'Rsq_' + exp_variant_obj.name + '.jpg'))
                        else:
                            r_sq_single.savefig(os.path.join(path, 'Rsq_' + exp_variant_obj.name + '.png'),
                                                dpi=300)

                    r_sq_all_combare_ax.plot(range(1, total_vec_rsq + 1), rsq_single_list, color=color_list[s])

                    r_sq_all_combare_ax.set_ylim([0, 1])

                    s += 1

                    if n_ind == 0:
                        r_sq_all_combare_ax.set_ylabel(r"${0:s}$".format(R2()))
                        r_sq_all_combare_ax.set_xlabel('Number of PCA components')

                    plt.close('all')
            custom_lines = []

            s = 0
            for kkk in range(len(all_label)):
                custom_lines.append(Line2D([0], [0], color=color_list[s], lw=4))
                s += 1

            path = 'experiments_results/Synergy/pattern/' + top_folder + '/' + subfolder + '/' + folder_name
            os.makedirs(path, exist_ok=True)
            ex = '_named'

            r_sq_all_combare.tight_layout()
            r_sq_all_combare_ax.set_ylim([0, 1.05])

            if tif:
                r_sq_all_combare.savefig(os.path.join(path, 'Rsq_all_' + type_ + ex + '_' + str(0)),
                                         dpi=300, format='tif')
            else:
                if not high_res:
                    r_sq_all_combare.savefig(os.path.join(path, 'Rsq_all_' + type_ + ex + '.jpg'))
                else:
                    r_sq_all_combare.savefig(os.path.join(path, 'Rsq_all_' + type_ + ex + '.png'),
                                             dpi=300)





