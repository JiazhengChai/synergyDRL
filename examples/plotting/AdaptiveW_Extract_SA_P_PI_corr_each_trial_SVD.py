import os
import csv
import argparse
import numpy as np
from scipy import integrate
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
from exp_variant_class import exp_variant
from sklearn.decomposition import TruncatedSVD

def gauss(x, mu, a = 1, sigma = 1/6):
    return a * np.exp(-(x - mu)**2 / (2*sigma**2))

def R2():
    return r'R^{{{e:d}}}'.format(e=int(2))

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

cmap = plt.cm.viridis
cmaplist = [cmap(i) for i in range(cmap.N)]
cmaplen=len(cmaplist)


plt.rcParams["figure.figsize"] = (10,8)

parser = argparse.ArgumentParser()

parser.add_argument('--tr', nargs='+', required=True)

parser.add_argument('--ee',type=str, nargs='+',choices=['E0','E0_TD3','E1','E1_TD3'])

parser.add_argument('--agentt',
                    type=str,choices=['HCheavy','HC','A','Antheavy','FC','Ctp','G','HC4dof','HC3dofb','HC3doff','HC5dof','HC2dof','VA','VA4dof','VA6dof','VA8dof'])

args = parser.parse_args()
agentt=args.agentt

svg = False
save = True
sortt = False
avg_plot = True
precheck = False
truncated = True
plot_r_sq = True
manual_rsq = True
standscale = True
named_label = False
energy_penalty = False
plot_norm_single = True


min_rsq = 0
num_epi = 10
recon_num = 8
LW_action = 4
ori_num_vec = 3
desired_length = 50  # 50#28#7
truncated_start = 200
ori_total_vec_rsq = 9
per_episode_max_length=1000

type_ = 'P'

if 'HC' in agentt and 'dof' not in agentt:
    total_vec = 6
    total_chk=30
    ori_final = 3000
    ori_begin = 100
    ori_step = 100
    x_speed_index=8
    desired_dist=500
    action_path = './experiments_results/collected_actions/trajectory_npy/actions_npy'
    reward_path = './experiments_results/collected_actions/trajectory_npy/reward_energy_dict'
    state_path = './experiments_results/collected_actions/trajectory_npy/states_npy'

elif 'FC' in agentt:
    total_vec = 12
    total_chk=30
    ori_final = 3000
    ori_begin = 100
    ori_step = 100
    x_speed_index=14
    desired_dist=500
    action_path = './experiments_results/collected_actions/trajectory_npy/actions_npy'
    reward_path = './experiments_results/collected_actions/trajectory_npy/reward_energy_dict'
    state_path = './experiments_results/collected_actions/trajectory_npy/states_npy'

elif agentt=='HC4dof':
    total_vec = 4
    total_chk=30
    ori_final = 3000
    ori_begin = 100
    ori_step = 100
    x_speed_index=6
    desired_dist=500
    action_path = './experiments_results/collected_actions/trajectory_npy/actions_npy'
    reward_path = './experiments_results/collected_actions/trajectory_npy/reward_energy_dict'
    state_path = './experiments_results/collected_actions/trajectory_npy/states_npy'

elif agentt=='HC2dof':
    total_vec = 2
    total_chk=30
    ori_final = 3000
    ori_begin = 100
    ori_step = 100
    x_speed_index=4
    desired_dist=500
    action_path = './experiments_results/collected_actions/trajectory_npy/actions_npy'
    reward_path = './experiments_results/collected_actions/trajectory_npy/reward_energy_dict'
    state_path = './experiments_results/collected_actions/trajectory_npy/states_npy'

elif agentt=='HC3doff':
    total_vec = 3
    total_chk=30
    ori_final = 3000
    ori_begin = 100
    ori_step = 100
    x_speed_index=5
    desired_dist=500
    action_path = './experiments_results/collected_actions/trajectory_npy/actions_npy'
    reward_path = './experiments_results/collected_actions/trajectory_npy/reward_energy_dict'
    state_path = './experiments_results/collected_actions/trajectory_npy/states_npy'

elif agentt=='HC3dofb':
    total_vec = 3
    total_chk=30
    ori_final = 3000
    ori_begin = 100
    ori_step = 100
    x_speed_index=5
    desired_dist=500
    action_path = './experiments_results/collected_actions/trajectory_npy/actions_npy'
    reward_path = './experiments_results/collected_actions/trajectory_npy/reward_energy_dict'
    state_path = './experiments_results/collected_actions/trajectory_npy/states_npy'

elif agentt=='HC5dof':
    total_vec = 5
    total_chk=30
    ori_final = 3000
    ori_begin = 100
    ori_step = 100
    x_speed_index=7
    desired_dist=500
    action_path = './experiments_results/collected_actions/trajectory_npy/actions_npy'
    reward_path = './experiments_results/collected_actions/trajectory_npy/reward_energy_dict'
    state_path = './experiments_results/collected_actions/trajectory_npy/states_npy'
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
    joint_list = ['shoulder','shoulder2', 'elbow','elbow2']
    action_path = './experiments_results/collected_actions/trajectory_npy/actions_npy'
    reward_path = './experiments_results/collected_actions/trajectory_npy/reward_energy_dict'
    state_path = './experiments_results/collected_actions/trajectory_npy/states_npy'
    agentt_folder = 'VA4dof'
    dll = 400
    truncated_start= 300
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

if 'E1' in args.ee:
    top_folder=agentt+'_E1'
else:
    top_folder=agentt

if not os.path.exists('experiments_results/Synergy/all_csv/raw_csv/'+top_folder):
    os.makedirs('experiments_results/Synergy/all_csv/raw_csv/'+top_folder, exist_ok=True)

if args.agentt=='A' and 'E0' in args.ee:
    args.tr=['']+args.tr

for tr in args.tr:
    for ee in args.ee:
        final = ori_final
        begin = ori_begin
        step = ori_step
        trial = tr
        subfolder=ee

        surface_csv = open(
            'experiments_results/Synergy/all_csv/raw_csv/' + top_folder  + '/' + agentt + '_' + ee +tr+'_all_surface.csv', 'w')

        writer = csv.writer(surface_csv, lineterminator='\n')

        writer.writerow(['Checkpoint', 'Surface Area','P', 'PI', 'E'])

        base= agentt + '_' + ee
        all_names=[]

        tmp=[]
        for cc in range(final, begin - step, -step):
            if cc==3000 and 'HC' in agentt:
                tmp.append(base + trial)
                try:
                    dummy = exp_variant(tmp[-1],action_path=action_path,reward_path=reward_path,state_path=state_path)
                except:
                    tmp.pop()
                    tmp.append(base + '_C' + str(cc) + trial)
            else:
                tmp.append(base + '_C' + str(cc) + trial)


        all_names.append(tmp)
        all_names.reverse()
        print(all_names)
        if precheck:
            top_tmp=[]
            for all_name in all_names:
                tmpp = []
                for n in all_name:
                    if action_path != None:
                        complete = exp_variant(n, action_path=action_path, reward_path=reward_path,
                                               state_path=state_path).check_complete_data()
                    else:
                        complete = exp_variant(n).check_complete_data()
                    if complete:
                        tmpp.append(n)
                top_tmp.append(tmpp)

            all_names=top_tmp



        rsq_all_list = []
        for ind_all_name,all_name in enumerate(all_names):
            agent = all_name[0] + '_spatiotemporal_evolution'

            folder_name=agent+'_Compare_'+type_+'_'+str(desired_length)
            if manual_rsq:
                folder_name=folder_name+'_manual_rsq'
            if standscale:
                folder_name = folder_name + '_SS'

            '''step = cmaplen // 30
            color_list=[]
            c=0
            for l in range(30):
                color_list.append(cmaplist[c])
                c+=step

            color_list=[color_list[0],color_list[-1]]'''
            all_label=[]
            exp_variant_list=[]
            for ind,n in enumerate(all_name):
                if action_path != None:
                    exp_variant_list.append(
                        exp_variant(n, action_path=action_path, reward_path=reward_path, state_path=state_path))

                else:
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

            r_sq_all_combare, r_sq_all_combare_ax = plt.subplots(1, 1)

            P_list=[]
            PI_list=[]
            E_list=[]
            SA_list=[]
            for n_ind,name in enumerate(all_name):

                exp_variant_obj=exp_variant_list[n_ind]

                name_list=name.split('_')
                current_checkpoint=0
                for nn in name_list:
                    if nn[0]=='C':
                        current_checkpoint=nn
                if current_checkpoint==0:
                    current_checkpoint='C'+str(ori_final)

                rsq_label = []

                X=np.load(exp_variant_obj.action_npy)
                state_ = np.load(exp_variant_obj.state_npy)


                mini = per_episode_max_length
                if X.shape == (num_epi,):
                    # print('a')
                    for i in range(num_epi):
                        amin = np.asarray(X[i]).shape[0]
                        if amin < mini:
                            mini = amin
                    print(mini)

                    tmp = np.expand_dims(np.asarray(X[0])[0:mini, :], 0)
                    for i in range(num_epi-1):
                        tmp = np.vstack((tmp, np.expand_dims(np.asarray(X[i + 1])[0:mini, :], 0)))
                    print(tmp.shape)
                    X = tmp

                    tmp2 = np.expand_dims(np.asarray(state_[0])[0:mini, :], 0)
                    for i in range(num_epi-1):
                        tmp2 = np.vstack((tmp2, np.expand_dims(np.asarray(state_[i + 1])[0:mini, :], 0)))
                    state_ = tmp2
                X=X[0:num_epi,:,:]
                state_ = state_[0:num_epi, :, :]
                distance = []
                if x_speed_index:
                    speed_record = state_[0, :, x_speed_index]
                    for i in range(len(speed_record)):
                        if i == 0:
                            distance.append(speed_record[0])
                        else:
                            distance.append(np.sum(speed_record[0:i]))

                    distance = np.asarray(distance)
                if truncated:
                    total_vec_rsq = ori_total_vec_rsq
                    if x_speed_index:
                        if mini == per_episode_max_length or mini>300:
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
                    else:
                        desired_length = dll

                    if mini == per_episode_max_length:
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

                rsq_single_list=[]
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

                if standscale:
                    mx = np.mean(X, axis=1)
                    for k in range(X.shape[1]):
                        X[:, k, :] = X[:, k, :] - mx

                X = reshape_into_spt_shape(X)

                for num_vec_to_keep_ in range(1,total_vec_rsq+1):

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

                    if manual_rsq:
                        Vm = np.mean(X, axis=0, keepdims=True)
                        resid = X - np.dot(Vm, np.ones((X.shape[1], 1)))
                        resid2 = X - X_prime
                        SST = np.linalg.norm(resid)
                        SSE = np.linalg.norm(resid2)
                        rsq = 1 - SSE / SST

                    else:
                        rsq = r2_score(X, X_prime)

                    rsq_single_list.append(rsq)

                rsq_label.append('Rsq_' + exp_variant_obj.name)

                rsq_all_list.append(rsq_single_list)

                surface_area=integrate.simps(rsq_single_list,range(1,total_vec_rsq+1))
                P=exp_variant_obj.eval('P')
                PI=exp_variant_obj.eval('PI')
                E = exp_variant_obj.eval('E')

                SA_list.append(surface_area)
                P_list.append(P)
                PI_list.append(PI)
                E_list.append(E)

                if np.isnan(surface_area):
                    surface_area=SA_list[n_ind-1]

                writer.writerow([current_checkpoint,surface_area,P,PI,E])


surface_csv.close()




