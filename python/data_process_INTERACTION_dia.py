import numpy as np
import glob
import os
import random
import pandas as pd
from scipy import spatial
import pickle
import zipfile

# Please change this to your need
data_root = '../dataset/INTERACTION/'
scenario_name = 'DR_CHN_Merging_ZS'
track_file_id = 0

test_ratio = 0.2  # test data proportion
random_seed = 0  # train test split ramdom seed

frame_rate = 5  # choose 1/2/5/10 fps
history_frames = 15  # 3 second * 5 frame/second
future_frames = 5  # 1 second * 5 frame/second
total_frames = history_frames + future_frames
max_num_object = 100  # maximum number of observed objects is 100
max_num_dia = 100  # maximum number of observed objects is 100

neighbor_distance = 30  # meter

# INTERACTION dataset format:
# object: track_id, frame_id, timestamp_ms, agent_type, x, y, vx, vy, psi_rad, length, width
# dia: frame_id, object_id, dia_id, Xs_x, Xs_y, Xs_vx, Xs_vy, Xs_theta, Xe_x, Xe_y, Xe_vx, Xe_vy, Xe_theta, Xf_p1_x, Xf_p1_y, Xf_p2_x, Xf_p2_y, Xf_p3_x, Xf_p3_y, Xf_p4_x, Xf_p4_y, Xf_p5_x, Xf_p5_y, Xf_p6_x, Xf_p6_y, Xf_length, Xf_theta
object_feature_dimension = 11 + 2
total_feature_dimension = 27 + 2  # we add mark "1: object / 0: dia" to the front of each row and mark "1" to the end to indicate that this row exists

agent_type = {'car': 1, 'truck': 2, 'bus': 3, 'motorcycle': 4,
              'bicycle': 5}  # Only TC data include different agent type, DR data are all cars


def train_test_split(data_set):
    '''
    randomly sample test set from data set
    return test set indices
        data_set: source data set
        test_ratio: proportion of test data range from (0,1)
        random_state: random seed
    '''
    random.seed(random_seed)
    data_size = data_set['frame_id'].max() + 1
    test_size = int(data_size * test_ratio) + 1
    test_cnt = 0
    sample_frames = int(total_frames * (10 / frame_rate))
    rand_max = data_size - sample_frames
    is_test = np.zeros(data_size)

    while test_cnt < test_size:
        sample_start = random.randint(1, rand_max)
        sample_end = sample_start + sample_frames
        for i in range(sample_start, sample_end):
            if is_test[i] == 0:
                is_test[i] = 1
                test_cnt = test_cnt + 1

    return is_test


def get_frame_instance_dict(pra_file_path, pra_is_train=True):
    '''
	Read raw data from files and return a dictionary:
		{frame_id:
			{object_id:
				# 11 features
				[object_id, frame_id, timestamp_ms, agent_type, x, y, vx, vy, psi_rad, length, width]
			}
		}
		object_type: agent_type{ car: 1,  trunk: 2}
	'''
    with open(pra_file_path, 'r') as reader:
        reader.readline()
        content = np.array([x.strip().split(',') for x in reader.readlines()]).astype(str)
        now_dict = {}
        for row in content:
            row[3] = agent_type[row[3]]
        content = content.astype(float)
        frame_interval = 10 / frame_rate
        if frame_interval > 1:
            for row in content:
                if pra_is_train and is_test[int(row[1])] == 1:
                    continue
                if not pra_is_train and is_test[int(row[1])] == 0:
                    continue
                if (row[2] / 100) % frame_interval == 1:  # resample
                    row[1] = (row[2] / 100) // frame_interval + 1  # reindex frame
                    n_dict = now_dict.get(row[1], {})
                    n_dict[row[0]] = row
                    now_dict[row[1]] = n_dict

        else:
            for row in content:
                if pra_is_train and is_test[int(row[2])] == 1:
                    continue
                if not pra_is_train and is_test[int(row[2])] == 0:
                    continue
                n_dict = now_dict.get(row[1], {})
                n_dict[row[0]] = row
                now_dict[row[1]] = n_dict

    return now_dict


def get_dia_dict(pra_file_path, pra_is_train=True):
    '''
    	Read raw data from files and return a dictionary:
    		{frame_id:
    			{object_id:
    			    {dia_id:
                        # 27 features
                        frame_id, object_id, dia_id, Xs_x, Xs_y, Xs_vx, Xs_vy, Xs_theta, Xe_x, Xe_y, Xe_vx, Xe_vy, Xe_theta, Xf_p1_x, Xf_p1_y, Xf_p2_x, Xf_p2_y, Xf_p3_x, Xf_p3_y, Xf_p4_x, Xf_p4_y, Xf_p5_x, Xf_p5_y, Xf_p6_x, Xf_p6_y, Xf_length, Xf_theta, distance
    			    }
    			}
    		}
    	'''
    if not os.path.exists(pra_file_path[:-3]+'csv'):
        z = zipfile.ZipFile(pra_file_path, 'r')
        z.extractall(pra_file_path[:-26])
        z.close()
    with open(pra_file_path[:-3]+'csv', 'r') as reader:
        content = np.array([x.strip().split(',') for x in reader.readlines()]).astype(float)
        now_dict = {}
        frame_interval = 10 / frame_rate
        if frame_interval > 1:
            for row in content:
                if pra_is_train and is_test[int(row[0])] == 1:
                    continue
                if not pra_is_train and is_test[int(row[0])] == 0:
                    continue
                if row[0] % frame_interval == 1:  # resample
                    row[0] = row[0] // frame_interval + 1  # reindex frame
                    n_dict = now_dict.get(row[0], {})
                    dia_dict = n_dict.get(row[1], {})
                    dia_dict[row[2]] = row
                    n_dict[row[1]] = dia_dict
                    now_dict[row[0]] = n_dict

        else:
            for row in content:
                if pra_is_train and is_test[int(row[0])] == 1:
                    continue
                if not pra_is_train and is_test[int(row[0])] == 0:
                    continue
                n_dict = now_dict.get(row[0], {})
                dia_dict = n_dict.get(row[1], {})
                dia_dict[row[2]] = row
                n_dict[row[1]] = dia_dict
                now_dict[row[0]] = n_dict

    return now_dict


def process_data(obj_dict, dia_dict, pra_start_ind, pra_end_ind, pra_observed_last):
    visible_object_id_list = list(obj_dict[pra_observed_last].keys())  # object_id appears at the last observed frame
    num_visible_object = len(visible_object_id_list)  # number of current observed objects

    # compute the mean values of x and y for zero-centralization.
    visible_object_value = np.array(list(obj_dict[pra_observed_last].values()))
    xy = visible_object_value[:, 4:6].astype(float)
    mean_xy = np.zeros_like(visible_object_value[0], dtype=float)
    m_xy = np.mean(xy, axis=0)
    mean_xy[4:6] = m_xy

    # compute distance between any pair of two objects
    dist_xy = spatial.distance.cdist(xy, xy)
    # if their distance is less than $neighbor_distance, we regard them are neighbors.
    neighbor_matrix = np.zeros((max_num_object, max_num_object + max_num_dia))
    neighbor_matrix[:num_visible_object, :num_visible_object] = (dist_xy < neighbor_distance).astype(int)

    # get dia features from data
    now_all_dia_id_list = list()
    visible_dia_id_list = list()
    now_dia_feature_dict = dict()

    for obj_id in dia_dict[pra_observed_last]:
        for dia_id in dia_dict[pra_observed_last][obj_id]:
            if dia_id not in visible_dia_id_list:
                visible_dia_id_list.append(dia_id)

    mean_xy_2 = np.zeros(total_feature_dimension - 2)
    mean_xy_2[3:5] = m_xy
    mean_xy_2[8:10] = m_xy
    mean_xy_2[13:25] = np.tile(m_xy, 6)
    for x in range(pra_start_ind, pra_end_ind):
        for obj_id in dia_dict[x]:
            for dia_id in dia_dict[x][obj_id]:
                if dia_id not in now_all_dia_id_list:
                    now_all_dia_id_list.append(dia_id)
                    if dia_id in visible_dia_id_list:
                        now_dia_feature_dict[dia_id] = [0] + list(dia_dict[x][obj_id][dia_id][:-1] - mean_xy_2) + [1]
                    else:
                        now_dia_feature_dict[dia_id] = [0] + list(dia_dict[x][obj_id][dia_id][:-1] - mean_xy_2) + [0]

    # if distance between object and dia is less than $neighbor_distance, we regard them are neighbors.
    for obj_id in dia_dict[pra_observed_last].keys():
        for dia_id in dia_dict[pra_observed_last][obj_id]:
            if dia_dict[pra_observed_last][obj_id][dia_id][-1] < neighbor_distance:
                neighbor_matrix[visible_object_id_list.index(obj_id)][num_visible_object + visible_dia_id_list.index(dia_id)] = 1


    now_all_object_id = set([val for x in range(pra_start_ind, pra_end_ind) for val in obj_dict[x].keys()])
    non_visible_object_id_list = list(now_all_object_id - set(visible_object_id_list))
    num_non_visible_object = len(non_visible_object_id_list)
    non_visible_dia_id_list = list(set(now_all_dia_id_list) - set(visible_dia_id_list))
    num_visible_dia = len(visible_dia_id_list)
    num_non_visible_dia = len(non_visible_dia_id_list)

    # for all history frames(15) or future frames(5), we only choose the objects listed in visible_object_id_list
    object_feature_list = []
    # non_visible_object_feature_list = []
    for frame_ind in range(pra_start_ind, pra_end_ind):
        # we add mark "1" to the end of each row to indicate that this row exists, using list(pra_now_dict[frame_ind][obj_id])+[1]
        # -mean_xy is used to zero_centralize data
        # now_frame_feature_dict = {obj_id : list(pra_now_dict[frame_ind][obj_id]-mean_xy)+[1] for obj_id in pra_now_dict[frame_ind] if obj_id in visible_object_id_list}
        now_frame_feature_dict = {obj_id: (
            [1] + list( obj_dict[frame_ind][obj_id] - mean_xy) + [1] if obj_id in visible_object_id_list else [1] + list(
                obj_dict[frame_ind][obj_id] - mean_xy) + [0]) for obj_id in obj_dict[frame_ind]}

        # if the current object is not at this frame, we return all 0s by using dict.get(_, np.zeros(total_feature_dimension))
        # add visible_object_feature
        now_visible_object_feature = np.array(
            [np.array(now_frame_feature_dict.get(vis_id, np.zeros(object_feature_dimension))) for vis_id in visible_object_id_list])
        now_visible_object_feature = np.pad(now_visible_object_feature,
                                            ((0, 0), (0, total_feature_dimension - object_feature_dimension)))
        now_frame_feature = now_visible_object_feature

        # add visible_dia_feature
        if num_visible_dia > 0:
            now_visible_dia_feature = np.array(
                [np.array(now_dia_feature_dict.get(vis_id, np.zeros(total_feature_dimension))) for vis_id in visible_dia_id_list])
            now_frame_feature = np.append(now_frame_feature, now_visible_dia_feature, axis=0)

        # add non_visible_object_feature
        if num_non_visible_object > 0:
            now_non_visible_object_feature = np.array(
                [np.array(now_frame_feature_dict.get(vis_id, np.zeros(object_feature_dimension))) for vis_id in non_visible_object_id_list])
            now_non_visible_object_feature = np.pad(now_non_visible_object_feature,
                                                ((0, 0), (0, total_feature_dimension - object_feature_dimension)))
            now_frame_feature = np.append(now_frame_feature, now_non_visible_object_feature, axis=0)

        # add non_visible_dia_feature
        if num_non_visible_dia > 0:
            now_non_visible_dia_feature = np.array(
                [np.array(now_dia_feature_dict.get(vis_id, np.zeros(total_feature_dimension))) for vis_id in non_visible_dia_id_list])
            now_frame_feature = np.append(now_frame_feature, now_non_visible_dia_feature, axis=0)

        object_feature_list.append(now_frame_feature)


    # object_feature_list has shape of (frame#, object#+dia#, 29) 29 = 27features + 1mark
    object_feature_list = np.array(object_feature_list)

    # object feature with a shape of (frame#, object#+dia#, 29) -> (object#+dia#, frame#, 29)
    object_frame_feature = np.zeros(( max_num_object + max_num_dia, pra_end_ind - pra_start_ind, total_feature_dimension))

    # np.transpose(object_feature_list, (1,0,2)): (object#+dia#, frame#, 29)
    object_frame_feature[:num_visible_object + num_non_visible_object + num_visible_dia + num_non_visible_dia] = np.transpose(object_feature_list, (1, 0, 2))
    return object_frame_feature, neighbor_matrix, m_xy


def generate_train_data(track_file_path, dia_file_path):
    '''
    Read data from $pra_file_path, and split data into clips with $total_frames length.
    Return: feature and adjacency_matrix
        feture: (N, T, V, C)
            N is the number of training data (number of start frame)
            T is the temporal length of the data. history_frames + future_frames
            V is the maximum number of objects. zero-padding for less objects.
            C is the dimension of features, 10raw_feature + 1mark(valid data or not)
	'''
    obj_dict = get_frame_instance_dict(track_file_path, pra_is_train=True)
    dia_dict = get_dia_dict(dia_file_path, pra_is_train=True)
    frame_id_set = sorted(set(obj_dict.keys()))
    dia_id_set = sorted(set(dia_dict.keys()))

    all_feature_list = []
    all_adjacency_list = []
    all_mean_list = []

    for start_ind in frame_id_set[:-total_frames + 1]:
        start_ind = int(start_ind)
        end_ind = int(start_ind + total_frames)
        observed_last = start_ind + history_frames - 1
        ind_in_set = True
        for x in range(start_ind, end_ind):
            if x not in frame_id_set or x not in dia_id_set:
                ind_in_set = False
        if ind_in_set:
            object_frame_feature, neighbor_matrix, mean_xy = process_data(obj_dict, dia_dict, start_ind, end_ind, observed_last)
            all_feature_list.append(object_frame_feature)
            all_adjacency_list.append(neighbor_matrix)
            all_mean_list.append(mean_xy)

    # (N, T, V, C)
    all_feature_list = np.transpose(all_feature_list, (0, 2, 1, 3))
    all_adjacency_list = np.array(all_adjacency_list)
    all_mean_list = np.array(all_mean_list)
    # print(all_feature_list.shape, all_adjacency_list.shape)
    return all_feature_list, all_adjacency_list, all_mean_list


def generate_test_data(track_file_path, dia_file_path):
    obj_dict = get_frame_instance_dict(track_file_path, pra_is_train=False)
    dia_dict = get_dia_dict(dia_file_path, pra_is_train=False)
    frame_id_set = sorted(set(obj_dict.keys()))
    dia_id_set = sorted(set(dia_dict.keys()))

    all_feature_list = []
    all_adjacency_list = []
    all_mean_list = []

    for start_ind in frame_id_set[:-history_frames + 1]:
        start_ind = int(start_ind)
        end_ind = int(start_ind + history_frames)
        observed_last = start_ind + history_frames - 1
        ind_in_set = True
        for x in range(start_ind, end_ind):
            if x not in frame_id_set or x not in dia_id_set:
                ind_in_set = False
        if ind_in_set:
            object_frame_feature, neighbor_matrix, mean_xy = process_data(obj_dict, dia_dict, start_ind, end_ind,
                                                                          observed_last)
            all_feature_list.append(object_frame_feature)
            all_adjacency_list.append(neighbor_matrix)
            all_mean_list.append(mean_xy)

    # (N, T, V, C)
    all_feature_list = np.transpose(all_feature_list, (0, 2, 1, 3))
    all_adjacency_list = np.array(all_adjacency_list)
    all_mean_list = np.array(all_mean_list)
    # print(all_feature_list.shape, all_adjacency_list.shape)
    return all_feature_list, all_adjacency_list, all_mean_list


def generate_data(track_file_path, dia_file_path, pra_is_train=True):
    all_data = []
    all_adjacency = []
    all_mean_xy = []
    if pra_is_train:
        now_data, now_adjacency, now_mean_xy = generate_train_data(track_file_path, dia_file_path)
    else:
        now_data, now_adjacency, now_mean_xy = generate_test_data(track_file_path, dia_file_path)

    all_data.extend(now_data)
    all_adjacency.extend(now_adjacency)
    all_mean_xy.extend(now_mean_xy)

    all_data = np.array(all_data)  # (N, T, V, C)
    all_adjacency = np.array(all_adjacency)  # (N, V, V + W) W: max number of dia
    all_mean_xy = np.array(all_mean_xy)  # (N, 2)

    print(np.shape(all_data), np.shape(all_adjacency), np.shape(all_mean_xy))

    # save training_data and trainjing_adjacency into a file.
    if pra_is_train:
        save_path = 'train_data.pkl'
    else:
        save_path = 'test_data.pkl'
    with open(save_path, 'wb') as writer:
        pickle.dump([all_data, all_adjacency, all_mean_xy], writer)


if __name__ == '__main__':
    track_file_path = glob.glob(os.path.join(data_root, 'recorded_trackfiles/', scenario_name, 'vehicle*'+str(track_file_id)+'.csv'))[0]
    dia_file_path = glob.glob(os.path.join(data_root, 'recorded_dia_files/', scenario_name, 'vehicle*'+str(track_file_id)+'_dia.zip'))[0]

    print('Splitting dataset...')
    data_set = pd.read_csv(track_file_path)
    is_test = train_test_split(data_set)
    del data_set

    print('Generating Training Data...')
    generate_data(track_file_path, dia_file_path, pra_is_train=True)

    print('Generating Testing Data...')
    generate_data(track_file_path, dia_file_path, pra_is_train=False)

