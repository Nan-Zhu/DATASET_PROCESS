import glob
import os
import pandas as pd
import random

data_root = '../dataset/INTERACTION/recorded_trackfiles/'
total_frames = 12


def train_test_split(data_set, test_size, random_state):
    '''
    Split data set into train data and test data
    (slice test data block as a whole to ensure integrity of trajectory)
        data_set: source data set
        test_size: proportion of test data range from (0,1)
        random_state: random seed
    '''
    print('spliting dataset...')
    random.seed(random_state)
    data_set_count = data_set.shape[0]
    split_start = random.randint(0, int(data_set_count * (1 - test_size)))
    split_end = int(split_start + data_set_count * test_size)
    train_data = data_set[:split_start]
    test_data = data_set[split_start:split_end]
    train_data_append = data_set[split_end:].copy(deep=True)
    track_id_diff = test_data['track_id'].max() - test_data['track_id'].min() - 1
    train_data_append['track_id'] = train_data_append['track_id'] - track_id_diff
    train_data = train_data.append(train_data_append)
    train_data.to_csv('../dataset/INTERACTION/prediction_train/train_data.csv', index=False)
    test_data.to_csv('../dataset/INTERACTION/prediction_test/test_data.csv', index=False)
    print('train_data split: size', train_data.shape)
    print('test_data split: size', test_data.shape)
    print('max object number is', train_data['track_id'].max())

def load_data(split_file_path_list):
    print('Loading dataset...')
    data_set = pd.read_csv(split_file_path_list[0])
    last_track_id = data_set['track_id'].max()
    last_frame_id = data_set['frame_id'].max() + total_frames
    last_timestamp_ms = data_set['timestamp_ms'].max() + total_frames * 100
    for file_path in split_file_path_list[1:]:
        df = pd.read_csv(file_path)
        df['track_id'] += last_track_id
        df['frame_id'] += last_frame_id
        df['timestamp_ms'] += last_timestamp_ms
        data_set = data_set.append(df)
        last_track_id = data_set['track_id'].max()
        last_frame_id = data_set['frame_id'].max() + total_frames
        last_timestamp_ms = data_set['timestamp_ms'].max() + total_frames * 100
    return data_set


if __name__ == '__main__':
    file_path_list = sorted(glob.glob(os.path.join(data_root, '*/vehicle*.csv')))
    split_data_set = load_data(file_path_list)
    train_test_split(split_data_set, 0.2, 0)
