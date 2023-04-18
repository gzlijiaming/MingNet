from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from air_dataset import AirDataset


def get_train_valid_loader(data_dir, source_type, weather_source, input_timesteps, predict_timesteps, batch_size, channel_index, channel_roll, shift_day, train_date, test_date, end_date, dev):
    train_val_dataset = AirDataset(data_dir, source_type, weather_source, shift_day,input_timesteps, predict_timesteps, channel_index, channel_roll, True, train_date, test_date, end_date, dev)

    num_train = len(train_val_dataset)
    indices = list(range(num_train))
    split = int(np.floor(0.1 * num_train))

    np.random.seed(19491001)
    np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = DataLoader(train_val_dataset, batch_size=batch_size, sampler=train_sampler)
    valid_loader = DataLoader(train_val_dataset, batch_size=batch_size, sampler=valid_sampler)

    return train_loader, valid_loader


def get_test_loader(data_dir, source_type, weather_source,input_timesteps, predict_timesteps, batch_size, channel_index, channel_roll, shift_day, train_date, test_date, end_date, dev):
    test_dataset = AirDataset(data_dir,source_type, weather_source, shift_day, input_timesteps, predict_timesteps, channel_index, channel_roll, False, train_date, test_date, end_date, dev)

    return DataLoader(test_dataset, batch_size=batch_size)