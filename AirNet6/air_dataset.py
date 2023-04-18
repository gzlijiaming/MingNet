from datetime import timedelta
import torch
from torch.utils.data import Dataset
import numpy as np
import os,sys
import pandas as pd
sys.path.append("DataPrepare") 
from data_file import DataFile, DataStep, SourceType



class AirDataset(Dataset):
    FLOAT_TYPE = torch.float32

    def __init__(self, data_dir,source_type,weather_source, shift_day, input_timesteps, predict_timesteps, channel_index, channel_roll, is_train, train_date, test_date, end_date, dev):
        self.channel_index = channel_index
        self.predict_timesteps = predict_timesteps
        self.y_pos = predict_timesteps.copy()
        self.input_timesteps = input_timesteps
        self.shift_day = shift_day
        self.is_train = is_train
        self.weather_source = weather_source
        self.channel_roll = channel_roll
        data_file = DataFile(data_dir)
        datas=data_file.read_data(channel_index,source_type,DataStep.normalized)
        metas= data_file.read_meta(source_type)
        cid =metas['channels'].iloc[channel_index].name
        _, timeFile, _ = data_file.get_meta_filename(source_type)
        times_df = pd.read_csv(timeFile, index_col=3, parse_dates=[3,4], infer_datetime_format=True)
        point_indexs = data_file.get_point_index_by_cid( metas['points']['cid'],cid)
        #weatherDatas [days,T(168 hours), V, C]  , 每一个day的数据为上一天预测的今天至7天的数据
        # weatherDatas=dataFile.readData(0,WeatherMap[ source_type],DataStep.normalized)
        # weatherDatas = weatherDatas[:,:,pointIndexs,:]
        weatherMetas= data_file.read_meta(self.weather_source)

        # newestDatas [T, V, C]
        newest_datapath = os.path.join(data_dir, f'{self.weather_source.name}_newestWeatherData.npy')
        newest_time_path = os.path.join(data_dir, f'{self.weather_source.name}_newestWeatherTime.csv')
        newest_weather_datas = np.load(newest_datapath )
        #最后一个channel用于测试，舍去
        newest_weather_datas = newest_weather_datas[:,:,:-1] # 去掉最后一个测试的channel
        newest_weather_time = pd.read_csv(newest_time_path, index_col=0, parse_dates=[0,1], infer_datetime_format=True)
        # 加入worktime
        newest_worktime = np.load(os.path.join(data_dir, f'{source_type.name}_newest_worktime.npy'))
        # weather_work_data = np.concatenate((newest_worktime[:,:,channel_index:channel_index+1], 
        #                                     np.roll(newest_weather_datas,self.channel_roll, 2) ), axis=2)
        weather_work_data = np.concatenate((newest_worktime[:,:,channel_index:channel_index+1], 
                                            newest_weather_datas), axis=2)
        # 选择业务点位
        weather_work_data = weather_work_data[:,point_indexs,:]






        # # 分割
        trian_index = times_df.index.get_loc(train_date)
        test_index = times_df.index.get_loc(test_date)
        end_index = times_df.index.get_loc(end_date + timedelta(hours=-1))+1
        weather_trian_index = newest_weather_time.index.get_loc(train_date) + self.shift_day*24
        weather_test_index = newest_weather_time.index.get_loc(test_date) + self.shift_day*24
        weather_end_index = newest_weather_time.index.get_loc(end_date + timedelta(hours=-1))+1 + self.shift_day*24
        if is_train:
            x = datas[trian_index:test_index,:,:]
            newest_weather_x = weather_work_data[weather_trian_index : weather_test_index, :, :]
            # self.weatherX = weatherDatas[weatherTrianIndex//24 : weatherTestIndex//24, :, :, :]
        else:
            x = datas[test_index:end_index,:,:]
            newest_weather_x = weather_work_data[weather_test_index : weather_end_index, :, :]
            # self.weatherX = weatherDatas[weatherTestIndex//24 : weatherEndIndex//24, :, :, :]

        # original order: T, V, C (vertex/city, timestep, channel/variable)
        # now the order is changed into: C, T, V
        self.x = torch.tensor(x).permute(2, 0, 1).to(AirDataset.FLOAT_TYPE).to(dev)
        self.newest_weather_x = torch.tensor(newest_weather_x).permute(2, 0, 1).to(AirDataset.FLOAT_TYPE).to(dev)
    

    def get_newest_weather(self, item):
        return self.newest_weather_x[:, item:item + self.input_timesteps, :]

    def channel_count(self):
        return self.x.shape[0] + self.newest_weather_x.shape[0]

    def point_count(self):
        return self.x.shape[2]

    def __getitem__(self, item):
        # C, T, V
        x = self.x[:, item:item + self.input_timesteps, :]
        weather_x = self.get_newest_weather(item)
        self.y_pos = item + self.input_timesteps +self.predict_timesteps-1
        y = self.x[0, self.y_pos, :]
        return torch.roll( torch.concat( (x, weather_x), axis=0), self.channel_roll, 0), y
        # return torch.concat( (x, weather_x), axis=0), y

    def __len__(self):
        return self.x.shape[1] - self.input_timesteps - self.predict_timesteps[len(self.predict_timesteps)-1] + 1
