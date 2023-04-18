#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pickle
from enum import Enum,unique
import pandas as pd
import numpy as np


ChannelSplit = '_'

@unique
class DataStep(Enum):
    src=0
    pre_audited = 1
    pos_audited=2
    spatial_traeted=3
    time_treated=4
    normalized=16

class SourceType(Enum):
    dataset=0
    guoKong=1
    shengKong=2
    gfs=423
    emission=4
    ca=5
    gfs_gd = 6
    gfs_ca = 7
    gfs_fj = 8
    gfs_hb = 9
    gfs_js = 10
    gfs_sx = 11
    gfs_zj = 12
    
    cmaq = 28       # 对比用
    wrfchem = 31    # 对比用
    ji_he = 394    # 对比用
    aq1_cb = 15
    combined=16
    air_net = 424
    prophet = 425





class DataFile():
    def __init__(self, dataDir):
        self.data_dir = dataDir

    def get_meta_filename(self,sourceType ):
        return os.path.join(self.data_dir , f'{sourceType.name}_meta_points.csv'), \
            os.path.join(self.data_dir , f'{sourceType.name}_meta_times.csv'), \
            os.path.join(self.data_dir , f'{sourceType.name}_meta_channels.csv')

    def read_meta(self, source_type):
        points_file, times_file, channels_file = self.get_meta_filename(source_type)
        points_df = pd.read_csv(points_file)
        times_df = pd.read_csv(times_file, index_col=0,parse_dates=['startTime','endTime'], infer_datetime_format=True)
        channels_df = pd.read_csv(channels_file, index_col=0)
        metas = {'points':points_df, 'times':times_df, 'channels':channels_df}
        return metas


    def write_meta(self, metas, source_type):
        points_file, times_file, channels_file = self.get_meta_filename(source_type)
        metas['points'].to_csv(points_file)
        metas['times'].to_csv(times_file)
        metas['channels'].to_csv(channels_file)


    def get_data_filename(self, channel_index, source_type, data_step):
        return os.path.join(self.data_dir , f'{source_type.name}_{data_step.name}_c{channel_index}.npy' )
    
    # original order: T, V, C (vertex/city, timestep, channel/variable)
    def read_data(self, channel_index, source_type, data_step):
        data_path = self.get_data_filename(channel_index, source_type, data_step)
        return np.load(data_path)
        # with open(dataPath, "rb") as f:
        #     datas = pickle.load(f)
        # return datas['all']


    def write_data(self,channel_index, datas,source_type, data_step):
        data_file = self.get_data_filename(channel_index, source_type, data_step)
        np.save(data_file,datas )
        # dataSet = {}
        # # # 分割的时候，取整天
        # # trainCount =int( datas.shape[0]*0.8//24*24)
        # # dataSet['train'],dataSet['test']=np.split( datas,[trainCount],axis=0)
        # dataSet['all'] = datas
        # with open(dataFile, "wb") as f:
        #     pickle.dump(dataSet,f)


    def get_point_index_by_cid(self, series, cid):
        indexs =[]
        cids = f'_{cid}_'
        for i in range(series.shape[0]):
            if cids in series.iloc[i]:
                indexs.append(i)
        return indexs
