#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 本代码只是实例，用正弦数据充当监测数据，必须用本地区真实空气质量历史数据代替

import argparse
import logging
from operator import index
import random
import shutil
import socket
import psycopg2
import os,sys
import pickle
from tqdm import tqdm
import numpy as np
from datetime import datetime,date,timedelta
sys.path.append("DataPrepare") 
from data_file import DataFile, DataStep,SourceType,ChannelSplit
import json
import math
import pandas as pd

_DEBUG_TIME = False

#监测因子（O 3、NO 2、CO、SO 2、PM   10、PM   2.5）
#按照HJ 633—2012，设定监测因子的最大值，规一化时要用。
_CHANNEL_MAX={
    286: 1200, #'O3'
    386:3840, #'NO 2'
    486:150, #'CO'
    586:2620, #'SO 2'
    8086:600, #'PM   10'
    8088:500,    #'PM   2.5'
}
_CYCLE1 = 24
_CYCLE2 = 7*_CYCLE1

def _setup_logger(logDir):
    # sourcery skip: extract-duplicate-method, inline-immediately-returned-variable, move-assign-in-block
    if not os.path.exists(logDir):
        os.mkdir(logDir)
    logger = logging.getLogger()
    logger.setLevel('INFO')
    BASIC_FORMAT = "%(asctime)s\t%(message)s"
    BASIC_DATE_FORMAT = '%H:%M:%S'
    DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(BASIC_FORMAT, BASIC_DATE_FORMAT)
    chlr = logging.StreamHandler() # 输出到控制台的handler
    chlr.setFormatter(formatter)
    chlr.setLevel('INFO')  # 也可以不设置，不设置就默认用logger的level
    FULL_FORMAT = "%(asctime)s:%(levelname)s\t%(message)s"
    formatter = logging.Formatter(FULL_FORMAT, DATE_FORMAT)
    hostname=socket.gethostname()
    fhlr = logging.FileHandler(os.path.join( logDir,f'db_reader_{hostname}.log')) # 输出到文件的handler
    fhlr.setFormatter(formatter)
    fhlr.setLevel('DEBUG')
    logger.addHandler(chlr)
    logger.addHandler(fhlr)
    return logger 





class DbReader():
    def __init__(self,data_dir, config_path, db_id, begin_date, end_date, production_dir) :
        self.begin_date=begin_date
        # self.beginDate=date(2019,1,1)
        self.end_date = end_date
        self.logger = logging.getLogger()
        self.data_dir = data_dir
        self.source_type = SourceType.guoKong
        self.data_file = DataFile(self.data_dir)
        self.production_dir =production_dir
        # with open(config_path, 'r') as f:
        #     configs = json.load(f)
        # for config in configs:
        #     if config['id']==db_id:
        #         self.config = config

    def load_data(self, is_production=False):
        # is_production 参数使用场景：==False时，为训练阶段，可以用审核后数据；==True时，为常规运行阶段，可以用未审核数据
        meta_data = self.data_file.read_meta(self.source_type)
        points_df = meta_data['points']
        channels_df = meta_data['channels']
        times_df = meta_data['times']
        cycle1_begin = self.begin_date.hour
        cycle2_begin = 7*self.begin_date.weekday()
        # data

        T = times_df.shape[0]
        data_step = DataStep.src
        out_data_channel_pos = 0
        C = out_data_channel_pos +1
        # original order: T, V, C 
        for row_index,row in channels_df.iterrows():
            channel_index = channels_df.index.get_loc(row_index)
            max_value=float( row['maxValue'])
            min_value=float( row['minValue'])

            data_inserted =0
            subpoints_df = points_df[points_df['cid'].str.contains(ChannelSplit+str(row.name)+ChannelSplit)]
            subpoints_df.set_index('vid' ,inplace=True)
            V = subpoints_df.shape[0]
            out_data =np.empty([T,V,C],dtype=float)
            out_data.fill(np.nan)

            random_count = T*V*C* 5
            for _ in tqdm(range(random_count ), desc=f'{row["symbol"]} {data_step.name} row: ', total=random_count, leave=False):
                tindex = random.randint(0,T-1)
                vindex = random.randint(0,V-1)
                mon_value = (math.sin(((tindex+cycle1_begin) % _CYCLE1)*2* math.pi/_CYCLE1)+2+ 
                             math.sin(((tindex+cycle2_begin) % _CYCLE2)*2* math.pi/_CYCLE2)*0.3+
                             vindex/10000.1)*max_value/8 
                            
                out_data[tindex,vindex,out_data_channel_pos] = mon_value
            self.data_file.write_data(channel_index , out_data,self.source_type, data_step)
            data_inserted = np.count_nonzero(out_data)
            self.logger.info(f"{data_step.name} { row['name']}\toutData: {T}*{V}*{C}={T*V*C}\tdata inserted: {data_inserted}")

    
    def load_meta(self):
        metadata = {}

        dirname, filename = os.path.split(os.path.abspath(__file__))
        # 监测因子  [cid, 名称，代号，最大值，最小值]
        channels_df = pd.read_csv(os.path.join(dirname, 'sample_data','sample_meta_channels.csv'))
        channels_df.set_index('cid',inplace=True)
        for channelMaxKey in _CHANNEL_MAX:
            channels_df.at[channelMaxKey,'maxValue'] = _CHANNEL_MAX[channelMaxKey]
        channels_df.reset_index(inplace=True)
        channels_df = channels_df.rename(columns = {'index':'cid'})
        channels_df.set_index('cid',inplace=True)
        metadata['channels'] = channels_df
        C = len(metadata['channels'])
        self.logger.info(f"channels count: {C}")

        # 点位  [uid,cid, name,region_code,region_name, count, lon, lat  ]
        points_df = pd.read_csv(os.path.join(dirname, 'sample_data','sample_meta_points.csv'),  index_col=0)
        metadata['points'] = points_df
        V = len(metadata['points'])
        self.logger.info(f"points count: {V}")


            
        times_df = self.load_times_meta()
        metadata['times'] = times_df
        self.data_file.write_meta(metadata, self.source_type)
                            
                            
    def load_times_meta(self):                        
        # 时间  [tid, code, name, startTime, endTime]
        csv_lines =[]
        tid = 8088
        current_time = self.begin_date
        while current_time < self.end_date:
            csv_lines.append({
                'tid':tid,
                'code':current_time.strftime("%Y%m%d%H%M%S"),
                'name':current_time.strftime("%Y年%-m月%-d日 %H:%M:%S"),
                'startTime':current_time,
                'endTime':current_time+timedelta(hours=1)
            })
            tid += 1
            current_time += timedelta(hours=1)
        
        times_df = pd.DataFrame(csv_lines)
        times_df.set_index('tid',inplace=True)
        return times_df
    
    def replace_times_meta(self):
        # 更新时间元数据，需要用真实的时间元数据替代本代码
        metatata = self.data_file.read_meta(self.source_type)
        times_df = self.load_times_meta()
        metatata['times'] = times_df
        _, timesFile, _ = self.data_file.get_meta_filename(self.source_type)
        times_df.to_csv(timesFile)


    def write_production_data(self):
        dst_dir = os.path.join(self.production_dir,self.source_type.name)
        if not os.path.exists(dst_dir):
            os.mkdir(dst_dir)
        points_file, times_file, channels_file =self.data_file.get_meta_filename(self.source_type)
        shutil.copyfile(points_file, os.path.join(dst_dir, os.path.basename(points_file)))
        shutil.copyfile(times_file, os.path.join(dst_dir, os.path.basename(times_file)))
        shutil.copyfile(channels_file, os.path.join(dst_dir, os.path.basename(channels_file)))

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-ro", "--data_dir", required=True, help="root")
    ap.add_argument("-ed", "--epa_dir", required=True, help="root")
    ap.add_argument("-sta", "--start", required=False, help="config")
    ap.add_argument("-end", "--end", required=False, help="config")
    ap.add_argument("-pd", "--production_dir", default='/datapool01/shared/production', required=False, help="GFS Data dir")
    args, unknowns = ap.parse_known_args()
    arguments = vars(args)

    data_dir = arguments['data_dir']  
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    logger = _setup_logger(os.path.join( data_dir,'log') )
    begin_date= datetime.fromisoformat( arguments['start'])
    end_date = datetime.fromisoformat( arguments['end'])
    loader = DbReader(data_dir, '/home/ming/.config/db/gdep-dbc.json','dm_env', begin_date, end_date, arguments['production_dir'])
    loader.load_meta()
    loader.load_data()
    # 放元数据到生产环境
    loader.write_production_data()

