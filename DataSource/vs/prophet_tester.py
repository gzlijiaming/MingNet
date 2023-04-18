#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import logging
import sys,os
from tempfile import tempdir
from tqdm import tqdm
import numpy as np
from datetime import datetime,date,time,timedelta
import pandas as pd
from prophet import Prophet
import ray
sys.path.append("DataPrepare") 
from data_file import DataFile, DataStep, SourceType
import matplotlib.pyplot as plt
import prophet_tester_slice as prophet_tester_sliceModule


_MAX_FORE_DAY = 7
_USE_RAY = True

#让ray任务每完成一个叫一次
def to_iterator(obj_ids):
    while obj_ids:
        done, obj_ids = ray.wait(obj_ids)
        yield ray.get(done[0])

@ray.remote(num_cpus=1,memory=15*1073741824)
def pre_one_point_remote( data_dir,temp_dir, source_type_name, channel_count, src_begin_index, src_end_index,action_time, result_count):
    return prophet_tester_sliceModule.pre_one_point( data_dir,temp_dir, source_type_name, channel_count, src_begin_index, src_end_index,action_time, result_count)





class ProphetTester():
    def __init__(self, data_dir, vs_dir, source_type, train_date, test_date , end_date) :
        self.data_dir = data_dir
        self.train_date = train_date
        self.test_date = test_date
        self.end_date = end_date
        self.source_type = source_type
        self.test_dir = os.path.join( vs_dir, 'prophet')
        if not os.path.exists(self.test_dir):
            os.mkdir(self.test_dir)
        self.test_dir = os.path.join( self.test_dir, source_type.name)
        if not os.path.exists(self.test_dir):
            os.mkdir(self.test_dir)
        self.temp_dir = os.path.join( self.test_dir, 'temp')
        if not os.path.exists(self.temp_dir):
            os.mkdir(self.temp_dir)
        self.logger = logging.getLogger()
   
    def forecast_test_data(self, channels_df, times_df, input_preday_list, steps_list, batchs_list):
        if _USE_RAY:                    
            runtime_env={"working_dir": os.path.dirname(os.path.abspath(__file__)), "excludes":[os.path.dirname(os.path.abspath(__file__))+'/__pycache__']}
            ray.init(address='ray://localhost:10001', include_dashboard=False, runtime_env = runtime_env)
            object_ids = []
        errors = []
        train_index = times_df.index.get_loc(self.train_date)
        test_index = times_df.index.get_loc(self.test_date)
        # train_index = test_index - 14*24
        result_count = _MAX_FORE_DAY*24
        #做预测
        temp_dir = os.path.join(self.data_dir ,'temp')
        if not os.path.exists(temp_dir):
            os.mkdir(temp_dir)
        current_time = self.test_date 
        day_index = 0
        day_count = (self.end_date - self.test_date).days
        source_type_name = self.source_type.name
        while current_time < self.end_date:   # 预测的，多maxForeDay-1天
            action_time = current_time + timedelta(hours=-1)
            if _USE_RAY:                    
                object_ids.append( pre_one_point_remote.remote(self.data_dir, self.temp_dir, source_type_name, channels_df.shape[0], train_index+ day_index*24, test_index+ day_index*24, action_time, result_count))
            else:
                x = prophet_tester_sliceModule.pre_one_point(self.data_dir, self.temp_dir, source_type_name, channels_df.shape[0], train_index+ day_index*24, test_index+ day_index*24, action_time, result_count)
                if x!=None and len(x)>0 :
                    errors.extend(x)
                    errorstr = '\n'.join( x)
                    self.logger.error(x)

            current_time += timedelta(days=1)
            day_index += 1


        # 所有任务完成后，重新组装
        if _USE_RAY:                    
            for x in tqdm(to_iterator(object_ids), total=len(object_ids)):
                if x!=None and len(x)>0 :
                    errors.extend(x)
                    errorstr = '\n'.join( x)
                    self.logger.error(x)

        errorstr = '\n'.join( errors)
        self.logger.error(f"error count={len(errors)}, errors:\n{errorstr}")
            


        

if __name__ == '__main__':
    programBegin = datetime.now()

    ap = argparse.ArgumentParser()
    ap.add_argument("-dp", "--data_dir", required=True,default="data/Air", help="path to the .pkl dataset, e.g. 'data/Dutch/dataset.pkl'")
    ap.add_argument("-src", "--source_type", required=True, type=str,default=1, help="数据源'")
    ap.add_argument("-vs", "--vs_dir", default='/datapool01/shared/vs', required=False, help="vs model Data dir")
    ap.add_argument("-td", "--train_date", required=True, help="")
    ap.add_argument("-ts", "--test_date", required=True, help="")
    ap.add_argument("-ed", "--end_date", required=True, help="")
    ags, unknowns = ap.parse_known_args()
    arguments = vars(ags)

    prophet = ProphetTester(arguments["data_dir"], 
                    arguments["vs_dir"], 
                    SourceType[ arguments["source_type"] ] , 
                    datetime.fromisoformat( arguments["train_date"]),
                    datetime.fromisoformat( arguments["test_date"]),
                    datetime.fromisoformat( arguments["end_date"])
                    )

    data_file = DataFile(prophet.data_dir)
    metas = data_file.read_meta(prophet.source_type)
    channels_df = metas['channels']
    _, timeFile, _ = data_file.get_meta_filename(prophet.source_type)
    times_df = pd.read_csv(timeFile, index_col=3, parse_dates=[3,4], infer_datetime_format=True)

    prophet.forecast_test_data(channels_df, times_df, None, None, None)



    print(f"ProphetTester  用时： {datetime.now() - programBegin}")





