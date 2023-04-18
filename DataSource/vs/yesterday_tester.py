#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys,os
from matplotlib.pyplot import axes
from tqdm import tqdm
import numpy as np
from datetime import datetime,date,time,timedelta
sys.path.append("DataPrepare") 
from data_file import DataFile, DataStep, SourceType
sys.path.append("AirNet") 
import air_dataset as AirDatasetModual


_MAX_FORE_DAY = 7


class YesterdayTester():
    def __init__(self, data_dir, vs_dir, source_type, train_date, test_date , end_date) :
        self.train_date = train_date
        self.test_date = test_date
        self.end_date = end_date
        self.source_type = source_type
        self.test_dir = os.path.join( vs_dir, 'yesterday')
        if not os.path.exists(self.test_dir):
            os.mkdir(self.test_dir)
        self.data_file = DataFile(data_dir)

    def forecast_test_data(self, channels_df, times_df, input_preday_list, steps_list, batchs_list):
        # 收集结果
        for row_index,row in channels_df.iterrows():
            channel_index = channels_df.index.get_loc(row_index)
            fdata_path = os.path.join( self.test_dir , f'{self.source_type.name}_fData_c{channel_index}.npy')
            odata_path = os.path.join( self.test_dir, f'{self.source_type.name}_oData_c{channel_index}.npy')
            
            #源数据db
            src_data = self.data_file.read_data(channel_index, self.source_type, DataStep.src) # [T, V, C]
            test_index = times_df.index.get_loc(self.test_date)
            end_index = times_df.index.get_loc(self.end_date + timedelta(hours=-1))+1
            odata = src_data[test_index:end_index+(_MAX_FORE_DAY-1)*24,:,0]
            days = (self.end_date - self.test_date).days
            fdata = np.empty((days, _MAX_FORE_DAY*24, odata.shape[1]),dtype=float)
            day_index = 0
            current_time = self.test_date 
            while current_time < self.end_date:
                src_begin_index = test_index+day_index*24 - 24
                for i in range(_MAX_FORE_DAY):
                    fdata[day_index , i*24: i*24+24:, :] = src_data[src_begin_index:src_begin_index + 24,:,0]
                day_index += 1
                current_time += timedelta(days=1)
  
            np.save(fdata_path,fdata)
            np.save(odata_path,odata)
            
            
                    
if __name__ == '__main__':
    program_begin = datetime.now()

    print(f"YesterdayTester fit 用时： {datetime.now() - program_begin}")





