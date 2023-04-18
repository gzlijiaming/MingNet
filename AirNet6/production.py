import argparse
import json
import logging
import os
import socket
import sys
from datetime import date, datetime, timedelta
from platform import release

import numpy as np
sys.path.append("DataSource/mon") 
from air_stat import AirStat
from test_model import Tester

sys.path.append("DataPrepare") 
import pandas as pd
from data_file import DataFile, DataStep, SourceType
from ploter import Ploter


def _setup_logger(log_dir):
    # sourcery skip: extract-duplicate-method, inline-immediately-returned-variable, move-assign-in-block
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
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    fhlr = logging.FileHandler(os.path.join( log_dir,f'production_airnet6_{hostname}.log')) # 输出到文件的handler
    fhlr.setFormatter(formatter)
    fhlr.setLevel('DEBUG')
    logger.addHandler(chlr)
    logger.addHandler(fhlr)
    return logger 

def _truncate_time( dt):
    return dt.replace(hour=0, minute=0, second=0, microsecond=0)


class Production():
                
    def __init__(self,data_dir, lookback_days, start_date, arguments, source_type, weather_source, model_dir=None) :
        self.lookback_days = lookback_days
        self.logger = logging.getLogger()
        self.source_type = SourceType.guoKong
        self.data_dir = os.path.join(data_dir, self.source_type.name)
        self.upload_dir = os.path.join(data_dir, 'upload', self.source_type.name)
        self.data_file = DataFile( self.data_dir)
        self.start_date = start_date
        self.min_date =datetime(1999,1,1)

        if not os.path.exists(self.upload_dir):
            os.mkdir(self.upload_dir)
        self.newest_time_path = os.path.join(self.data_dir, 'newestTime.csv')
        self.metas = self.data_file.read_meta(self.source_type)
        self.points_df = self.metas['points']
        self.channels_df = self.metas['channels']
        self.tester = Tester(self.data_dir, arguments, source_type, weather_source, '', 0, model_dir)
        self.read_start_time = max( self.start_date, _truncate_time(datetime.now())+timedelta(days = -self.lookback_days))

    def load_newest(self):
        C = self.channels_df.shape[0]
        end_date = _truncate_time(datetime.now()) + timedelta(days=30)
        self.newest_data = [None for _ in range(C)]
        if os.path.exists(self.newest_time_path):
            self.is_all_new = False
            self.newest_time = pd.read_csv(self.newest_time_path, index_col=0, parse_dates=[0,1], infer_datetime_format=True)
            for channelIndex in range(C):
                # newestDatas [T, V, C]
                self.newest_data[channelIndex] = np.load(os.path.join(self.data_dir, f'newestData_c{channelIndex}.npy'))
            self.newest_end_date = self.start_date + timedelta(hours= self.newest_time.shape[0])

    
    def write_fore_data(self, action_time):
        self.load_newest()
        # write mon data, done by datasource ,not here
        # write fore data
        csv_lines = []
        src_start = 0
        for row_index,row in self.channels_df.iterrows():
            channel_index = self.channels_df.index.get_loc(row_index)
            result_data_path = os.path.join(self.data_dir, 'temp', f'resultData_c{channel_index}_{action_time:%Y%m%d_%H}.npy')
            result_data = np.load(result_data_path );
            src_end = result_data.shape[0]
            dst_start = self.newest_time.index.get_loc(action_time)+1
            dst_end = dst_start +result_data.shape[0]
            self.newest_data[channel_index][dst_start:dst_end, :, 0] = result_data[src_start:src_end, :, 0]
            newest_data_path = os.path.join(self.data_dir, f'newestData_c{channel_index}.npy')
            np.save(newest_data_path, self.newest_data[channel_index])

            self.prepare_csv(action_time, csv_lines, result_data, row_index)
            os.remove( result_data_path)


        # self.newestTime.to_csv(self.newestTimePath)
        csv_path = os.path.join(self.upload_dir, f'{self.source_type.name}_{action_time:%Y%m%d_%H}.csv')
        result_df = pd.DataFrame(csv_lines)
        result_df.to_csv(csv_path)



    def prepare_csv(self,action_time, csv_lines, datas, row_index):
        # 已经有，就不用做了
        # forecast_time = datetime.now()
        forecast_time = action_time
        tcid = 3
        data_type=SourceType.guoKong.value
        data_type_name = SourceType.guoKong.name
        point_indexs = self.data_file.get_point_index_by_cid( self.points_df['cid'], row_index)
        for pindex, index in enumerate(point_indexs):
            point = self.points_df.loc[index]
            uid = point['vid']
            name = point['name']
            region_code = point['region_code']
            region_name = point['region_name']
            cid = row_index
            c_name = self.channels_df.loc[row_index]['name']
            for tIndex in range(datas.shape[0]):
                monitor_time = action_time + timedelta(hours= tIndex+1)
                
                monitor_value = datas[tIndex, pindex, 0]
                csv_lines.append({'uid':uid, 
                                'name':name, 
                                'region_code':region_code,
                                'region_name':region_name,
                                'forecast_time':forecast_time, 
                                'tcid':tcid, 
                                'data_type':data_type, 
                                'data_type_name':data_type_name, 
                                'monitor_time':monitor_time,
                                'cid':cid, 
                                'c_name':c_name,
                                'monitor_value':monitor_value})




    def forecast_current(self, action_time, param_group_id):
        action_time = self.tester.forecast_current(action_time, param_group_id)
        self.write_fore_data(action_time)
        return action_time
        
if __name__ == '__main__':
    program_begin = datetime.now()
    ap = argparse.ArgumentParser()
    ap.add_argument("-ro", "--data_dir", required=True, help="root")
    ap.add_argument("-st", "--source_type", required=True, help="config")
    ap.add_argument("-ws", "--weather_source", required=False, help="config")
    ap.add_argument("-lb", "--lookback_days", type=int, required=True, help="config")
    ap.add_argument("-ns", "--start_date", required=True, help="config")
    ap.add_argument("-md", "--model_dir", default='', required=False, help="vs model Data dir")
    ap.add_argument("-pg", "--param_groud_id", required=False,  type=int,default=1, help="param_groud_id")
    ags, unknowns = ap.parse_known_args()
    arguments = vars(ags)

    data_dir = arguments['data_dir']
    logger = _setup_logger(os.path.join( data_dir,'log') )
    start_date = datetime.fromisoformat( arguments["start_date"])
    production = Production(data_dir,arguments["lookback_days"], start_date, 
                            arguments,
                            SourceType[arguments["source_type"]],SourceType[arguments["weather_source"]],
                            arguments["model_dir"],)
    # 做最新    （真实）
    action_time = datetime.now()
    # action_time = datetime(2023,4,10,0)
    action_time = action_time.replace( minute=0, second=0, microsecond=0)
    action_time = production.forecast_current( action_time, arguments["param_groud_id"])
    ploter = Ploter(production.data_dir, DataStep.normalized, production.source_type)
    ploter.plot_newest(_truncate_time(datetime.now()+ timedelta(days=-8)), 
                       _truncate_time(datetime.now()+ timedelta(days=8)), action_time, xrange='hour')
    air_stat = AirStat(data_dir, SourceType[arguments["source_type"]], action_time, action_time + timedelta(hours=7*24+1))
    air_stat.stat_all()
    
    # 替换历史 begin   (临时)
    # action_time = datetime(2023,3,9,16)
    # while action_time<datetime(2023,3,13,16):
    #     production.forecast_current( action_time, arguments["param_groud_id"])

    #     air_stat = AirStat(data_dir, SourceType[arguments["source_type"]], action_time, action_time + timedelta(hours=7*24+1))
    #     air_stat.stat_all()
    #     action_time += timedelta(hours=1)
    # 替换历史 end
    
    production.logger.info(f'{production.source_type.name} forecast production 用时： {str(datetime.now() - program_begin)}')
