from datetime import datetime, timedelta, date
import json
from platform import release
import numpy as np
import socket
import logging
import os, sys
import argparse
from db_reader import DbReader
from pretreat import pretreat
import pandas as pd
sys.path.append("DataPrepare") 
from data_file import DataFile, DataStep,SourceType
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
    fhlr = logging.FileHandler(os.path.join( log_dir,f'production_db_{hostname}.log')) # 输出到文件的handler
    fhlr.setFormatter(formatter)
    fhlr.setLevel('DEBUG')
    logger.addHandler(chlr)
    logger.addHandler(fhlr)
    return logger 

def _truncate_time( dt):
    return dt.replace(hour=0, minute=0, second=0, microsecond=0)


class Production():
                
    def __init__(self,data_dir, config_path, id, lookback_days, start_date) :
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
        
        self.read_start_time = max( self.start_date, _truncate_time(datetime.now())+timedelta(days = -self.lookback_days))
        self.load_newest()
        self.loader = DbReader(self.data_dir, config_path,id, self.read_start_time, datetime.now(), data_dir)

    def load_newest(self):
        V = self.points_df.shape[0]
        C = self.channels_df.shape[0]
        end_date = _truncate_time(datetime.now()) + timedelta(days=30)
        self.newest_data = [None for _ in range(C)]
        if os.path.exists(self.newest_time_path):
            self.is_all_new = False
            self.newest_time = pd.read_csv(self.newest_time_path, index_col=0, parse_dates=[0,1], infer_datetime_format=True)
            for cindex in range(C):
                # newestDatas [T, V, C]
                self.newest_data[cindex] = np.load(os.path.join(self.data_dir, f'newestData_c{cindex}.npy'))
            self.newest_end_date = self.start_date + timedelta(hours= self.newest_time.shape[0])
        else:
            self.is_all_new = True
            self.newest_time = pd.DataFrame(data = [], columns=['forecast_time'],index=[], dtype='datetime64[ns]')
            self.read_start_time = self.start_date
            self.newest_end_date = self.start_date 
        if self.newest_end_date < end_date:
            self.expand_newest(V, C)

    # 扩张 newest
    def expand_newest(self, V, C):
        self.newest_end_date = _truncate_time( datetime.now()) + timedelta(days=100)
        add_hours =( (self.newest_end_date- self.start_date).days)*24 - self.newest_time.shape[0]
        
        begin_add_time = self.start_date +timedelta(hours=self.newest_time.shape[0])
        add_time_data = [self.min_date for _ in range(add_hours)]
        add_time_index = [begin_add_time + timedelta(hours=hour) for hour in range(add_hours)]
        add_time = pd.DataFrame(data=add_time_data, index=add_time_index,columns=['forecast_time'], dtype='datetime64[ns]')
        self.newest_time = pd.concat((self.newest_time, add_time), axis=0)
        
        if not self.is_all_new:
            for cindex in range(C):
                add_data = np.empty((add_hours, self.newest_data[cindex].shape[1], self.newest_data[cindex].shape[2]), dtype=float)
                add_data.fill(np.nan)
                self.newest_data[cindex] = np.concatenate((self.newest_data[cindex], add_data),axis=0)
        

    def load_last(self, actionTime):
        self.loader.replace_times_meta()
        self.loader.load_data(is_production=True)
        
        C = self.channels_df.shape[0]
        V = self.points_df.shape[0]
        src_data = [None for _ in range(C)]
        # pos_data = [None for _ in range(C)]
        for cindex in range(C):
            # newData [T,V,C] channel:  # valid, wsh, hasData, markers, wsh is data
            src_data[cindex] = self.data_file.read_data(cindex, self.source_type, DataStep.src)
            # pos_data[cindex] = self.data_file.read_data(cindex, self.source_type, DataStep.pos_audited)
        if self.is_all_new:
            add_hours =( (self.newest_end_date- self.start_date).days)*24
            for cindex in range(C):
                add_data = np.empty((add_hours, src_data[cindex].shape[1], 1), dtype=float)
                add_data.fill(np.nan)
                self.newest_data[cindex] = add_data
            
        src_begin_time = self.loader.begin_date
        dst_begin_time = self.start_date
        for tindex in range(src_data[0].shape[0]):
            monitor_time = src_begin_time + timedelta(hours=tindex)
            if self.newest_time.loc[monitor_time,'forecast_time'] is None or self.newest_time.loc[monitor_time,'forecast_time'] < actionTime:
                self.newest_time.loc[monitor_time,'forecast_time'] = actionTime
                for cindex in range(C):
                    dst_tindex = self.newest_time.index.get_loc(monitor_time)
                    for pindex in range(V): #先填pre，再填pos
                        # markers = src_data[cindex][tindex, pindex, 3:]
                        if not np.isnan(src_data[cindex][tindex, pindex, 0] ): # 有marker的数据，就不要了
                            self.newest_data[cindex][dst_tindex, pindex, 0] = src_data[cindex][tindex, pindex, 0]
                        # if posData[cIndex][tIndex, pIndex, 0] != np.nan: # 有marker的数据，就不要了
                        #     self.newestData[cIndex][dstTIndex, pIndex, 0] = posData[cIndex][tIndex, pIndex, 0]
        try:
            for cindex in range(C):
                newest_data_path = os.path.join(self.data_dir, f'newestData_c{cindex}.npy')
                np.save(newest_data_path, self.newest_data[cindex])
            self.newest_time.to_csv(self.newest_time_path)
        except Exception as e:
            self.logger.error(f'saveNewest  save file {newest_data_path} \
                错误：{e}')
        
        
if __name__ == '__main__':
    program_begin = datetime.now()
    ap = argparse.ArgumentParser()
    ap.add_argument("-ro", "--data_dir", required=True, help="root")
    ap.add_argument("-dc", "--db_config", required=True, help="config")
    ap.add_argument("-db", "--db", required=True, help="config")
    ap.add_argument("-lb", "--lookback_days", type=int, required=True, help="config")               
    ap.add_argument("-ns", "--start_date", required=True, help="config")
    ags, unknowns = ap.parse_known_args()
    arguments = vars(ags)

    data_dir = arguments['data_dir']
    logger = _setup_logger(os.path.join( data_dir,'log') )
    
    production = Production(data_dir, arguments["db_config"], arguments["db"], 
                            arguments["lookback_days"], datetime.fromisoformat( arguments["start_date"]))
    production.load_last( datetime.now())
    pt = pretreat(production.data_dir, production.source_type, False)
    pt.pretreat_all()
    
    plot = Ploter(production.data_dir, DataStep.src, production.source_type)
    plot.plot_all(xrange='hour')
    plot = Ploter(production.data_dir, DataStep.normalized, production.source_type)
    plot.plot_all(xrange='hour')
    plot = Ploter(production.data_dir, DataStep.normalized, production.source_type)
    plot.plot_newest(_truncate_time(datetime.now()), _truncate_time(datetime.now()+ timedelta(days=1)), None, xrange='hour')
    
    production.logger.info(f'{production.source_type.name} production 用时： {str(datetime.now() - program_begin)}')
