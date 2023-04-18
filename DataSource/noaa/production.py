from datetime import datetime, timedelta, date
from platform import release
import numpy as np
import socket
import logging
import os, sys
import argparse
from tqdm import tqdm
from gfs2point import Gfs2Point
from nomads import Nomads
import gfs2point_slice as Gfs2pointSliceModule
import pandas as pd
import matplotlib.pyplot as plt
sys.path.append("DataPrepare") 
from data_file import DataFile, DataStep,SourceType
sys.path.append("DataSource/economy") 
from worktime import Worktime
from pathlib import Path


def _truncate_time( dt):
    return dt.replace(hour=0, minute=0, second=0, microsecond=0)

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
    fhlr = logging.FileHandler(os.path.join( log_dir,f'production_noaa_{hostname}.log')) # 输出到文件的handler
    fhlr.setFormatter(formatter)
    fhlr.setLevel('DEBUG')
    logger.addHandler(chlr)
    logger.addHandler(fhlr)
    return logger 

class Production():
                
    def __init__(self,data_dir,mon_source, weather_source, start_date, lookback_days) :
        self.data_dir = data_dir
        self.weather_source = weather_source
        self.lookback_days = lookback_days
        self.logger = logging.getLogger()
        self.gfs_dir = os.path.join( self.data_dir, self.weather_source.name)
        self.upload_dir = os.path.join(data_dir, 'upload',SourceType.gfs.name)
        self.start_date = start_date
        self.min_date =datetime(1999,1,1)
        if not os.path.exists(self.upload_dir):
            Path( self.upload_dir ).mkdir( parents=True, exist_ok=True )
        self.gfs_dir = os.path.join( self.data_dir, self.weather_source.name)
        self.mon_dir = os.path.join( self.data_dir, mon_source.name)
        self.nomads = Nomads( self.data_dir,  self.weather_source, lookback_days)
        # self.g2p = Gfs2Point(self.monDir,SourceType[arguments['weather_source']], self.dataDir,
        #             SourceType[arguments['source_type']],self.dataDir )
        self.plot_dir = os.path.join(self.gfs_dir,'plot')
        if not os.path.exists(self.plot_dir):
            os.mkdir(self.plot_dir)
        self.newest_data_path = os.path.join(self.mon_dir, f'{self.weather_source.name}_newestWeatherData.npy')
        self.newest_time_path = os.path.join(self.mon_dir, f'{self.weather_source.name}_newestWeatherTime.csv')
        self.worktime_data_path = os.path.join(self.mon_dir, f'{mon_source.name}_newest_worktime.npy')
        # points [vid	cid	name	region_code	region_name	count	lon	lat ]
        self.points_df = pd.read_csv(os.path.join(self.gfs_dir, 'meta_points.csv'))
        # channels [cid	name	symbol	maxValue	minValue]
        self.channels_df = pd.read_csv(os.path.join(self.gfs_dir, 'meta_channels.csv'))
        self.monChannels_df = pd.read_csv(os.path.join(self.mon_dir, f'{mon_source.name}_meta_channels.csv'))
        plt.rcParams["figure.figsize"]=(10,4) #设置图大小
        plt.rcParams["figure.dpi"]=300 #设置图大小
        plt.rcParams["xtick.labelsize"]=6
        
        self.load_newest()
        self.wt = Worktime(self.mon_dir, 
                        self.data_dir, 
                        mon_source, 
                        weather_source, 
                        self.nomads.today_begin_time
                        )

    def load_newest(self):
        V = self.points_df.shape[0]
        C = self.channels_df.shape[0]
        monC = self.monChannels_df.shape[0]
        end_date = datetime.now() + timedelta(days=30)
        if os.path.exists(self.newest_data_path):
            self.is_all_new = False
            # newestDatas [T, V, C]
            self.newest_data = np.load(self.newest_data_path)
            self.newest_time = pd.read_csv(self.newest_time_path, index_col=0, parse_dates=[0,1], infer_datetime_format=True)
        else:
            self.is_all_new = True
            self.newest_data= np.empty((0, V,C), dtype=float)
            self.newest_time = pd.DataFrame(data = [], columns=['forecast_time'],index=[], dtype='datetime64[ns]')
        if os.path.exists(self.worktime_data_path):
            self.worktime_data = np.load(self.worktime_data_path)
        else:
            self.worktime_data= np.empty((self.newest_time.shape[0], V,monC), dtype=float)
            
        newest_end_date = self.start_date + timedelta(hours= self.newest_data.shape[0])
        if newest_end_date < end_date:
            self.expand_newest(V, C, monC)
            self.newest_time.to_csv(self.newest_time_path)


    # 扩张 newest
    def expand_newest(self, V, C, monC):
        newest_end_date = datetime.now() + timedelta(days=100)
        add_hours =( (newest_end_date- self.start_date).days)*24 - self.newest_data.shape[0]
        add_data = np.empty((add_hours, V,C), dtype=float)
        add_worktime_data = np.empty((add_hours, V,monC), dtype=float)
        add_data.fill(np.nan)
        add_worktime_data.fill(np.nan)
        begin_add_time = self.start_date +timedelta(hours=self.newest_data.shape[0])
        add_time_data = [self.min_date for _ in range(add_hours)]
        add_time_index = [begin_add_time + timedelta(hours=hour) for hour in range(add_hours)]
        add_time = pd.DataFrame(data=add_time_data, index=add_time_index,columns=['forecast_time'], dtype='datetime64[ns]')
        self.newest_data = np.concatenate((self.newest_data, add_data),axis=0)
        self.newest_time = pd.concat((self.newest_time, add_time), axis=0)
        self.worktime_data = np.concatenate((self.worktime_data, add_worktime_data),axis=0)
        
        

    def do_one_day(self, todo_date):
        grid_file, errors = self.nomads.get_one_day(todo_date)
        if grid_file is not None:
            split_dst_dir = os.path.join(self.gfs_dir,'zpoint_slice')
            if not os.path.exists(split_dst_dir):
                os.mkdir(split_dst_dir)
            self.logger.info(f'grid2PointSliceCsv start: {grid_file}')
            if errors := Gfs2pointSliceModule.grid2point_slice(self.gfs_dir, grid_file,1 ) :#self.nomads.interHours):
                self.logger.info(f'grid2PointSlice result: {errors}')
            if errors := self.save_csv_and_newest(grid_file):
                self.logger.info(f'grid2PointSliceCsv result: {errors}')
        
    def fill_missing_days(self):
        if self.is_all_new:
            day_count =min(( self.nomads.today_begin_time - self.start_date).days+1,self.nomads.lookback_days)
        else:
            day_count = self.nomads.lookback_days // 2
        forecast_times = self.newest_time['forecast_time'].values
        for todo_date in [self.nomads.today_begin_time+timedelta(days= -d) for d in range(day_count-1,-1,-1)]:
            self.do_one_day( todo_date)
        self.fill_missing_worktime(self.nomads.today_begin_time+timedelta(days= -day_count),
                                 self.nomads.today_begin_time+timedelta(hours= self.nomads.max_fore_hour) )
   
    def fill_missing_worktime(self,  startTime, endTime):
        self.wt.make(self.worktime_data, startTime, endTime)
        
    def save_csv_and_newest(self, gridFile):
        errors = []
        data_type=SourceType.gfs.value
        data_type_name = SourceType.gfs.name
        model_cycle_runtime = gridFile[-6:-4]
        delta_hours = self.nomads.timezone + int(model_cycle_runtime)
        npy_path = os.path.join(self.gfs_dir, 'zpoint_slice', gridFile)
        csv_path = os.path.join(self.upload_dir, f'{data_type_name}_' + gridFile.replace('.npy', '.csv'))
        if os.path.exists(npy_path):
            #datas [T, V, C]， 最后一个channel用于测试，不用，所以减一。
            datas = np.load(npy_path)
            tcid = 3
            forecast_time = datetime.strptime(gridFile[:8],'%Y%m%d')
            forecast_time += timedelta(hours= delta_hours)
            self.save_csv(errors, data_type, data_type_name, csv_path, datas, tcid, forecast_time)
            
            #newest store normalized data
            datas_scaled = np.copy( datas)
            for cIndex in range(datas.shape[2]):
                scaler_max = float(self.channels_df.iloc[cIndex]['maxValue'])
                scaler_min = float(self.channels_df.iloc[cIndex]['minValue'])
                datas_scaled[:,:,cIndex] = (datas[:,:,cIndex] - scaler_min)/(scaler_max-scaler_min)
            datas_scaled[ np.where(datas_scaled<0) ]=0
            datas_scaled[ np.where(datas_scaled>1) ]=1
            self.save_newest(errors,  datas_scaled,  forecast_time)
        else:
            errors.append(f'{npy_path} not exists!')
        return errors

    def save_newest(self, errors, datas, forecast_time):
        for tIndex in range(datas.shape[0]):
            monitor_time = forecast_time + timedelta(hours=tIndex+1)
            if self.newest_time.loc[monitor_time,'forecast_time'] is None or self.newest_time.loc[monitor_time,'forecast_time'] < forecast_time:
                self.newest_time.loc[monitor_time,'forecast_time'] = forecast_time
                self.newest_data[self.newest_time.index.get_loc(monitor_time), :, :] = datas[tIndex, :, :]
        
        try:
            np.save(self.newest_data_path, self.newest_data)
            self.newest_time.to_csv(self.newest_time_path)
        except Exception as e:
            errors.append(f'saveNewest  save file {self.newest_data_path} \
                错误：{e}')

    def save_csv(self, errors, data_type, data_type_name, csv_path, datas, tcid, forecast_time):
        # 已经有，就不用做了
        if os.path.exists(csv_path) and os.path.getsize(csv_path)>1024*1024:   
            return 
                
        row_list = []
        for pindex in range(datas.shape[1]):
            uid = self.points_df.iloc[pindex]['vid']
            name = self.points_df.iloc[pindex]['name']
            region_code = self.points_df.iloc[pindex]['region_code']
            region_name = self.points_df.iloc[pindex]['region_name']
            for cindex in range(datas.shape[2]):
                cid = self.channels_df.iloc[cindex]['cid']
                c_name = self.channels_df.iloc[cindex]['name']
                for tindex in range(datas.shape[0]):
                    monitor_time = forecast_time + timedelta(hours= tindex)
                    
                    monitor_value = datas[tindex, pindex, cindex]
                    row_list.append({'uid':uid, 
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


        try:
            result_df = pd.DataFrame(row_list)
            result_df.to_csv(csv_path)
        except Exception as e:
            errors.append(f'grid2PointSliceCsv  save file {csv_path} \
                错误：{e}')


    def plot_newest(self):
        #datas [days,T(168 hours), V, C]  , 每一个day的数据为上一天预测的今天至7天的数据

        fig = plt.figure(figsize=(20,4))
        start_time =max( self.newest_time.index.min(), _truncate_time( datetime.now()-timedelta(days=self.lookback_days)))
        start_time_index = self.newest_time.index.get_loc(start_time)
        end_time = min(self.newest_time.index.max(), _truncate_time(datetime.now())+timedelta(hours=datetime.now().hour+self.nomads.max_fore_hour))
        end_time_index = self.newest_time.index.get_loc(end_time)
        # timeStart = datetime.strptime(timeStart, "%Y-%m-%d")
        hours = end_time_index - start_time_index
        x = [start_time+timedelta(hours=d) for d in range(hours)]


        # error
        y=np.arange(0, self.newest_data.shape[1], 1, dtype=int)
        for row_index,row in tqdm( self.channels_df.iterrows(), total=self.channels_df.shape[0], leave=False):
            channel_index = self.channels_df.index.get_loc(row_index)

            #datas [days,T(168 hours), V, C]  , 每一个day的数据为上一天预测的今天至7天的数据
            plot_datas = self.newest_data[:,:,channel_index] 
            plot_datas = plot_datas[start_time_index:end_time_index,:]
            fig.clf()
            ax=fig.add_axes([0.1,0.1,0.8,0.8],facecolor='black')
            ax.set_title(f"{self.weather_source.name} {row['name']}  data error")
            ax.set_xlabel("date")
            ax.set_ylabel("point" )
            cmap = plt.get_cmap('viridis')
            # cmap.set_bad('black')
            im=ax.pcolor(x,y,plot_datas.swapaxes(0,1),vmin=0,vmax=1)
            fig.colorbar(im, ax=ax, cmap=cmap)
            # plt.xticks(x,rotation = 30)
            fig.savefig(os.path.join(self.plot_dir,f'{self.weather_source.name}_newest_error_c{channel_index}.png'), bbox_inches='tight')
        
        self.wt.plot()

        
if __name__ == '__main__':
    programBegin = datetime.now()
    ap = argparse.ArgumentParser()
    ap.add_argument("-ro", "--data_dir", required=True, help="root")
    ap.add_argument("-ws", "--weather_source", required=True, help="config")
    ap.add_argument("-st", "--source_type", required=True, help="config")
    ap.add_argument("-ns", "--start_date", required=True, help="config")
    ap.add_argument("-lb", "--lookback_days", type=int, required=True, help="config")
    ags, unknowns = ap.parse_known_args()
    arguments = vars(ags)

    data_dir = arguments['data_dir']
    logger = _setup_logger(os.path.join( data_dir,'log') )
    production = Production(data_dir, 
                            SourceType[arguments["source_type"]],
                            SourceType[arguments["weather_source"]],
                            datetime.fromisoformat( arguments["start_date"]),
                            arguments["lookback_days"])
    production.fill_missing_days()
    production.plot_newest()


    production.logger.info(f'{production.weather_source.name} production 用时： {str(datetime.now() - programBegin)}')
