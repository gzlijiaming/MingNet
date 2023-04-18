from datetime import datetime, timedelta, date
import holidays
from chinese_calendar import  is_workday as isCnWorkday
import argparse
import numpy as np
import os, sys
import pandas as pd
sys.path.append("DataPrepare") 
from data_file import DataFile, DataStep,SourceType
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle


class Worktime():
    def __init__(self, data_dir, production_dir, source_type, weather_source,  test_date) :
        self.data_dir = data_dir
        self.production_dir = production_dir
        self.workday_dir = os.path.join(production_dir, 'workday')
        if not os.path.exists(self.workday_dir):
            os.mkdir(self.workday_dir)
        self.source_type = source_type
        self.weather_source = weather_source
        newest_weather_time_path = os.path.join(self.data_dir, f'{self.weather_source.name}_newestWeatherTime.csv')
        self.newest_weather_time = pd.read_csv (newest_weather_time_path, index_col=0,parse_dates=True)
        self.start_date = self.newest_weather_time.index[0]
        self.end_date = self.newest_weather_time.index[-1]+timedelta(hours=-1)
        self.test_date = test_date
        self.us_ca_holidays = holidays.country_holidays('US', subdiv='CA')
        if source_type == SourceType.ca:
            self.is_workday = self.is_ca_workday
        elif source_type == SourceType.guoKong:
            self.is_workday = self.is_cn_workday

        self.days = (self.end_date - self.start_date).days 
        self.data_file =  DataFile(self.data_dir)
        points_file, times_file, channels_file = self.data_file.get_meta_filename(self.source_type)
        self.points_df = pd.read_csv(points_file, index_col=0)
        self.channels_df = pd.read_csv(channels_file, index_col=0)
        self.stat_file = os.path.join(self.workday_dir, f'{self.source_type.name}_worktime_stat.pkl')
        self.channel_count = self.channels_df.shape[0]
        
    def make(self, newest_worktime = None, start_time = None, end_time  = None):
        point_count = self.points_df.shape[0]
        if newest_worktime is None:
            newest_worktime = np.zeros((self.newest_weather_time.shape[0], point_count, self.channel_count),dtype=float)
        if start_time is None:
            start_tindex = 0
        else:
            start_tindex = self.newest_weather_time.index.get_loc(start_time+timedelta(hours=-1))+1
        if end_time is None:
            end_tindex = newest_worktime.shape[0]
        else:
            end_tindex = self.newest_weather_time.index.get_loc(end_time+timedelta(hours=-1))+1
        with open(self.stat_file, 'rb') as f:
            stat_data = pickle.load(f)
        for cindex in range(self.channel_count):
            channel_stat_data = stat_data[f'c{cindex}']
            for tindex in range(start_tindex, end_tindex, 24):
                dindex = tindex //24
                todo_date = self.start_date + timedelta(days=dindex)
                pre_date =  self.start_date + timedelta(days=dindex-1)
                pos_date =  self.start_date + timedelta(days=dindex+1)
                todo_is_workday = 'w' if self.is_workday(todo_date) else 'r'
                pre_is_workday = 'w' if self.is_workday(pre_date) else 'r'
                pos_is_workday = 'w' if self.is_workday(pos_date) else 'r'
                rest_work = pre_is_workday +todo_is_workday + pos_is_workday
                count = min(24, end_tindex-tindex)
                newest_worktime[tindex: tindex + count, :, cindex] = channel_stat_data[rest_work ][0:count,:]
                # for pindex in range(point_count):
                #     newest_worktime[tindex: tindex + count, pindex, cindex] = channel_stat_data[rest_work ][0:count,pindex]
        np.save(os.path.join(self.data_dir, f'{self.source_type.name}_newest_worktime.npy') ,newest_worktime )
        

    def plot(self):
        newsest_worktime = np.load(os.path.join(self.data_dir, f'{self.source_type.name}_newest_worktime.npy')  )
        channel_count = newsest_worktime.shape[2]
        hour_count = newsest_worktime.shape[0]
        x = [self.start_date + timedelta(hours=h) for h in range(hour_count)]
        fig, axs = plt.subplots(channel_count, 1, sharey=False, figsize=(80,10))            
        for cindex in range(channel_count):
            rw_datas = newsest_worktime[:,0,cindex]
            im=axs[cindex].plot(x,rw_datas,label =f"c{cindex}",linewidth=1 )
            # axs[cindex].set_ylim([0, 1])
            axs[cindex].grid()
        fig.savefig(os.path.join(self.data_dir, f'{self.source_type.name}_newest_worktime.png'), bbox_inches='tight')
            
            
    
    def is_ca_workday(self, todoDay ):
        return todoDay.isoweekday() not in [6,7] and todoDay not in self.us_ca_holidays
    
    def is_cn_workday(self, todoDay ):
        return isCnWorkday(todoDay)
    
    def inspect_test_data(self):
        errors = []
        if self.source_type.name == 'guoKong':
            country_name = 'CN'
        elif self.source_type.name == 'ca':
            country_name = 'US'
        else:
            country_name = ''
        src = 'spatial_traeted' #这里不能用src，因为有些点位长期缺数，需要用已经空间插值的
        # fig.clf()
        day_groups = [
            [['w','r'],['w','r'],['w','r'],'123'],
            # [['wr'],   ['w','r'],['wr'],   '-2-'],
            # [['w','r'],['w','r'],['wr'],   '12-'],
            # [['wr'],   ['w','r'],['w','r'],'-23'],
        ]
        stat_data = {}
        for row_index,row in self.channels_df.iterrows():
            channel_index = self.channels_df.index.get_loc(row_index)
            channel_stat_data = {}
            stat_data[f'c{channel_index}'] = channel_stat_data
            # data_file = os.path.join(self.data_dir , f'{self.source_type.name}_{src}_c{channel_index}.npy' )
            # datas = np.load(data_file) # [T, V, C]
            datas = self.data_file.read_data(channel_index, self.source_type, DataStep.normalized)
            
            start_index = self.newest_weather_time.index.get_loc(self.start_date)
            test_index = self.newest_weather_time.index.get_loc(self.test_date+timedelta(hours=-1))+1
            datas = datas[start_index:test_index, :, 0]
            hours = datas.shape[0]
            days = hours//24
            x = [i for i in range(24)]
            if hours>days*24:
                datas = np.delete(datas,range(days*24,hours),axis=0)
                
            avg_datas = datas
            hours = avg_datas.shape[0]
            days = hours//24
            avg_datas = np.reshape(avg_datas,[days,24,datas.shape[1]])
            avg_datas = np.nanmean(avg_datas,axis=0)
            # avg_datas = np.nanmean(avg_datas,axis=1)
                
            sum_data={}
            for day_group in day_groups:
                fig, axs = plt.subplots(1, 1, sharey=False, figsize=(10,5))            
                for yesterday in day_group[0]: 
                    for today in day_group[1]: 
                        for tomorrow in day_group[2]: 
                            restWork =  (yesterday if len(yesterday) == 1 else '-') + today + (tomorrow if len(tomorrow) == 1 else '-')

                            rw_datas = self.rest_work_day(datas, yesterday , today , tomorrow)
                            hours = rw_datas.shape[0]
                            if hours>0:
                                days = hours//24
                                rw_datas = np.reshape(rw_datas,[days,24,datas.shape[1]])
                                rw_datas = np.nanmean(rw_datas,axis=0)
                                # rw_datas = np.nanmean(rw_datas,axis=1)
                            else:
                                days = 0
                                rw_datas = avg_datas
                            im=axs.plot(x,np.nanmean(rw_datas,axis=1),label =f"{restWork} {days}days",linewidth=5 )
                            sum_data[restWork] = np.nanmean(rw_datas)
                            channel_stat_data[restWork] = rw_datas
                                
                sum_data[day_group[3]] = 0
                axs.set_title(f"{self.source_type.name}  24 hour c{channel_index} {row['symbol']} ")
                axs.set_xlabel("hour")
                axs.set_ylabel("value" )
                axs.set_ylim([0, None])
                axs.xaxis.set_ticks(np.arange(min(x), max(x)+1, 2))
                axs.legend()
                axs.grid()
                fig.savefig(os.path.join(self.workday_dir, f'{self.source_type.name}_{day_group[3]}_c{channel_index}.png'), bbox_inches='tight')
            fig, axs = plt.subplots(1, 1, sharey=False, figsize=(10,5))
            names = list(sum_data.keys())
            values = list(sum_data.values())
            axs.set_title(f"{self.source_type.name}  bars c{channel_index} {row['symbol']} ")
            axs.bar(names, values)
            fig.savefig(os.path.join(self.workday_dir, f'{self.source_type.name}_bars_c{channel_index}.png'), bbox_inches='tight')
            with open(self.stat_file, 'wb') as f:
                pickle.dump(stat_data, f)
            # self.statData = statData
        return  errors

    def rest_work_day(self, datas, yesterday, today, tomorrow):
        out_datas = np.empty((0,datas.shape[1]))
        for tindex in range(0,datas.shape[0], 24):
            dindex = tindex // 24
            todo_date = self.start_date + timedelta(days=dindex)
            pre_date = self.start_date + timedelta(days=dindex-1)
            pos_date = self.start_date + timedelta(days=dindex+1)
            todo_is_workday = 'w' if self.is_workday(todo_date) else 'r'
            pre_is_workday = 'w' if self.is_workday(pre_date) else 'r'
            pos_is_workday = 'w' if self.is_workday(pos_date) else 'r'
            
            if pre_is_workday in yesterday and todo_is_workday in today and pos_is_workday in tomorrow:
                out_datas = np.concatenate((out_datas, datas[tindex:tindex+24, :]),axis=0)
        return out_datas
    
    
    
if __name__ == '__main__':
    program_begin = datetime.now()

    ap = argparse.ArgumentParser()
    ap.add_argument("-dp", "--data_dir", required=True,default="data/Air", help="path to the .pkl dataset, e.g. 'data/Dutch/dataset.pkl'")
    ap.add_argument("-src", "--source_type", required=True, type=str,default=1, help="数据源'")
    ap.add_argument("-vs", "--production_dir", default='/datapool01/shared/production', required=False, help="vs model Data dir")
    ap.add_argument("-ws", "--weather_source", required=True, type=str,default=1, help="数据源'")
    ap.add_argument("-ts", "--start", required=True, help="")
    ap.add_argument("-te", "--end", required=True, help="")
    ap.add_argument("-td", "--test_date", required=True, help="")
    ags, unknowns = ap.parse_known_args()
    arguments = vars(ags)

    worktime = Worktime(arguments["data_dir"], 
                    arguments["production_dir"], 
                    SourceType[ arguments["source_type"] ], 
                    SourceType[ arguments["weather_source"] ], 
                    datetime.fromisoformat( arguments["test_date"])
                    )
    worktime.inspect_test_data()
    worktime.make()
    worktime.plot()


    print(f"worktime  用时： {datetime.now() - program_begin}")





