import argparse
import csv
import enum
import json
import logging
import os
import socket
import sys
from datetime import date, datetime, timedelta
from math import ceil
from platform import release

import numpy as np
from aqi_pollutants import *

sys.path.append("DataPrepare") 
import pandas as pd
from data_file import DataFile, DataStep, SourceType

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
    fhlr = logging.FileHandler(os.path.join( log_dir,f'air_stat_{hostname}.log')) # 输出到文件的handler
    fhlr.setFormatter(formatter)
    fhlr.setLevel('DEBUG')
    logger.addHandler(chlr)
    logger.addHandler(fhlr)
    return logger 



def _find_lohi_index(monLohi, minValue, maxValue, cp):
    if cp<minValue or cp > maxValue :
        return -1
    for index, item in enumerate(monLohi):
        if cp <= item:
            return index -1


def _cal_linear(iaqi_lo, iaqi_hi, bp_lo, bp_hi, monValue):
    return (iaqi_hi - iaqi_lo) * (monValue - bp_lo) / (bp_hi - bp_lo) +iaqi_lo

def _get_iaqi(monValue, iaqiLohi, monLohi, minValue, maxValue ):
    lo = _find_lohi_index( monLohi, minValue, maxValue , monValue)
    if lo is None or lo<0:
        return np.nan
    return ceil( _cal_linear(iaqiLohi[lo], iaqiLohi[lo+1], monLohi[lo], monLohi[lo+1], monValue))

# class AqiPollutants(Enum):
#     primary=1526
#     aqi_level=1532
#     aqi_type=1535
#     aqi=1528

# O3_CID = 84
# NO2_CID = 226
# CO_CID = 227
# SO2_CID = 248
# PM10_CID = 796
# PM25_CID = 798

# IAQI_POLLUTANTS = {
#     O3_CID: {'name':'O3分指数','value':1527},
#     NO2_CID:{'name':'NO2分指数','value':1536},
#     CO_CID:{'name':'CO分指数','value':1529},
#     SO2_CID:{'name':'SO2分指数','value':1533},
#     PM10_CID:{'name':'PM10分指数','value':1534},
#     PM25_CID:{'name':'PM25分指数','value':1530},
# }
    

class AirStat():
                
    def __init__(self, data_dir, source_type,  action_time, end_time) :
        self.logger = logging.getLogger()
        self.lohi = {
            'iaqi':[0, 50, 100, 150, 200, 300, 400, 500],
            f'{O3_CID}':[0, 160, 200, 300, 400, 800, 1000, 1200],
            f'{O3_CID}_8hua':[0, 100, 160, 215, 265, 800],
            f'{NO2_CID}':[0, 100, 200, 700, 1200, 2340, 3090, 3840],
            f'{NO2_CID}_24':[0, 40, 80, 180, 280, 565, 750, 940],
            f'{CO_CID}':[0, 5, 10, 35, 60, 90, 120, 150],
            f'{CO_CID}_24':[0, 2, 4, 14, 24, 36, 48, 60],
            f'{SO2_CID}':[0, 150, 500, 650, 800],
            f'{SO2_CID}_24':[0, 50, 150, 475, 800, 1600, 2100, 2620],
            f'{PM10_CID}_24':[0, 50, 150, 250, 350, 420, 500, 600],
            f'{PM25_CID}_24':[0, 35, 75, 115, 150, 250, 350, 500],
        }
        self.action_time = action_time
        self.end_time = end_time
        self.source_type = source_type
        self.data_dir = os.path.join(data_dir, source_type.name)
        self.newest_time_path = os.path.join(self.data_dir, 'newestTime.csv')
        self.upload_dir = os.path.join(data_dir, 'upload', self.source_type.name)
        self.data_file = DataFile( self.data_dir)
        self.metas = self.data_file.read_meta(self.source_type)
        self.points_df = self.metas['points']
        self.channels_df = self.metas['channels']
        self.vget_iaqi = np.vectorize( _get_iaqi,  excluded=(1,2,3,4) )

        C = self.channels_df.shape[0]
        self.newest_data = {}
        self.newest_time = pd.read_csv(self.newest_time_path, index_col=0, parse_dates=[0,1], infer_datetime_format=True)
        self.action_time_index = self.newest_time.index.get_loc(self.action_time)
        end_time_index = self.newest_time.index.get_loc(self.end_time + timedelta(hours=-1))+1
        self.day_start_time = _truncate_time( action_time + timedelta(hours=1))
        self.day_start_time_index = self.newest_time.index.get_loc(self.day_start_time)
        for row_index,row in self.channels_df.iterrows():
            # newestDatas [T, V, C]
            channel_index = self.channels_df.index.get_loc(row_index)
            # 去掉后面无用部分
            self.newest_data[row_index] = np.load(os.path.join(self.data_dir, f'newestData_c{channel_index}.npy'))[0:end_time_index, :, 0]
        self.hour_series = pd.Series( self.newest_time.index).iloc[self.action_time_index+1:end_time_index]
        
    def stat_all(self):
        self.csv_lines = []
        self.city_dfs = []
        self.calc_ozone_8hua()
        self.calc_pm_24hua()
        self.calc_point_day()
        self.calc_ozone_day_max_8hua()
        self.calc_ozone_day_max1()
        self.calc_city_hour()
        self.calc_city_day()
        
        csvPath = os.path.join(self.upload_dir, f'stat_{self.action_time:%Y%m%d_%H}.csv')
        resultDf = pd.DataFrame(self.csv_lines)
        resultDf2 = pd.concat(self.city_dfs)
        resultDf = pd.concat([resultDf, resultDf2])
        resultDf.to_csv(csvPath)
        return csvPath

    def calc_city_day(self):
        iaqi_dfs = []
        tcid = 5
        for row_index,row in self.channels_df.iterrows():
            # newestDatas [T, V, C]
            channel_index = self.channels_df.index.get_loc(row_index)
            csv_lines = self.point_day[row_index]
            city_df = pd.DataFrame(csv_lines)[['region_code', 'region_name', 'monitor_time', 'monitor_value']]
            city_df = city_df.groupby(['region_code', 'region_name', 'monitor_time']).mean()
            self.fill_city_miss_columns(row, city_df, tcid, '城市日平均')
            if row_index not in [O3_CID]:
                self.fill_iaqi(self.lohi[f'{row_index}_24'], city_df,  '城市日IAQI', iaqi_dfs, row_index)
        # O3 8滑最大
        datas = self.ozone_day_max_8hua
        row = self.channels_df.loc[O3_CID]
        row_index = row.name
        csv_lines = []
        self.assemble_csv_lines(datas, row, '城市最大8小时滑动平均', tcid, csv_lines)
        city_df = pd.DataFrame(csv_lines)[['region_code', 'region_name', 'monitor_time', 'monitor_value']]
        city_df = city_df.groupby(['region_code', 'region_name', 'monitor_time']).mean()
        self.fill_city_miss_columns(row, city_df, tcid, '城市最大8小时滑动平均')
        self.fill_iaqi(self.lohi[f'{O3_CID}_8hua'], city_df,  '城市日IAQI', iaqi_dfs, row_index)
        # O3 小时最大
        datas = self.ozone_day_max1
        row = self.channels_df.loc[O3_CID]
        csv_lines = []
        self.assemble_csv_lines(datas, row, '城市最大1小时平均', tcid, csv_lines)
        city_df = pd.DataFrame(csv_lines)[['region_code', 'region_name', 'monitor_time', 'monitor_value']]
        city_df = city_df.groupby(['region_code', 'region_name', 'monitor_time']).mean()
        self.fill_city_miss_columns(row, city_df, tcid, '城市最大1小时平均')
        self.fill_iaqi(self.lohi[f'{O3_CID}'], city_df,  '城市日IAQI', iaqi_dfs, row_index)
       
        self.calc_aqi(iaqi_dfs, tcid, '城市日AQI')

    def calc_city_hour(self):
        iaqi_dfs = []
        tcid = 3
        for row_index,row in self.channels_df.iterrows():
            # newestDatas [T, V, C]
            channelIndex = self.channels_df.index.get_loc(row_index)
            datas = self.newest_data[row_index]
            datas = datas[self.action_time_index+1:, :]
            csv_lines = []
            self.assemble_csv_lines(datas, row, '城市小时平均', tcid, csv_lines)
            city_df = pd.DataFrame(csv_lines)[['region_code', 'region_name', 'monitor_time', 'monitor_value']]
            city_df = city_df.groupby(['region_code', 'region_name', 'monitor_time']).mean()
            self.fill_city_miss_columns(row, city_df, tcid, '城市小时平均')
            if row_index not in [PM25_CID, PM10_CID]:
                self.fill_iaqi(self.lohi[f'{row_index}'], city_df,  '城市小时IAQI', iaqi_dfs, row_index)
        # pm10 24滑动
        datas = self.Pm10_24Hua
        row = self.channels_df.loc[PM10_CID]
        row_index = row.name
        csv_lines = []
        self.assemble_csv_lines(datas, row, '城市24小时滑动平均', tcid, csv_lines)
        city_df = pd.DataFrame(csv_lines)[['region_code', 'region_name', 'monitor_time', 'monitor_value']]
        city_df = city_df.groupby(['region_code', 'region_name', 'monitor_time']).mean()
        self.fill_city_miss_columns(row, city_df, tcid, '城市24小时滑动平均')
        self.fill_iaqi(self.lohi[f'{row_index}_24'], city_df,  '城市小时IAQI', iaqi_dfs, row_index)
        # pm2.5 24滑动
        datas = self.Pm25_24Hua
        row = self.channels_df.loc[PM25_CID]
        row_index = row.name
        csv_lines = []
        self.assemble_csv_lines(datas, row, '城市24小时滑动平均', tcid, csv_lines)
        city_df = pd.DataFrame(csv_lines)[['region_code', 'region_name', 'monitor_time', 'monitor_value']]
        city_df = city_df.groupby(['region_code', 'region_name', 'monitor_time']).mean()
        self.fill_city_miss_columns(row, city_df, tcid, '城市24小时滑动平均')
        self.fill_iaqi(self.lohi[f'{row_index}_24'], city_df,  '城市小时IAQI', iaqi_dfs, row_index)
        # O3 8滑动
        datas = self.ozone_8hua
        row = self.channels_df.loc[O3_CID]
        row_index = row.name
        csv_lines = []
        self.assemble_csv_lines(datas, row, '城市8小时滑动平均', tcid, csv_lines)
        city_df = pd.DataFrame(csv_lines)[['region_code', 'region_name', 'monitor_time', 'monitor_value']]
        city_df = city_df.groupby(['region_code', 'region_name', 'monitor_time']).mean()
        self.fill_city_miss_columns(row, city_df, tcid, '城市8小时滑动平均')
        self.fill_iaqi(self.lohi[f'{row_index}_8hua'], city_df,  '城市小时IAQI', iaqi_dfs, row_index)
        
        self.calc_aqi(iaqi_dfs, tcid, '城市小时AQI')
        
    
    def calc_aqi(self, iaqi_dfs, tcid, stat_name):
        iaqi_df = pd.concat(iaqi_dfs)
        ## iaqi自己也要加进去结果
        self.city_dfs.append(iaqi_df)
        
        iaqi_df = iaqi_df[['region_code', 'region_name', 'monitor_time', 'monitor_value', 'c_name']]
        group = iaqi_df.groupby(['region_code', 'region_name', 'monitor_time'], as_index =False)
        csv_lines = []
        result_df = group.apply(self.find_max_head, csv_lines)
        city_df = pd.DataFrame(csv_lines)
        city_df.set_index(['region_code', 'region_name', 'monitor_time'],inplace=True)
        self.fill_city_miss_columns(None, city_df, tcid, stat_name)
        
    def find_max_head(self, df, csv_lines):
        max_value = df['monitor_value'].max()
        csv_lines.append({'region_code':df.iloc[0]['region_code'], 'region_name':df.iloc[0]['region_name'], 'monitor_time':df.iloc[0]['monitor_time'],
                         'c_name':AqiPollutants.aqi.name,'cid':AqiPollutants.aqi.value, 'monitor_value':max_value})
        if max_value>50:
            name_list = df[df['monitor_value'] == max_value]['c_name']
            csv_lines.append({'region_code':df.iloc[0]['region_code'], 'region_name':df.iloc[0]['region_name'], 'monitor_time':df.iloc[0]['monitor_time'],
                             'c_name':'首要污染物','cid':AqiPollutants.primary.value, 'monitor_value':','.join(name_list.to_list()).replace('分指数','')})
        return max_value
    
    def fill_iaqi(self, mon_lohi, city_df,  stat_name, aqi_dfs, cid):
        iaqi_pollutant = IAQI_POLLUTANTS[cid]
        aqi_df = city_df.copy()
        iaqi_lohi = self.lohi['iaqi']
        min_value = min( mon_lohi )
        max_value = max( mon_lohi )
        for i in range(aqi_df.shape[0]):
            aqi_df.at[i,'monitor_value'] = _get_iaqi( aqi_df.at[i,'monitor_value'], iaqi_lohi, mon_lohi, min_value, max_value )
            aqi_df.at[i,'cid'] = iaqi_pollutant['value']
            aqi_df.at[i,'c_name'] = iaqi_pollutant['name']
        aqi_df['stat_name'] = stat_name
        aqi_dfs.append(aqi_df)

    def fill_city_miss_columns(self, channel, city_df, tcid, stat_name):
        city_df.reset_index(inplace=True)
        city_df['uid'] = 0
        city_df['name'] = city_df['region_name']
        city_df['tcid'] = tcid
        city_df['data_type']=SourceType.guoKong.value
        city_df['data_type_name'] = SourceType.guoKong.name
        if channel is not None:
            city_df['c_name'] = str( channel['symbol'])
            city_df['cid'] = str( channel.name)
        city_df['forecast_time'] = self.action_time
        city_df['stat_name'] = stat_name
        self.city_dfs.append(city_df)

    def calc_ozone_day_max1(self):
        tcid = 5
        datas = self.newest_data[O3_CID][self.day_start_time_index:, :]
        hours = datas.shape[0]
        days = hours//24
        if hours>days*24:
            datas = np.delete(datas,range(days*24,hours),axis=0)
        datas = np.reshape(datas,[days,24,datas.shape[1]])
        datas = np.nanmax(datas,axis=1)
        self.ozone_day_max1 = datas
        self.assemble_csv_lines(datas, self.channels_df.loc[O3_CID], '点位臭氧日最大小时值', tcid)
        
        
    def calc_ozone_day_max_8hua(self):
        tcid = 5
        datas = self.ozone_8hua.copy()
        hours = datas.shape[0]
        days = hours//24
        if hours>days*24:
            datas = np.delete(datas,range(days*24,hours),axis=0)
        datas = np.reshape(datas,[days,24,datas.shape[1]])
        datas[:,0:7,:] = 0
        datas = np.nanmax(datas,axis=1)
        self.ozone_day_max_8hua = datas
        self.assemble_csv_lines(datas, self.channels_df.loc[O3_CID], '点位臭氧日最大8小时滑动平均', tcid)
    
    def calc_point_day(self):
        self.point_day = {}
        tcid = 5
        for rowIndex,row in self.channels_df.iterrows():
            # newestDatas [T, V, C]
            channelIndex = self.channels_df.index.get_loc(rowIndex)
            datas = self.newest_data[rowIndex]
            datas = datas[self.day_start_time_index:, :]
            hours = datas.shape[0]
            days = hours//24
            if hours>days*24:
                datas = np.delete(datas,range(days*24,hours),axis=0)
            datas = np.reshape(datas,[days,24,datas.shape[1]])
            datas = np.nanmean(datas,axis=1)
            csvLines = []
            self.assemble_csv_lines(datas, row, '点位日平均', tcid, csvLines)
            self.point_day[rowIndex] = csvLines
            self.csv_lines += csvLines


    def calc_pm_24hua(self):
        tcid = 3
        datas = self.newest_data[PM10_CID]
        datas = self.get_hua(24, datas[self.action_time_index-(24-2):, :])
        self.Pm10_24Hua = datas
        csv_lines = []
        self.assemble_csv_lines(datas, self.channels_df.loc[PM10_CID], '点位PM10 24小时滑动平均', tcid, csv_lines)
        self.csv_lines += csv_lines

        datas = self.newest_data[PM25_CID]
        datas = self.get_hua(24, datas[self.action_time_index-(24-2):, :])
        self.Pm25_24Hua = datas
        csv_lines = []
        self.assemble_csv_lines(datas, self.channels_df.loc[PM25_CID], '点位PM2.5 24小时滑动平均', tcid, csv_lines)
        self.csv_lines += csv_lines
    
    
    def calc_ozone_8hua(self):
        datas = self.newest_data[O3_CID]
        datas = self.get_hua(8, datas[self.action_time_index-(8-2):, :])
        self.ozone_8hua = datas
        csvLines = []
        tcid = 3
        self.assemble_csv_lines(datas, self.channels_df.loc[O3_CID], '点位臭氧8小时滑动平均', tcid, csvLines)
        self.csv_lines += csvLines
        
    
    def assemble_csv_lines(self,datas,channel, statName, tcid, csv_lines = None):
        if csv_lines is None:
            csv_lines = self.csv_lines
        cid = str( channel.name)
        data_type=SourceType.guoKong.value
        data_type_name = SourceType.guoKong.name
        c_name = str( channel['symbol'])
        for pindex in range(self.points_df.shape[0]):
            point = self.points_df.iloc[pindex]
            uid = point['vid']
            name = point['name']
            region_code = point['region_code']
            region_name = point['region_name']
            for tIndex in range(datas.shape[0]):
                if tcid == 3:
                    monitor_time = self.action_time + timedelta(hours= tIndex+1)
                else:
                    monitor_time = self.day_start_time + timedelta(days= tIndex)
                monitor_value = datas[tIndex, pindex]
                csv_lines.append({'uid':uid, 
                                'name':name, 
                                'region_code':region_code,
                                'region_name':region_name,
                                'forecast_time':self.action_time, 
                                'tcid':tcid, 
                                'data_type':data_type, 
                                'data_type_name':data_type_name, 
                                'monitor_time':monitor_time,
                                'cid':cid, 
                                'c_name':c_name,
                                'monitor_value':monitor_value,
                                'stat_name':statName})

        
    def get_iaqis(self, cid, datas):
        iaqi_lohi = self.lohi['iaqi']
        mon_lohi = self.lohi[cid]
        min_value = min( mon_lohi )
        max_value = max( mon_lohi )
        return self.vget_iaqi( datas, iaqi_lohi, mon_lohi, min_value, max_value )
    
    def get_hua(self, num, datas):
        result = np.empty((datas.shape[0]-(num-1), datas.shape[1]), dtype=float)
        kernel = self.get_kernel(num)
        for pindex in range(datas.shape[1]):
            result[:,pindex] = np.convolve(datas[:,pindex], kernel,mode='valid')
        return result

    def get_kernel(self, num):
        return np.full(num,1/num)
        
        
if __name__ == '__main__':
    program_begin = datetime.now()
    ap = argparse.ArgumentParser()
    ap.add_argument("-ro", "--data_dir", required=True, help="root")
    ap.add_argument("-st", "--source_type", required=True, help="config")
    # ap.add_argument("-lb", "--lookback_days", type=int, required=True, help="config")
    # ap.add_argument("-ns", "--start_date", required=True, help="config")
    ags, unknowns = ap.parse_known_args()
    arguments = vars(ags)

    data_dir = arguments['data_dir']
    logger = _setup_logger(os.path.join( data_dir,'log') )
    airstat = AirStat(data_dir, SourceType[arguments["source_type"]], datetime(2022,9,17,23), datetime(2022,9,25,0))
    airstat.stat_all()
    airstat.logger.info(f'air stat 用时： {str(datetime.now() - program_begin)}')
    