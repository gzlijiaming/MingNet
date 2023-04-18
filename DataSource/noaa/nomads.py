# 注意pydap必须更新到3.3.0，需要用conda安装，因为pip只更新到3.2.2， 而3.2.2版本不能解释gzip，会出decoding error
# ./ps.sh "/home/ming/anaconda3/condabin/conda install -n air pydap -y"
# Resolve the latest GFS dataset
# https://www.pydap.org/en/latest/client.html
# https://nomads.ncep.noaa.gov/dods/gfs_0p25/gfs20220826/gfs_0p25_00z.info

from datetime import datetime, timedelta, date
from platform import release
import numpy as np
import socket
import logging
import os, sys
import requests
import atexit
import json
from pydap.client import open_url
import argparse
import pandas as pd
sys.path.append("DataPrepare") 
from data_file import DataFile, DataStep,SourceType


_ONE_DAY = timedelta(days = 1)
_MAX_RETRY_TIMES = 10
_TIMEOUT_SECONDS = 5*60
_MISSING_VALUE =  9.9E20 # remote value is 9.999E20

def _setup_logger(log_dir):
    logger = logging.getLogger()
    logger.setLevel('INFO')
    BASIC_FORMAT = "%(asctime)s\t%(message)s"
    FULL_FORMAT = "%(asctime)s:%(levelname)s\t%(message)s"
    BASIC_DATE_FORMAT = '%H:%M:%S'
    DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(BASIC_FORMAT, BASIC_DATE_FORMAT)
    chlr = logging.StreamHandler() # 输出到控制台的handler
    chlr.setFormatter(formatter)
    chlr.setLevel('INFO')  # 也可以不设置，不设置就默认用logger的level
    formatter = logging.Formatter(FULL_FORMAT, DATE_FORMAT)
    hostname=socket.gethostname()
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    fhlr = logging.FileHandler(os.path.join( log_dir,f'nomads_{hostname}.log')) # 输出到文件的handler
    fhlr.setFormatter(formatter)
    fhlr.setLevel('DEBUG')
    logger.addHandler(chlr)
    logger.addHandler(fhlr)
    return logger

def _print_vars(channel_inv_names):
    out_names=[]
    with  open("rda-tds-vars.txt", "w") as text_file :
        for (key,var) in channel_inv_names.items():
            if var.ndim >=3 :
                out_names.append(var.name)
                text_file.write(f'{var.name}\t{var.long_name}\t{var.units}\t{var.ndim}\t{var.shape}\n')
    return out_names


class Nomads():
                
    def __init__(self,data_dir, weather_source, lookback_days) :
        self.lookback_days  = lookback_days
        config_file = f'nomads_{weather_source.name}.json'
        dirname, filename = os.path.split(os.path.abspath(__file__))
        with open(os.path.join(dirname,config_file), 'r', encoding='utf-8') as f:
           self.config = json.load(f)
        gfs_dir = os.path.join(data_dir, weather_source.name)
        if not os.path.exists(gfs_dir):
            os.mkdir(gfs_dir, )
        lon_lat =self.config['gridLonLat']
        self.channel_inv_names = [var[0] for var in self.config['vars']]
        pointSize =self.config['pointSize']
        self.max_fore_hour =self.config['maxForeHour']# 24*7+4  # 从0到144小时，即一星期，加4是因为前面第一个小时是20点，还只是今天，明天凌晨要4小时后
        self.services = self.config['services']
        # maxDate = date(2019,3,1)
        # model_cycle_runtimes ='12' #UTC的12点，相当于北京时间20点，是23：50之前最近的一个时间
        # model_cycle_runtimes ='06' #UTC的06点，相当于加州时间22点，是23：50之前最近的一个时间
        # self.model_cycle_runtimes =self.config['model_cycle_runtimes']
        self.timezone = self.config['timeZone']
        
        self.block={'y':{'start':int((90+lon_lat['south'])/pointSize),
                            'end': int((90+lon_lat['north'])/pointSize)+1},
                    'x': {'start':int((lon_lat['west'])/pointSize),
                            'end': int((lon_lat['east'])/pointSize)+1}}
        self.lon_count = self.block['x']['end']-self.block['x']['start']
        self.lat_count = self.block['y']['end']-self.block['y']['start']
        self.slice_dir = os.path.join(gfs_dir,'rda_slice')
        if not os.path.exists(self.slice_dir):
            os.mkdir(self.slice_dir, )
        self.today_begin_time = self.truncate_time( datetime.utcnow())
        self.logger = logging.getLogger()

    def get_one_day(self, todo_date):
        errors = []
        fhour=-1
        var_name=''
        self.logger.info(f'Start! getOneDay {todo_date}')
        session = requests.Session()
        try:
            datasets, model_cycle_runtime, done_before = self.get_dataset(session, todo_date)
            if done_before:
                # dataset is filename
                self.logger.info(f'{datasets} was downloaded before!')
                return datasets, errors
            if datasets is None:
                errors.append('can not find dataset')
                return None, errors
            
            slice_name = f"{todo_date.strftime('%Y%m%d')}_{model_cycle_runtime}.npy"
            slice_path =os.path.join(self.slice_dir, slice_name)
            time_count= self.max_fore_hour
            datas =np.empty([1,time_count ,self.lon_count, self.lat_count,len(self.channel_inv_names)],dtype=float) #[day, foreHour/3,lon, lat, channel]
            # channelInvNames = printVars(dataset.variables)
            for sindex , service in enumerate( self.services):
                dataset = datasets[sindex]
                
                for var_index, var_name in enumerate( self.channel_inv_names):
                    # if varIndex<7:    #for debug
                    #     continue
                    retry_times = 0
                    while retry_times< _MAX_RETRY_TIMES:
                        try:
                            retry_times += 1
                            remote_grid = dataset[var_name]
                            ndim = len(remote_grid.shape)
                            # print(varName)
                            if ndim == 3:
                                block_data=np.array(remote_grid[service['tStartIndex']:service['tEndIndex'],
                                                              self.block['y']['start']:self.block['y']['end'], self.block['x']['start']:self.block['x']['end']])
                            elif ndim == 4:
                                block_data=np.array(remote_grid[service['tStartIndex']:service['tEndIndex'],
                                                              0,self.block['y']['start']:self.block['y']['end'], self.block['x']['start']:self.block['x']['end']])
                            else:
                                raise ValueError(f"{var_name} 's ndim is {ndim}, only 3/4 are allowed!")
                            # blockData[blockData>missing_value] = np.nan     # missing_value: 9.999E20
                            block_data = np.flip(block_data, axis=1)
                            block_data = block_data.transpose(0,2,1)
                            for tindex in range(service['tStartIndex'],service['tEndIndex']):
                                for hindex in range(service['interHours']):
                                    datas[0,tindex * service['interHours'] +hindex ,:,:,var_index]= \
                                        block_data[tindex - service['tStartIndex'], :, :]#[day, preHour,lon, lat, channel]
                            self.logger.info(f'OK! getOneDay {todo_date} : var{var_index}:{var_name}  retryTimes={retry_times}')
                            break
                        except Exception as e:
                            err_msg = f'Error! gdBlock : {e} \
                                hisDate={todo_date} \
                                fHour={fhour} varName: {var_name}'
                            self.logger.info(err_msg)
                    if retry_times == _MAX_RETRY_TIMES:
                        errors.append(err_msg)
                        return None, errors

        except Exception as e:
            err_msg = f'Error! getOneDay: {e} \
                hisDate={todo_date} \
                fHour={fhour} varName: {var_name}'
            errors.append(err_msg)
            return None, errors

        if not errors and not os.path.exists(slice_path):   #文件已经有就不用再保存了，应该是阻塞的过程中，有别人做了
            np.save(slice_path,datas)     #[day(0), foreHour/3,lon, lat, channel]
        return  slice_name, errors

    def get_dataset(self, session, todo_date):
        for model_cycle_runtime in ['18','12','06','00']:
            slice_name = f"{todo_date.strftime('%Y%m%d')}_{model_cycle_runtime}.npy"
            slice_path =os.path.join(self.slice_dir, slice_name)
            if os.path.exists(slice_path) and os.path.getsize(slice_path)>1024*1024:   #文件已经有,而且大过1M，就不用再下载了
                return slice_name, None, True

            issue_time = todo_date + timedelta(hours= self.timezone+int(model_cycle_runtime))
            if issue_time< datetime.now():
                datasets = []
                for service in self.services:
                    cat_url = service['baseUrl'] + f"gfs{todo_date.strftime('%Y%m%d')}/{service['name']}_{model_cycle_runtime}z"
                    try:
                        dataset = open_url(cat_url, timeout=300, session=session, output_grid=False)
                        datasets.append(dataset)
                    except Exception as e:
                        err_msg = f'Error! open_url : {e} \
                            hisDate={todo_date} url={cat_url}'
                        self.logger.info(err_msg)
                if len(datasets) == len(self.services): 
                    return datasets, model_cycle_runtime, False
        return None, None, False

    def get_todo_date(self, todo_date = None):
        release_time = self.getReleaseTime( self.today_begin_time )
        max_todo_date = self.today_begin_time
        if datetime.now()<=release_time:
            max_todo_date += timedelta(days=-1)
        if todo_date is None:
            todo_date = datetime.now() 
        todo_date = self.truncate_time(todo_date)
        return min(todo_date, max_todo_date)

    # def getReleaseTime(self, dayBeginTime):
    #     return dayBeginTime+ timedelta(hours=self.timeZone+int( self.model_cycle_runtimes))

    def truncate_time(self, dt):
        return dt.replace(hour=0, minute=0, second=0, microsecond=0)





if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-ro", "--data_dir", required=True, help="root")
    ap.add_argument("-ws", "--weather_source", required=False, help="config")
    ap.add_argument("-sd", "--start_date", required=False, help="config")
    ap.add_argument("-lb", "--lookback_days", type=int, required=True, help="config")
    args, unknowns = ap.parse_known_args()
    arguments = vars(args)

    data_dir = arguments['data_dir']
    logger = _setup_logger(os.path.join( data_dir,'log') )
    start = datetime.now()
    atexit.register(lambda: logger.info( f"用时 :{ datetime.now()-start }" ))


    nomads = Nomads(data_dir,  SourceType[arguments["weather_source"]],
                            arguments["lookback_days"])
    # 测试采集一天
    if arguments['start_date'] is None:
        start_date = None
    else:
        start_date = datetime.fromisoformat(arguments['start_date']) 
    todoDate = nomads.get_todo_date(start_date)
    nomads.get_one_day(start_date)

