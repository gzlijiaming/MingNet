# Resolve the latest GFS dataset
# https://www.pydap.org/en/latest/client.html
# https://rda.ucar.edu/thredds/catalog/catalog_ds084.1.html
# https://rda.ucar.edu/thredds/dodsC/files/g/ds084.1/2022/20220104/gfs.0p25.2022010400.f030.grib2.html
# https://rda.ucar.edu/thredds/dodsC/files/g/ds084.1/2022/20220104/gfs.0p25.2022010400.f030.grib2

from datetime import datetime, timedelta, date
import numpy as np
import socket
import logging
import os, sys
import requests
import atexit
import json
from pydap.client import open_url
import argparse

                
_MAX_RETRY_TIMES = 10
_ONE_DAY = timedelta(days = 1)

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
    fhlr = logging.FileHandler(os.path.join( log_dir,f'rda-tds_{hostname}.log')) # 输出到文件的handler
    fhlr.setFormatter(formatter)
    fhlr.setLevel('DEBUG')
    logger.addHandler(chlr)
    logger.addHandler(fhlr)
    return logger 



class RdaDownloader():
    def __init__(self,arguments):
        weather_source = arguments["weather_source"]
        gfs_root_dir = arguments['gfs_root_dir']
        dirname, filename = os.path.split(os.path.abspath(__file__))
        with open(os.path.join(dirname, f'rda_tds_{weather_source}.json'), 'r', encoding='utf-8') as f:
            config = json.load(f)
        self.start_date = date.fromisoformat(arguments['start']) +timedelta(days=-1)  #找埋前一天，因为前一天的预测是今天
        self.end_date = date.fromisoformat(arguments['end'])
        self.base_url = 'https://rda.ucar.edu/thredds/dodsC/files/g/ds084.1/'
        gfs_dir = os.path.join(gfs_root_dir, weather_source)
        lon_lat = config['gridLonLat']
        self.channel_inv_names = [var[0] for var in config['vars']]
        point_size = config['pointSize']
        self.max_fore_hour = config['maxForeHour']# 24*7+4  # 从0到144小时，即一星期，加4是因为前面第一个小时是20点，还只是今天，明天凌晨要4小时后
        self.inter_hours = config['interHours']


        # maxDate = date(2019,3,1)
        self.model_cycle_runtimes = config['model_cycle_runtimes']

        self.block={'y':{'start':int((90-lon_lat['north'])/point_size),
                    'end': int((90-lon_lat['south'])/point_size)+1},
                'x': {'start':int((lon_lat['west'])/point_size),
                    'end': int((lon_lat['east'])/point_size)+1}}
        self.lon_count = self.block['x']['end']-self.block['x']['start']
        self.lat_count = self.block['y']['end']-self.block['y']['start']
        self.slice_dir = os.path.join(gfs_dir,'rda_slice')
        if not os.path.exists(self.slice_dir):
            os.mkdir(self.slice_dir)

    def print_vars(self):
        out_names=[]
        with  open("rda-tds-vars.txt", "w") as text_file :
            for (key,var) in self.channel_inv_names.items():
                if var.ndim >=3 :
                    out_names.append(var.name)
                    text_file.write(f'{var.name}\t{var.long_name}\t{var.units}\t{var.ndim}\t{var.shape}\n')
        return out_names



    def get_one_day(self, todo_date):
        errors = []
        fhour=-1
        var_name=''
        logger.info(f'Start! get_one_day {todo_date}')
        slice_path =os.path.join(self.slice_dir, f"{todo_date.strftime('%Y%m%d')}_{self.model_cycle_runtimes}.npy")
        if os.path.exists(slice_path) and os.path.getsize(slice_path)>1024*1024:   #文件已经有,而且大过1M，就不用再下载了
            return errors
        datas =np.empty([1, self.max_fore_hour//self.inter_hours+1,self.lon_count, self.lat_count,len(self.channel_inv_names)],dtype=float) #[day, foreHour/3,lon, lat, channel]
        try:
            session = requests.Session()
            # session.verify = False
            # session.proxies = {
            #     # 'http': 'socks5://172.21.11.24:1080', 'https': 'socks5://172.21.11.24:1080',
            #     'http': 'socks5://192.168.1.93:1080', 'https': 'socks5://192.168.1.93:1080',
            #     }
            for fhour in range(0,self.max_fore_hour+1,self.inter_hours):
                # catUrl = 'https://rda.ucar.edu/thredds/dodsC/files/g/ds084.1/2022/20220104/gfs.0p25.2022010400.f030.grib2'
                cat_url = f"{self.base_url}{todo_date.strftime('%Y')}/{todo_date.strftime('%Y%m%d')}/gfs.0p25.{todo_date.strftime('%Y%m%d')}{model_cycle_runtimes}.f{fhour:03}.grib2"
                retry_times1 = 0
                while retry_times1< _MAX_RETRY_TIMES:
                    try:
                        retry_times1 += 1
                        dataset = open_url(cat_url, timeout=300, session=session)
                        break
                    except Exception as e:
                        err_msg = f'Error! open_url : {e} \
                            hisDate={todo_date} url={cat_url} \
                            fHour={fhour}'
                        logger.info(err_msg)
                if retry_times1 == _MAX_RETRY_TIMES:
                    errors.append(err_msg)
                    return errors
                # self.channelInvNames = printVars(dataset.variables)

                for varIndex, var_name in enumerate( self.channel_inv_names):
                    retry_times2 = 0
                    while retry_times2< _MAX_RETRY_TIMES:
                        try:
                            retry_times2 += 1
                            remote_grid = dataset[var_name]
                            ndim = len(remote_grid.shape)
                            # print(varName)
                            if ndim == 3:
                                block_data=np.array(remote_grid[0,self.block['y']['start']:self.block['y']['end'], self.block['x']['start']:self.block['x']['end']].array[0])
                            elif ndim == 4:
                                block_data=np.array(remote_grid[0,0,self.block['y']['start']:self.block['y']['end'], self.block['x']['start']:self.block['x']['end']].array[0,0])
                            else:
                                raise ValueError(f"{var_name} 's ndim is {ndim}, only 3/4 are allowed!")

                            datas[0,fhour//self.inter_hours,:,:,varIndex]=block_data.T #[day, preHour/3,lon, lat, channel]
                            logger.info(f'OK! getOneDay {todo_date} : fHour={fhour} varIndex={varIndex} retryTimes={retry_times2}')
                            break
                        except Exception as e:
                            err_msg = f'Error! self.gdBlock : {e} \
                                hisDate={todo_date} url={cat_url} \
                                fHour={fhour} varName: {var_name}'
                            logger.info(err_msg)
                    if retry_times2 == _MAX_RETRY_TIMES:
                        errors.append(err_msg)
                        return errors

        except Exception as e:
            err_msg = f'Error! getOneDay: {e} \
                hisDate={todo_date} url={cat_url} \
                fHour={fhour} varName: {var_name}'
            errors.append(err_msg)
            return errors

        if not errors and not os.path.exists(slice_path):   #文件已经有就不用再保存了，应该是阻塞的过程中，有别人做了
            np.save(slice_path,datas)     #[day(0), foreHour/3,lon, lat, channel]
        return  errors


    def get_range(self):
        todo_date = self.start_date
        while todo_date <= self.end_date:
            self.get_one_day(todo_date)
            todo_date += _ONE_DAY


    def find_missing(self):
        todo_date = self.start_date
        missings = []
        while todo_date <= self.end_date:
            slice_path =os.path.join(self.slice_dir, f"{todo_date.strftime('%Y%m%d')}_{self.model_cycle_runtimes}.npy")
            #文件已经有,而且大过1M，就不用再下载了
            if not os.path.exists(slice_path) or os.path.getsize(slice_path) <= 1024 * 1024:
                todo_string = todo_date.strftime('%Y-%m-%d')
                missings.append(f'python3 rda_dap.py --action download --start {todo_string} --end {todo_string} --weather_source {weather_source} --gfs_root_dir /home/ming/ucar --config_file rda_tds_{weather_source}.json')
            todo_date += _ONE_DAY
        for missing in missings:
            print(missing)
        print(f'total {len(missings)} missed')





if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-st", "--start", required=True, help="start")
    ap.add_argument("-en", "--end", required=True, help="end")
    ap.add_argument("-ro", "--gfs_root_dir", required=True, help="root")
    ap.add_argument("-ws", "--weather_source", required=True, help="config")
    ap.add_argument("-ac", "--action", required=False, default='find', help="config")
    args, unknowns = ap.parse_known_args()
    arguments = vars(args)

    

    logger = _setup_logger(arguments['gfs_root_dir'] )
    start = datetime.now()
    atexit.register(lambda: logger.info( f"用时 :{ datetime.now()-start }" ))


    downloader = RdaDownloader(arguments)
    if arguments["action"] == 'download':
        downloader.get_range()
    else:
        downloader.find_missing()



