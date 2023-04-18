import argparse
import logging
import socket
import pandas as pd
from train_model import AirNetTrainer
import os, sys
sys.path.append("DataPrepare") 
from data_file import DataFile, DataStep, SourceType
from datetime import datetime, timedelta,time
from test_model import Tester

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
    fhlr = logging.FileHandler(os.path.join( log_dir,f'train_{hostname}.log')) # 输出到文件的handler
    fhlr.setFormatter(formatter)
    fhlr.setLevel('DEBUG')
    logger.addHandler(chlr)
    logger.addHandler(fhlr)
    return logger 


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-dp", "--data_dir", required=True,default="data/Air", help="path to the .pkl dataset, e.g. 'data/Dutch/dataset.pkl'")
    ap.add_argument("-src", "--source_type", required=True, type=str,default=1, help="数据源'")
    ap.add_argument("-ws", "--weather_source", required=False, help="config")
    ap.add_argument("-e", "--epochs", type=int, default=50, help="Number of epochs")
    ap.add_argument("-c", "--channel_index", required=False,  type=int,default=0, help="channel Index'")
    ap.add_argument("-vs", "--vs_dir", default='/datapool01/shared/vs', required=False, help="vs model Data dir")
    ap.add_argument("-md", "--model_dir", default='', required=False, help="vs model Data dir")
    ap.add_argument("-ml", "--model_layers", default='', type=str, required=False, help="vs model Data dir")
    ap.add_argument("-td", "--train_date", required=True, help="")
    ap.add_argument("-ts", "--test_date", required=True, help="")
    ap.add_argument("-ed", "--end_date", required=True, help="")
    ap.add_argument("-pg", "--param_groud_id", required=False,  type=int,default=1, help="param_groud_id")
    ags, unknowns = ap.parse_known_args()
    arguments = vars(ags)
    
    logger = _setup_logger(os.path.join( arguments["data_dir"],'log') )
    
    
    best_param = pd.read_excel(os.path.join(arguments["data_dir"], f'{arguments["source_type"]}_best_param.xlsx'))
    best_param = best_param.loc[best_param['param_group_id'] == arguments["param_groud_id"]]
    channel_index = arguments["channel_index"] 
    channel_param = best_param.loc[best_param['c_index'] == channel_index]
    for stepsIndex, param_row in channel_param.iterrows():
        steps = param_row['predict_timesteps'].split(',')
        steps = [int(i) for i in steps]
        batch_size = int( param_row['batch_size'])
        input_predays = int(param_row['input_predays'])
        base_timesteps = int(param_row['base_timesteps'])
        channel_roll = int(param_row['channel_roll'])
        model_layers = param_row['model_layers']


        airnet = AirNetTrainer(model_layers,
                        arguments["data_dir"], arguments, 
                        SourceType[ arguments["source_type"] ], SourceType[ arguments["weather_source"] ], 
                        arguments["epochs"], 
                        param_row["predict_timesteps"], base_timesteps, 
                        batch_size,
                        input_predays, channel_index, channel_roll,
                        arguments["model_dir"], 
                        datetime.fromisoformat( arguments["train_date"]),
                        datetime.fromisoformat( arguments["test_date"]),
                        datetime.fromisoformat( arguments["end_date"])
                        )
        train_time, last_epoch, used_memory, model_filename = airnet.train()
        # trainTime,lastEpoch, usedMemory, modelFileName = '08:08:38.88888',  88, 8888, None

        tester = Tester(arguments["data_dir"], arguments, 
                        SourceType[ arguments["source_type"] ], SourceType[ arguments["weather_source"] ], 
                        arguments["vs_dir"], channel_index,
                        arguments["model_dir"], 
                        datetime.fromisoformat( arguments["train_date"]),
                        datetime.fromisoformat( arguments["test_date"]),
                        datetime.fromisoformat( arguments["end_date"])
                        )
        steps, results = tester.test_one(model_layers, param_row["predict_timesteps"], base_timesteps,
                        batch_size, input_predays,
                        channel_index, channel_roll, train_time, last_epoch, used_memory, model_filename)
        airnet.log_train_result(train_time, last_epoch, used_memory, model_filename, steps, results)