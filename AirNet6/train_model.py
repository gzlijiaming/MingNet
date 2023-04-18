import argparse
import os
import sys
import time
from calendar import c
from datetime import datetime, timedelta
from fileinput import filename
from inspect import isclass

import torch
import torch.nn.functional as F
from numpy import double
from scipy.io import loadmat
from torch import optim, true_divide
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, TensorDataset

sys.path.append("DataPrepare") 
from data_file import DataFile, DataStep, SourceType
import logging
import re
import socket
import subprocess

import test_model
from air_dataloader import *
from airnet_model import Model
from test_model import Tester
from torch.utils.tensorboard import SummaryWriter
import pandas as pd


_SLEEP_SECONDS = 10
_MAX_TRAIN_ID = 9999
# stepsList = [[3,6,12,24],  [48,72,120,168]]


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


class AirNetTrainer():
    def __init__(self, model_layers, data_dir, arguments, source_type, weather_source, epochs, predict_timesteps,base_timesteps, batch_size,input_predays,channel_index, channel_roll, model_dir, train_date = None, test_date = None, end_date = None):

        self.hostname=socket.gethostname()
        self.logger = logging.getLogger()
        self.data_dir = data_dir
        self.source_type = source_type
        self.weather_source = weather_source
        self.epochs = epochs
        self.predict_timesteps = predict_timesteps
        self.base_timesteps = base_timesteps
        self.batch_size = batch_size
        self.input_predays = input_predays
        self.channel_index = channel_index
        self.channel_roll = channel_roll
        self.train_date = train_date
        self.test_date = test_date
        self.end_date = end_date
        self.model_layers = model_layers
        if model_dir is None or len(model_dir) == 0:
            self.model_dir = os.path.join(data_dir, 'trained_models')
        else:
            self.model_dir = os.path.join(data_dir, model_dir)
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)
        self.arguments =arguments
        today_int = 0#int( datetime.now().strftime('%Y%m%d') )
        self.experiment_id = os.environ.get('EXPERIMENT_ID', today_int)
        torch.manual_seed(19491001)
        torch.set_printoptions(precision=20)

    def get_train_id(self, stepsString):
        fileName = os.path.join(self.data_dir,'trained_models', 'trainId.log' )
        check_filename = os.path.join(self.data_dir, 'trained_models', '~trainId.log' )
        sleep_count =0
        while os.path.exists(check_filename) and sleep_count<10:  # 如果有其他人在写这个文件，等10秒
            time.sleep(1000)
            print(f'{check_filename} exist {sleep_count}!')
            sleep_count +=1
        if sleep_count<10:
            train_id = self.get_train_id_with_file(
                check_filename, fileName, stepsString
            )
        else:
            train_id = _MAX_TRAIN_ID
        return f'{_MAX_TRAIN_ID - train_id:04}'

    def get_train_id_with_file(self, check_filename, filename, steps_string):
        with open(check_filename, 'w') as checkFile:
            checkFile.write(f'@{self.hostname}')
        if os.path.exists(filename):
            with open(filename, 'r') as file:
                lines = file.readlines()
        else:
            lines = []
        lines.append(f'\n@{self.hostname}_{steps_string}_{self.input_predays}')
        with open(filename, 'w') as file:
            file.writelines(lines)
        result = len(lines)
        os.remove(check_filename)
        return result


    def train(self):
        
        logger = self.logger
        dev = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        logger.info(dev)

        pattern = re.compile(r'Used\s+:\s+([0-9]+)\s+MiB')
        channel_index=self.channel_index

        log_writer = SummaryWriter(os.path.join( self.data_dir,'trained_models','runs'),comment=f'{self.hostname}')
        is_cuda = ( dev == torch.device("cuda:0"))
        now = datetime.now()
        date_time_string =f"{now.strftime('%Y%m%d')}_{now.strftime('%H%M%S')}"
        logger.info(f"Date time string: {date_time_string}")

        steps = [int(s) for s in self.predict_timesteps.split(',')]
        predict_timesteps = np.array( steps,dtype=int)
        predict_timestep_count = len(predict_timesteps)  # 预测3、6、12、24、48、72、120、168小时这8个数据
        self.input_timesteps = self.base_timesteps #+ self.input_predays*24+maxPredictTimeStep
        self.logger.info(f'train args:\n{self.arguments}\tinput_timesteps:{self.input_timesteps}')

        steps_string = ','.join([str(i) for i in steps])
        model_filename = os.path.join( self.model_dir,f"airnet6_{self.source_type.name }_ml{self.model_layers}_b{self.base_timesteps}_p{self.input_predays}_ps{steps_string}_bat{self.batch_size}_c{channel_index}_r{self.channel_roll}_{date_time_string}.pt")
        self.logger.info(f'train to ：{model_filename}')
        shift_day = self.input_predays
        train_dl, valid_dl = get_train_valid_loader(self.data_dir,self.source_type, self.weather_source, self.input_timesteps, predict_timesteps, self.batch_size, channel_index, self.channel_roll, shift_day, self.train_date, self.test_date, self.end_date, dev)
        channel_count = train_dl.dataset.channel_count()
        point_count = train_dl.dataset.point_count()
        logger.info(f"train and valid data: {channel_count}*{train_dl.dataset.x.shape[1]}*{train_dl.dataset.x.shape[2]}")
        model = Model(self.model_layers, self.input_timesteps, predict_timestep_count, point_count, channel_count, is_cuda)
        model = model.to(AirDataset.FLOAT_TYPE).to(dev)

        loss_func = F.l1_loss

        opt = optim.Adam(model.parameters())

        best_mean_valid_loss = 1e4

        train_id = self.get_train_id(steps_string)
        has_log_gpu_memory = False
        last_epoch = 0
        empty_epoch = '' #看看中间有几个epoch没有收敛
        epoch_begin_time = datetime.now()
        for epoch in range(self.epochs):
            model.train()
            for xb, yb in train_dl:
                pred = model(xb)
                loss = loss_func(pred, yb)

                opt.zero_grad()
                loss.backward()
                opt.step()

            model.eval()
            with torch.no_grad():
                valid_loss = 0.0
                valid_sum = 0.0

                for xb, yb in valid_dl:
                    valid_loss += loss_func(model(xb), yb, reduction='sum')
                    valid_sum += torch.sum(yb)

                mean_valid_loss = valid_loss/valid_sum # average of all samples

                log_writer.add_scalar(f'air/  {train_id}:@{self.hostname}_{steps_string}_c{channel_index}', float(mean_valid_loss), epoch)
                if mean_valid_loss < best_mean_valid_loss:
                    torch.save(model.state_dict(), model_filename)
                    logger.info(f"---epoch={epoch} {empty_epoch}Model with the following mean valid loss saved: { mean_valid_loss.item()*100:.2f}%")
                    best_mean_valid_loss = mean_valid_loss
                    last_epoch = epoch
                    empty_epoch = ''
                else:
                    empty_epoch += '.'   # 没有收敛

            if not has_log_gpu_memory and is_cuda:
                all_use_time = (self.epochs -1) *( datetime.now() - epoch_begin_time +timedelta(seconds=_SLEEP_SECONDS))
                result = subprocess.run(['/usr/bin/nvidia-smi', '-q','-i','0','-d','MEMORY,TEMPERATURE'], stdout=subprocess.PIPE)
                gpu_string = result.stdout.decode('utf-8')
                used_memory = pattern.findall(gpu_string)[0]
                logger.info(gpu_string)
                logger.info(f'Task will cost {all_use_time} , will finish at {datetime.now()+all_use_time:%m-%d %H:%M}')
                has_log_gpu_memory = True


            #休息10秒,GPU太热会reboot
            time.sleep(_SLEEP_SECONDS)

        del model
        if is_cuda:
            torch.cuda.empty_cache()

        log_writer.close()
        timeUsed = datetime.now()-now
        logger.info(f"训练用时： {str(timeUsed)}")
        return str(timeUsed), last_epoch, used_memory, model_filename

    def log_train_result(self,  train_time, last_epoch, used_memory, model_filename, steps, results):
        fileName = os.path.join(self.data_dir,'trained_models', 'trainResult.csv' )
        # checkFileName = os.path.join(self.data_dir, 'trained_models', '~writing_trainResult.csv' )
        avg = sum(results) / len(results)
        csv_lines = []
        for index, step in enumerate(steps):
            csv_lines.append({
                'experimentId': self.experiment_id,
                'modelFileName': model_filename,
                'trainTime': train_time,
                'lastEpoch': last_epoch,
                'usedMemory': used_memory,
                'arguments': self.arguments,
                'base_timesteps': self.base_timesteps,
                'batch_size': self.batch_size,
                'channel_index': self.channel_index,
                'channel_roll': self.channel_roll,
                'epochs': self.epochs,
                'input_predays': self.input_predays,
                'predict_timesteps': self.predict_timesteps,
                'avg': avg,
                'step': step,
                'result': results[index],
                'trainFinished': datetime.now(),
                'trainHostname': self.hostname,
                'model_layers': self.model_layers
            })
        # Make data frame of above data
        df = pd.DataFrame(csv_lines)
        self.logger.info(df.to_csv())
        if os.path.exists(fileName):
            # append data frame to CSV file
            df.to_csv(fileName, mode='a', index=False, header=False)        
        else:
            df.to_csv(fileName, mode='w', index=False, header=True)        
    
    
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-dp", "--data_dir", required=True,default="data/Air", help="path to the .pkl dataset, e.g. 'data/Dutch/dataset.pkl'")
    ap.add_argument("-src", "--source_type", required=True, type=str,default=1, help="数据源'")
    ap.add_argument("-ws", "--weather_source", required=False, help="config")
    ap.add_argument("-e", "--epochs", type=int, default=50, help="Number of epochs")
    ap.add_argument("-c", "--channel_index", required=False,  type=int,default=0, help="channel Index'")
    ap.add_argument("-cr", "--channel_roll", required=False,  type=int,default=0, help="channel_roll'")
    ap.add_argument("-id", "--input_predays", type=int, required=True, help="number of days to be included in the input")
    ap.add_argument("-ps", "--predict_timesteps", type=str, required=True, default='3,6,12,24', help="number of timesteps (hours) ahead for the prediction")
    ap.add_argument("-bt", "--base_timesteps", required=True, type=int,default=120, help="batch size for dataloader'")
    ap.add_argument("-bat", "--batch_size", required=True, type=int,default=64, help="batch size for dataloader'")
    ap.add_argument("-vs", "--vs_dir", default='/datapool01/shared/vs', required=False, help="vs model Data dir")
    ap.add_argument("-md", "--model_dir", default='', required=False, help="vs model Data dir")
    ap.add_argument("-ml", "--model_layers", default='', type=str, required=False, help="vs model Data dir")
    ap.add_argument("-td", "--train_date", required=True, help="")
    ap.add_argument("-ts", "--test_date", required=True, help="")
    ap.add_argument("-ed", "--end_date", required=True, help="")
    ags, unknowns = ap.parse_known_args()
    arguments = vars(ags)

    logger = _setup_logger(os.path.join( arguments["data_dir"],'log') )

    airnet = AirNetTrainer(arguments["model_layers"],
                    arguments["data_dir"], arguments, 
                    SourceType[ arguments["source_type"] ], SourceType[ arguments["weather_source"] ], 
                    arguments["epochs"], 
                    arguments["predict_timesteps"], arguments["base_timesteps"], 
                    arguments["batch_size"],
                    arguments["input_predays"], arguments["channel_index"], arguments["channel_roll"],
                    arguments["model_dir"], 
                    datetime.fromisoformat( arguments["train_date"]),
                    datetime.fromisoformat( arguments["test_date"]),
                    datetime.fromisoformat( arguments["end_date"])
                    )
    train_time, last_epoch, used_memory, model_filename = airnet.train()
    # trainTime,lastEpoch, usedMemory, modelFileName = '08:08:38.88888',  88, 8888, None

    tester = Tester(arguments["data_dir"], arguments, 
                    SourceType[ arguments["source_type"] ], SourceType[ arguments["weather_source"] ], 
                    arguments["vs_dir"], arguments["channel_index"],
                    arguments["model_dir"], 
                    datetime.fromisoformat( arguments["train_date"]),
                    datetime.fromisoformat( arguments["test_date"]),
                    datetime.fromisoformat( arguments["end_date"])
                    )
    steps, results = tester.test_one(arguments["model_layers"], arguments["predict_timesteps"], arguments["base_timesteps"],
                    arguments["batch_size"], arguments["input_predays"],
                    arguments["channel_index"], arguments["channel_roll"], train_time, last_epoch, used_memory, model_filename)
    airnet.log_train_result(train_time, last_epoch, used_memory, model_filename, steps, results)