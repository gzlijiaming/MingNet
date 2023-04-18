from cgi import test
from curses import meta
from datetime import datetime, timedelta,time
import json
import math
import shutil
from tkinter import LEFT
from turtle import forward
import torch
from airnet_model import Model
from air_dataloader import *
import torch.nn.functional as F
import argparse
import numpy as np 
import sys,os,glob
import logging
import socket
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
# import args 
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
# from air_dataset import AirDataset
# import air_dataset as AirDatasetModule
import pickle
import pandas as pd
import ray
sys.path.append("DataSource/vs") 
from prophet_tester import ProphetTester
from yesterday_tester import YesterdayTester
from chem_tester import ChemTester
from lstm_tester import LstmTester
from combine_tester import CombineTester

sys.path.append("DataPrepare") 
from data_file import DataFile, DataStep, SourceType
sys.path.append("DataSource/mon") 
from air_stat import AirStat



plt.rcParams["font.sans-serif"]=["HYJunhei"] #设置中文字体
plt.rcParams["axes.titlesize"]=8 #设置标题字体
plt.rcParams["axes.unicode_minus"]=False #该语句解决图像中的“-”负号的乱码问题
plt.rcParams["figure.figsize"]=(6,4) #设置图大小
plt.rcParams["figure.dpi"]=160 #设置图大小
# plt.rcParams["xtick.labelsize"]=4
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())




_FORWARD_COUNTS ={
    168:48,
    120:48,
    72:24,
    48:24,
    24:12,
    12:6,
    6:3,
    3:3,
    1:1
}
_MAX_FORE_DAY = 7


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
    fhlr = logging.FileHandler(os.path.join( log_dir,f'test_{hostname}.log')) # 输出到文件的handler
    fhlr.setFormatter(formatter)
    fhlr.setLevel('DEBUG')
    logger.addHandler(chlr)
    logger.addHandler(fhlr)
    return logger 

class Tester():
    def __init__(self , data_dir, arguments, source_type,weather_source, vs_dir, channel_index, model_dir=None, train_date = None, test_date = None, end_date = None):
        self.logger = logging.getLogger()
        self.data_dir = data_dir
        self.source_type = source_type
        self.weather_source = weather_source
        self.vs_dir = vs_dir
        self.channel_index = channel_index
        self.train_date = train_date
        self.test_date = test_date
        self.end_date = end_date
        if model_dir is None or len(model_dir) == 0:
            self.model_dir = os.path.join(data_dir, 'trained_models')
        else:
            self.model_dir = os.path.join(data_dir, model_dir)
        self.test_dir = os.path.join( self.model_dir, 'test')
        if not os.path.exists(self.test_dir):
            os.mkdir(self.test_dir)
        self.models={}
        self.logger.info(f'test args:\n{arguments}')
        torch.manual_seed(123)
        torch.set_printoptions(precision=20)
        self.min_date =datetime(1999,1,1)

        # self.dev =  torch.device("cpu")
        self.dev = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        print(self.dev)
        self.data_file = DataFile(data_dir)
        self.metas = self.data_file.read_meta(source_type)

        #from scaler.pkl (ralted to dataset.pkl) - WIND SPEED
        self.scaler_min = torch.tensor([0.000e+00]).to(self.dev)
        # 第一个因子是臭氧，最大值从元数据那里拿
        self.scaler_max = torch.tensor([0.000e+00]).to(self.dev)
        # self.scaler_max[0] = float(self.metas['channels'][channelIndex][3])

        self.loss_func = F.l1_loss
        self.loss_func_2 = F.mse_loss

        self.fig = plt.figure()
        

    def get_model(self,model_layers,input_predays, base_timesteps, batch_size, channel_index , channel_roll, steps_string, predict_timestep_count,point_count, channel_count, model_filename=None):
        input_timesteps = base_timesteps #+ input_predays*24+predictTimesteps[len(predictTimesteps)-1]
        if model_filename is None:
            model_files = os.path.join( self.model_dir,f"airnet6_{self.source_type.name }_ml{model_layers}_b{base_timesteps}_p{input_predays}_ps{steps_string}_bat{batch_size}_c{channel_index}_r{channel_roll}_*.pt")
            lst= glob.glob(model_files,recursive=False)
            lst.sort(reverse=True)
            model_path = lst[0]
        else:
            model_path = model_filename
        self.logger.info(f'predict using：{model_path}')
        if model_path in self.models:
            return self.models[model_path]
        is_cuda = ( self.dev == torch.device("cuda:0"))
        best_model = Model(model_layers, input_timesteps, predict_timestep_count,point_count, channel_count,is_cuda)
        best_model = best_model.to(AirDataset.FLOAT_TYPE).to(self.dev) 
        best_model.load_state_dict(torch.load(model_path))
        self.models[model_path] = best_model
        return best_model



    def test_one(self,modelLayers, predict_timesteps,base_timesteps, batch_size, input_predays, channel_index, channel_roll, train_time='', last_epoch=0, used_memory='', model_filename=None):
        now = datetime.now()
        steps = [int(s) for s in predict_timesteps.split(',')]
        predict_timesteps = np.array( steps,dtype=int)
        predict_timestep_count = len(predict_timesteps)  # 预测3、6、12、24、48、72、120、168小时这8个数据
        # maxPredictTimeStep = np.max(predictTimesteps).item()
        input_timesteps = base_timesteps #+ input_predays*24+maxPredictTimeStep


        steps_string = ','.join([str(i) for i in steps])
        # shiftDay = shiftDays[predictTimesteps[len(predictTimesteps)-1]]
        test_dl = get_test_loader(self.data_dir, self.source_type,self.weather_source, input_timesteps, predict_timesteps, batch_size, channel_index, channel_roll, input_predays, self.train_date, self.test_date, self.end_date, self.dev)
        channel_count = test_dl.dataset.channel_count()
        point_count =test_dl.dataset.point_count()
        best_model = self.get_model(modelLayers, input_predays,base_timesteps, batch_size, channel_index, channel_roll, steps_string, predict_timestep_count,point_count,channel_count, model_filename)

        best_model.eval()
        with torch.no_grad():

            F = [torch.empty(0, point_count, dtype=AirDataset.FLOAT_TYPE).to(self.dev) for _ in range(predict_timestep_count)]
            O = [torch.empty(0, point_count, dtype=AirDataset.FLOAT_TYPE).to(self.dev) for _ in range(predict_timestep_count)]
            NMEs = [[] for _ in range(predict_timestep_count)]
            self.scaler_max[0] = float(self.metas['channels'].iloc[channel_index]['maxValue'])
            self.scaler_min[0] = float(self.metas['channels'].iloc[channel_index]['minValue'])

            for xb, yb in test_dl:
                #xb: [batch_size, C, T, V]
                
                pred = best_model(xb)
                yb = yb
                for predict_timestep_index in range(predict_timestep_count):

                    pred_unscaled = pred[:, predict_timestep_index, :] * (self.scaler_max - self.scaler_min) + self.scaler_min
                    yb_unscaled = yb[:, predict_timestep_index, :] * (self.scaler_max - self.scaler_min) + self.scaler_min

                    F[predict_timestep_index] = torch.cat((F[predict_timestep_index],pred_unscaled),0)
                    O[predict_timestep_index] = torch.cat((O[predict_timestep_index],yb_unscaled),0)
                    # NMEs[predictTimestepIndex].append(float( self.loss_func(pred_unscaled,yb_unscaled,reduction='sum')/torch.sum(yb_unscaled) ))


        self.logger.info(f"————channel {channel_index}: {self.metas['channels'].iloc[channel_index]['name']}")
        experimental_data = ''
        steps = []
        results = []
        for predict_timestep_index in reversed(range(predict_timestep_count)):
            # test_loss /= test_num # average of all samples
            # calc NME
            # for tIndex in range(0,O[predictTimestepIndex].shape[0],24):
            for pIndex in range(O[predict_timestep_index].shape[1]):
                pred_unscaled = F[predict_timestep_index][:, pIndex]
                yb_unscaled = O[predict_timestep_index][:, pIndex]
                NMEs[predict_timestep_index].append(float( self.loss_func(pred_unscaled,yb_unscaled,reduction='sum')/torch.sum(yb_unscaled) ))

            self.logger.info(f"————predictTimestep[{predict_timestep_index}]={predict_timesteps[predict_timestep_index]}")
            MAE = self.loss_func(F[predict_timestep_index],O[predict_timestep_index],reduction='mean')
            self.logger.info(f"Mean Error : {MAE} ")
            # MSE = self.loss_func_2(F[predictTimestepIndex],O[predictTimestepIndex],reduction='mean')
            # self.logger.info(f"Root Mean Square Error : {MSE}")
            NME = np.array(NMEs[predict_timestep_index]).mean().item()
            self.logger.info(f"Normalized Mean Bias : {NME*100:.2f}%")
            experimental_data += f'{NME*100:.2f}%\n'
            steps.append( predict_timesteps[predict_timestep_index] )
            results.append( NME )
        self.logger.info(f"实验数据： {used_memory}\t{train_time}\n{experimental_data[:-1]}\t{last_epoch}\n")


        # test_loss_2_scaled /= test_num # average of all samples
        # logger.info(f"MSE(scaled) Average: {torch.mean(test_loss_2_scaled).item()} , Max: {torch.max(test_loss_2_scaled).item()}, Min: {torch.min(test_loss_2_scaled).item()}")

        time_used = datetime.now()-now
        self.logger.info(f"测试用时： {str(time_used)}")
        return steps, results



    def forecast_current(self, action_time, param_group_id):
        logger = self.logger
        # newestTimePath = os.path.join(self.data_dir, 'newestTime.csv')
        # newestTime = pd.read_csv(newestTimePath, index_col=0, parse_dates=[0,1], infer_datetime_format=True)
        mon_time_path = os.path.join(self.data_dir, f'{self.source_type.name}_meta_times.csv')
        mon_time = pd.read_csv(mon_time_path, index_col=3, parse_dates=[3,4], infer_datetime_format=True)
        weather_time_path = os.path.join(self.data_dir, f'{self.weather_source.name}_newestWeatherTime.csv')
        weather_time_df = pd.read_csv(weather_time_path, index_col=0, parse_dates=[0,1], infer_datetime_format=True)
        weather_data_path = os.path.join(self.data_dir, f'{self.weather_source.name}_newestWeatherData.npy')
        weather_data = np.load(weather_data_path)
        weather_data = weather_data[:,:,:-1] # 去掉最后一个测试的channel
        
        # 加入worktime
        newest_worktime = np.load(os.path.join(self.data_dir, f'{self.source_type.name}_newest_worktime.npy'))

        best_param = pd.read_excel(os.path.join(self.data_dir, f'{self.source_type.name}_best_param.xlsx'))
        best_param = best_param.loc[best_param['param_group_id'] == param_group_id]


        if action_time not in mon_time.index:
            action_time = mon_time.index.max()
        mon_action_index = mon_time.index.get_loc(action_time) +1
        weather_action_index = weather_time_df.index.get_loc(action_time) +1
        channels_df = self.metas['channels']
        max_fore_hour = 0
        channel_count = 1+1+ weather_data.shape[2]
        for index, param_row in best_param.iterrows():
            steps = param_row['predict_timesteps'].split(',')
            steps = [int(i) for i in steps]
            max_fore_hour = max(max_fore_hour, max(steps))
     
        for row_index,channel_row in channels_df.iterrows():
            channel_index = channels_df.index.get_loc(row_index)
            channel_param = best_param.loc[best_param['c_index'] == channel_index]
            
            # if channelIndex>0:
            #     continue
            logger.info(f"————{action_time} \tchannel {channel_index}: {channel_row['symbol']}")
            self.scaler_max[0] = float(channel_row['maxValue'])
            self.scaler_min[0] = float(channel_row['minValue'])

            # change dim order: T, V, C -> C, T, V
            point_indexs = self.data_file.get_point_index_by_cid( self.metas['points']['cid'], row_index)
            # monDataPath = os.path.join(self.data_dir, f'newestData_c{channelIndex}.npy')
            mon_data = self.data_file.read_data(channel_index, self.source_type, DataStep.normalized) # [T, V, C]
            point_count =mon_data.shape[1]
            result_data = np.empty((max_fore_hour, point_count , 1),dtype=float)
            result_data.fill(np.nan)
            # change dim order: T, V, C -> C, T, V
            mon_data = torch.tensor(mon_data).permute(2, 0, 1).to(AirDataset.FLOAT_TYPE) 
            # resultData = np.empty((forwardCount, 0, pointCount ),dtype=float)
            
            for stepsIndex, param_row in channel_param.iterrows():
                steps = param_row['predict_timesteps'].split(',')
                steps = [int(i) for i in steps]
                batch_size = int( param_row['batch_size'])
                input_preday = int(param_row['input_predays'])
                base_timesteps = int(param_row['base_timesteps'])
                channel_roll = int(param_row['channel_roll'])
                model_layers = param_row['model_layers']
                predict_timesteps = np.array( steps,dtype=int)
                predict_timestep_count = len(predict_timesteps)  # 预测3、6、12、24、48、72、120、168小时这8个数据
                max_predict_timestep = int(np.max(predict_timesteps))
                forward_count = _FORWARD_COUNTS[max_predict_timestep]
                input_timesteps = base_timesteps #+ input_preday*24+maxPredictTimeStep
                steps_string = ','.join([str(i) for i in steps])
                # weather_work_data = np.concatenate((newest_worktime[:,:,channel_index:channel_index+1], 
                #                                     np.roll(weather_data, channel_roll, 2) ), axis=2)
                weather_work_data = np.concatenate((newest_worktime[:,:,channel_index:channel_index+1], 
                                                    weather_data ), axis=2)
                point_weather_data = weather_work_data[:,point_indexs,:]
                point_weather_data = torch.tensor(point_weather_data).permute(2, 0, 1).to(AirDataset.FLOAT_TYPE)
                # stepsResult[T, stepIndex, P]
                steps_result = np.empty((forward_count, len(steps), point_count ),dtype=float)
                steps_result.fill(np.nan)
                shift_hours = input_preday*24
                xb = np.empty((batch_size, channel_count, input_timesteps, point_count),dtype=float )
                xb = torch.tensor(xb).to(AirDataset.FLOAT_TYPE)
                best_model = self.get_model(model_layers, input_preday,base_timesteps, batch_size, channel_index, channel_roll, steps_string, predict_timestep_count,point_count,channel_count)
                best_model.eval()
                with torch.no_grad():
                    for tindex in range(0,forward_count,batch_size):
                        tindex2 = min(tindex, forward_count-batch_size)
                        mon_tindex = mon_action_index- input_timesteps- forward_count+tindex2
                        weather_tindex = weather_action_index - input_timesteps - forward_count+tindex2 + shift_hours
                        for bat_index in range(batch_size):
                            # xb[bat_index, 0, :, :] = mon_data[0, mon_tindex + bat_index:  mon_tindex + bat_index+ input_timesteps, :]
                            # xb[bat_index, 1:, :, :] = point_weather_data[:, weather_tindex + bat_index:  weather_tindex + bat_index+ input_timesteps, :]
                            combine = torch.concat( (mon_data[0:1, mon_tindex + bat_index:  mon_tindex + bat_index+ input_timesteps, :],
                                                     point_weather_data[:, weather_tindex + bat_index:  weather_tindex + bat_index+ input_timesteps, :]), 
                                                     axis=0)
                            xb[bat_index, :, :, :] = torch.roll(combine, channel_roll, 0)
                        pred = best_model(xb.to(self.dev) ) #[bat, steps, V]
                        pred = torch.where(pred <0,0,pred)
                        pred = torch.where(pred >1,1,pred)
                        pred_unscaled = pred * (self.scaler_max - self.scaler_min) + self.scaler_min
                        used_count = min(batch_size, steps_result.shape[0])
                        steps_result[tindex2:tindex2+batch_size,:,: ] = pred_unscaled.cpu().detach().numpy()[-used_count:,:,:]
                
                for step_index, step in enumerate(steps):
                    forward_count = _FORWARD_COUNTS[step]
                    src_end = steps_result.shape[0]
                    src_start = src_end - forward_count
                    dst_end = step
                    dst_start = dst_end - forward_count
                    result_data[dst_start:dst_end, :, 0] = steps_result[src_start:src_end, step_index, :]
                    print(f'{step} max {np.max(steps_result[src_start:src_end, step_index, :]):.2f} ')
            temp_dir = os.path.join(self.data_dir ,'temp')
            if not os.path.exists(temp_dir):
                os.mkdir(temp_dir)
            result_data_path = os.path.join(temp_dir,  f'resultData_c{channel_index}_{action_time:%Y%m%d_%H}.npy')
            np.save(result_data_path, result_data);    
        return action_time
        
    # p_bs：多个[预测时间，input_predays, 批量数]的组合，timeInDayString：什么时刻做预测，比如说晚上23:50
    def all_days_test(self,  param_group_id, db_config, db):
        # sourcery skip: merge-list-append, move-assign-in-block
        channels_df = self.metas['channels']
        _, timeFile, _ = self.data_file.get_meta_filename(self.source_type)
        times_df = pd.read_csv(timeFile, index_col=3, parse_dates=[3,4], infer_datetime_format=True)
        mon_dfs = []
        aqi_dfs = []

        test_data_type = f'AirNet6 {self.source_type.name}_{self.test_date:%Y_%m}'
        self.forecast_test_data( param_group_id)
        self.collect_test_data(channels_df, times_df, param_group_id, self.data_dir)   #迁就 forecastCurrent
        mon_dfs.append(self.mon_stat_test_data(test_data_type, channels_df, times_df, self.data_dir))
        aqi_dfs.append(self.air_stat_test_data(channels_df, times_df, test_data_type,   self.data_dir))

        test_data_type = f'Yesterday {self.source_type.name}_{self.test_date:%Y_%m}'
        yesterday = YesterdayTester(self.data_dir, self.vs_dir, self.source_type, self.train_date, self.test_date, self.end_date)
        # yesterday.forecast_test_data(channels_df, times_df, None, None, None)
        mon_dfs.append(self.mon_stat_test_data(test_data_type, channels_df, times_df, yesterday.test_dir))
        aqi_dfs.append(self.air_stat_test_data(channels_df, times_df, test_data_type,   yesterday.test_dir))

        test_data_type = f'Prophet {self.source_type.name}_{self.test_date:%Y_%m}'
        prophet = ProphetTester(self.data_dir, self.vs_dir, self.source_type, self.train_date, self.test_date, self.end_date)
        # prophet.forecast_test_data(channels_df, times_df, None, None, None)
        self.collect_test_data(channels_df, times_df, param_group_id, prophet.test_dir)
        mon_dfs.append(self.mon_stat_test_data(test_data_type, channels_df, times_df, prophet.test_dir))
        aqi_dfs.append(self.air_stat_test_data(channels_df, times_df, test_data_type,   prophet.test_dir))

        test_data_type = f'lstm {self.source_type.name}_{self.test_date:%Y_%m}'
        lstm = LstmTester(self.data_dir, self.vs_dir, self.source_type, self.train_date, self.test_date, self.end_date)
        # lstm.forecast_test_data(channels_df, times_df)
        self.collect_test_data(channels_df, times_df, param_group_id, lstm.test_dir)
        mon_dfs.append(self.mon_stat_test_data(test_data_type, channels_df, times_df, lstm.test_dir))
        aqi_dfs.append(self.air_stat_test_data(channels_df, times_df, test_data_type,   lstm.test_dir))

        if self.weather_source == SourceType.gfs_gd:
            test_data_type = f'WrfChem {self.source_type.name}_{self.test_date:%Y_%m}'
            chem = ChemTester(db_config, db, SourceType.wrfchem, 
                              self.data_dir, self.vs_dir, self.source_type, self.train_date, self.test_date, self.end_date)
            chem.forecast_test_data(channels_df, times_df)
            mon_dfs.append(self.mon_stat_test_data(test_data_type, channels_df, times_df, chem.test_dir))
            aqi_dfs.append(self.air_stat_test_data(channels_df, times_df, test_data_type,   chem.test_dir))

            corrects = [0.65, 0.52, -0.46, 0.12, -0.42, -0.28]
            test_data_type = f'WrfChem_correct {self.source_type.name}_{self.test_date:%Y_%m}'
            chem = ChemTester(db_config, db, SourceType.wrfchem, 
                              self.data_dir, self.vs_dir, self.source_type, self.train_date, self.test_date, self.end_date)
            chem.correct_test_data(channels_df, times_df, corrects)
            mon_dfs.append(self.mon_stat_test_data(test_data_type, channels_df, times_df, chem.correct_dir))
            aqi_dfs.append(self.air_stat_test_data(channels_df, times_df, test_data_type,   chem.correct_dir))

            test_data_type = f'Ji_he {self.source_type.name}_{self.test_date:%Y_%m}'
            ji_he = ChemTester(db_config, db, SourceType.ji_he, 
                              self.data_dir, self.vs_dir, self.source_type, self.train_date, self.test_date, self.end_date)
            ji_he.forecast_test_data(channels_df, times_df)
            mon_dfs.append(self.mon_stat_test_data(test_data_type, channels_df, times_df, ji_he.test_dir))
            aqi_dfs.append(self.air_stat_test_data(channels_df, times_df, test_data_type,   ji_he.test_dir))


        mon_df = pd.concat(mon_dfs)
        mon_df.to_csv(os.path.join(self.vs_dir,f'{self.source_type.name}_{self.test_date:%Y_%m}_单项污染物浓度预报统计评估.csv'))
        if self.source_type == SourceType.guoKong:
            aqi_df = pd.concat(aqi_dfs)
            aqi_df.to_csv(os.path.join(self.vs_dir,f'{self.source_type.name}_{self.test_date:%Y_%m}_空气质量指数预报评估.csv'))
            test_dirs = {
                'AirNet6':self.data_dir,
                'Yesterday':yesterday.test_dir,
                'Prophet':prophet.test_dir,
                'lstm':lstm.test_dir,
            }
            if self.weather_source == SourceType.gfs_gd:
                test_dirs |= {
                    'WrfChem': chem.test_dir,
                    'WrfChem_correct': chem.correct_dir,
                    'Ji_he': ji_he.test_dir,
                }
            self.make_aqi_plot_data(test_dirs)


    def make_aqi_plot_data(self, test_dirs):
        test_dir = list(test_dirs.values())[0]
        df = pd.read_csv(os.path.join( test_dir, f'{self.source_type.name}_aqi.csv'))
        name = os.path.basename(os.path.normpath(test_dir))
        df['fValue'] = df['oValue']
        df['测试数据'] = '实况数据'
        dfs = [df]
        for test_name,test_dir in test_dirs.items():
            df = pd.read_csv(os.path.join( test_dir, f'{self.source_type.name}_aqi.csv'))
            # name = os.path.basename(os.path.normpath(test_dir))
            df['测试数据'] = test_name
            dfs.append(df)
        plot_df = pd.concat(dfs)
        plot_df.to_csv(os.path.join(self.vs_dir,f'{self.source_type.name}_{self.test_date:%Y_%m}_AQI数据集.csv'))

    def air_stat_test_data(self, channels_df, times_df, test_data_type,   test_dir):
        if self.source_type == SourceType.ca:
            return []
        
        result_csv_file = os.path.join(test_dir, f'{self.source_type.name}_{test_data_type}_空气质量指数预报评估.csv')
        if os.path.exists(result_csv_file):
            return pd.read_csv(result_csv_file)
        self.air_stat_aqi(channels_df, times_df, test_data_type, test_dir)
        self.air_stat_level(test_dir)
        
        csv_lines = []
        aqi_dfs = pd.read_csv(os.path.join(test_dir, f'{self.source_type.name}_aqi.csv'), parse_dates=['forecast_time','monitor_time'],
                             dtype={'oValue': np.float64,'fValue': np.float64,  'preDay':np.int64}, )
        aqi_dfs['AQI范围预报准确'] = (aqi_dfs['oValue']<=aqi_dfs['fValue']*1.25 ) & ( aqi_dfs['oValue']>=aqi_dfs['fValue']*0.75)
        aqi_dfs['AQI级别预报准确'] = (aqi_dfs['oAQI级别']==aqi_dfs['fAQI级别'])
        group = aqi_dfs.groupby(['region_code', 'region_name', 'preDay'], as_index =False)
        group.apply(self.converge_aqi_range, csv_lines, test_data_type)
        group = aqi_dfs.groupby(['region_code', 'region_name', 'preDay'], as_index =False)
        group.apply(self.converge_aqi_level, csv_lines, test_data_type)
        group = aqi_dfs.groupby(['region_code', 'region_name', 'preDay', 'fAQI级别'], as_index =False)
        group.apply(self.converge_aqi_per_level, csv_lines, test_data_type)
        group = aqi_dfs.groupby(['region_code', 'region_name', 'preDay'], as_index =False)
        group.apply(self.converge_aqi_high, csv_lines, test_data_type)
        
        
        
        head_dfs = pd.read_csv(os.path.join(test_dir, f'{self.source_type.name}_head.csv'),parse_dates=['forecast_time','monitor_time'],
                              dtype={ 'preDay':np.int64})
        head_dfs['首要污染物预报准确'] = (head_dfs['oValue']==head_dfs['fValue'])
        group = head_dfs.groupby(['region_code', 'region_name', 'preDay'], as_index =False)
        group.apply(self.converge_aqi_head, csv_lines, test_data_type)
        
        resule_df = pd.DataFrame(csv_lines)
        resule_df.to_csv(result_csv_file)
        return resule_df
        
        
    def collect_test_data(self, channels_df, times_df,  param_group_id, test_dir):
        # 收集结果
        for row_index,row in channels_df.iterrows():
            channel_index = channels_df.index.get_loc(row_index)
            fdata_path = os.path.join( test_dir , f'{self.source_type.name}_fData_c{channel_index}.npy')
            odata_path = os.path.join( test_dir, f'{self.source_type.name}_oData_c{channel_index}.npy')
            
            #源数据db
            src_data = self.data_file.read_data(channel_index, self.source_type, DataStep.src) # [T, V, C]
            test_index = times_df.index.get_loc(self.test_date)
            end_index = times_df.index.get_loc(self.end_date + timedelta(hours=-1))+1
            odata = src_data[test_index:end_index+(_MAX_FORE_DAY-1)*24,:,0]
            # outData:  [predDays,T, V]，其中predDays为提早几天预测，T要比x的T要多，因为后面加多了预测数据
            days = (self.end_date - self.test_date).days
            fdata = np.empty((days, _MAX_FORE_DAY*24, odata.shape[1]),dtype=float)
            day_index = 0
            current_time = self.test_date 
            while current_time < self.end_date:
                action_time = current_time + timedelta(hours=-1)
                result_data_path = os.path.join(test_dir, 'temp',  f'resultData_c{channel_index}_{action_time:%Y%m%d_%H}.npy')
                result_data = np.load(result_data_path )[:,:,0];    
                fdata[day_index, :, :] = result_data
                # os.remove( resultDataPath)
                day_index += 1
                current_time += timedelta(days=1)
                
            np.save(fdata_path,fdata)
            np.save(odata_path,odata)

    def forecast_test_data(self, param_group_id):
        #做预测
        current_time = self.test_date 
        while current_time < self.end_date:   # 预测的，多maxForeDay-1天
            actionTime = current_time + timedelta(hours=-1)
            self.forecast_current(actionTime, param_group_id)
            current_time += timedelta(days=1)
        
    def mon_stat_test_data(self, test_data_type, channels_df, times_df, test_dir):
        #统计
        result_csv_file = os.path.join(test_dir,f'{self.source_type.name}_{test_data_type}_单项污染物浓度预报统计评估.csv')
        # if os.path.exists(result_csv_file):
        #     return pd.read_csv(result_csv_file)
        csv_lines = []
        for row_index,row in channels_df.iterrows():
            channel_index = channels_df.index.get_loc(row_index)
            fdata_path = os.path.join( test_dir , f'{self.source_type.name}_fData_c{channel_index}.npy')
            odata_path = os.path.join( test_dir, f'{self.source_type.name}_oData_c{channel_index}.npy')
            fdata = np.load(fdata_path)
            odata = np.load(odata_path)
            self.plot_values(fdata,odata, channel_index, row, test_data_type,   test_dir)
            # 转维度 [predDays(所有天),T(168小时), V] -》[predDays(7天),T(所有小时), V]
            F = np.zeros((_MAX_FORE_DAY, odata.shape[0], odata.shape[1]))
            for dindex in range(fdata.shape[0]):
                for hIndex in range(0, fdata.shape[1], 24):
                    copyHours = min(dindex*24+hIndex+24, F.shape[1] ) - (dindex*24+hIndex)
                    F[hIndex//24, dindex*24+hIndex:dindex*24+hIndex+copyHours, :] = fdata[dindex, hIndex:hIndex+copyHours,:]
            
            experimentalData = ''
            for pred_day in range(_MAX_FORE_DAY):
                O1 = odata[pred_day*24:odata.shape[0]-(_MAX_FORE_DAY-1-pred_day)*24,:]
                F1 = F[pred_day,pred_day*24:odata.shape[0]-(_MAX_FORE_DAY-1-pred_day)*24,:]
                mae = self.assess_hj1130(test_data_type, pred_day, row['symbol'], F1, O1, csv_lines)
                experimentalData += f'{mae:.1%}, '
            experimentalData += '\n'
            self.logger.info(f"{test_data_type} {row['symbol']} 实验数据：\n{experimentalData}")
        df = pd.DataFrame(csv_lines)
        df.to_csv(result_csv_file)
        return df
    
    

    def converge_aqi_head(self, df, csv_lines, test_data_type):
        count = df.shape[0]
        right = df[df['首要污染物预报准确'] == True].shape[0]
        csv_lines.append({
            '测试数据':test_data_type,'region_code':df.iloc[0]['region_code'], 'region_name':df.iloc[0]['region_name'],
            '预测天数':df.iloc[0]['preDay'],'评估因子':'首要污染物预报准确率',
            '评估值':right/count})
        
    def converge_aqi_per_level(self, df, csv_lines, test_data_type):
        count = df.shape[0]
        right = df[df['AQI级别预报准确'] == True].shape[0]
        csv_lines.append({
            '测试数据':test_data_type,'region_code':df.iloc[0]['region_code'], 'region_name':df.iloc[0]['region_name'],
            '预测天数':df.iloc[0]['preDay'],'评估因子':f"AQI分级别{df.iloc[0]['fAQI级别']}_预报准确率",
            '评估值':right/count})

    def converge_aqi_level(self, df, csv_lines, test_data_type):
        count = df.shape[0]
        right = df[df['AQI级别预报准确'] == True].shape[0]
        csv_lines.append({
            '测试数据':test_data_type,'region_code':df.iloc[0]['region_code'], 'region_name':df.iloc[0]['region_name'],
            '预测天数':df.iloc[0]['preDay'],'评估因子':'AQI级别预报准确率',
            '评估值':right/count})

    def converge_aqi_high(self, df, csv_lines, test_data_type):
        count = df[df['oValue'] > 200].shape[0]
        if count>0:
            right = df[df['AQI级别预报准确'] == True].shape[0]
            csv_lines.append({
                '测试数据':test_data_type,'region_code':df.iloc[0]['region_code'], 'region_name':df.iloc[0]['region_name'],
                '预测天数':df.iloc[0]['preDay'],'评估因子':'重污染天预报准确率',
                '评估值':right/count})

    def converge_aqi_range(self, df, csv_lines, test_data_type):
        count = df.shape[0]
        right = df[df['AQI范围预报准确'] == True].shape[0]
        csv_lines.append({
            '测试数据':test_data_type,'region_code':df.iloc[0]['region_code'], 'region_name':df.iloc[0]['region_name'],
            '预测天数':df.iloc[0]['preDay'],'评估因子':'AQI范围预报准确率',
            '评估值':right/count})

    def air_stat_level(self, test_dir):
        o_dfs = self.read_of_csv(test_dir, '_oData.csv', 'oValue')
        f_dfs = self.read_of_csv(test_dir, '_fData.csv', 'fValue')
        join_dfs = pd.merge(o_dfs, f_dfs, how='left', on=['region_code','region_name','forecast_time','monitor_time','c_name'])
        join_dfs['preDay'] =  (pd.to_datetime(join_dfs['monitor_time']) - pd.to_datetime(join_dfs['forecast_time'])).dt.days +1
        aqi_dfs = join_dfs[join_dfs['c_name']=='aqi']
        aqi_dfs.drop(aqi_dfs[aqi_dfs['oValue'].isna() | aqi_dfs['fValue'].isna()].index, inplace=True)
        aqi_dfs['oValue'] = pd.to_numeric(aqi_dfs['oValue'])
        aqi_dfs['fValue'] = pd.to_numeric(aqi_dfs['fValue'])
        aqi_dfs['oAQI级别'] = 1
        aqi_dfs['fAQI级别'] = 1
        aqi_dfs['oAQI级别'][aqi_dfs['oValue']>=51] = 2
        aqi_dfs['fAQI级别'][aqi_dfs['fValue']>=51] = 2
        aqi_dfs['oAQI级别'][aqi_dfs['oValue']>=101] = 3
        aqi_dfs['fAQI级别'][aqi_dfs['fValue']>=101] = 3
        aqi_dfs['oAQI级别'][aqi_dfs['oValue']>=151] = 4
        aqi_dfs['fAQI级别'][aqi_dfs['fValue']>=151] = 4
        aqi_dfs['oAQI级别'][aqi_dfs['oValue']>=201] = 5
        aqi_dfs['fAQI级别'][aqi_dfs['fValue']>=201] = 5
        aqi_dfs['oAQI级别'][aqi_dfs['oValue']>300] = 6
        aqi_dfs['fAQI级别'][aqi_dfs['fValue']>300] = 6
        head_dfs = join_dfs[join_dfs['c_name']=='首要污染物']
        head_dfs.drop(head_dfs[head_dfs['oValue'].isna()].index, inplace=True)
        aqi_dfs.to_csv(os.path.join(test_dir, f'{self.source_type.name}_aqi.csv'))
        head_dfs.to_csv(os.path.join(test_dir, f'{self.source_type.name}_head.csv'))

    # TODO Rename this here and in `airStatLevel`
    def read_of_csv(self, test_dir, filename, prop_name):
        result = pd.read_csv(
            os.path.join(test_dir, f'{self.source_type.name}{filename}'),
            parse_dates=['forecast_time', 'monitor_time'],
            infer_datetime_format=True,
        )
        result = result[
            [
                'region_code',
                'region_name',
                'forecast_time',
                'monitor_time',
                'c_name',
                'monitor_value',
            ]
        ]
        result.rename(columns={'monitor_value': prop_name}, inplace=True)
        return result


    def air_stat_aqi(self, channels_df, times_df, test_data_type, test_dir):
        stat_dir = os.path.join(test_dir, self.source_type.name)
        if not os.path.exists(stat_dir):
            os.mkdir(stat_dir)
        upload_dir = os.path.join(test_dir, 'upload')
        if not os.path.exists(upload_dir):
            os.mkdir(upload_dir)
        sub_upload_dir = os.path.join(upload_dir, self.source_type.name)
        if not os.path.exists(sub_upload_dir):
            os.mkdir(sub_upload_dir)
        fdatas = []
        odatas = []
        newest_datas = []
        time_start = self.test_date
        preday_index = times_df.index.get_loc(time_start) - 24 #前一天，计算滑动平均需要用
        for cindex in range(channels_df.shape[0]):
            fdata_path = os.path.join( test_dir , f'{self.source_type.name}_fData_c{cindex}.npy')
            odata_path = os.path.join( test_dir, f'{self.source_type.name}_oData_c{cindex}.npy')
            fdatas.append( np.load(fdata_path))
            odatas.append( np.load(odata_path))
            src_data = self.data_file.read_data(cindex, self.source_type, DataStep.src)
            newest_datas.append(np.full((odatas[cindex].shape[0]+24, odatas[cindex].shape[1],1),dtype=float,fill_value=np.nan))
            newest_datas[cindex][0:24,:,:] = src_data[preday_index:preday_index+24,:,:]  
        
        
        add_hours = newest_datas[0].shape[0] + 1 #最后一个为endTime准备，没数据的
        add_time_data = [self.min_date for _ in range(add_hours)]
        newest_begin_time = time_start + timedelta(days=-1)
        add_time_index = [newest_begin_time + timedelta(hours=hour) for hour in range(add_hours)]
        newest_time = pd.DataFrame(data=add_time_data, index=add_time_index,columns=['forecast_time'], dtype='datetime64[ns]')
        newest_time_path = os.path.join(stat_dir, 'newestTime.csv')
        newest_time.to_csv(newest_time_path)
        points_file, times_file, channels_file =self.data_file.get_meta_filename(self.source_type)
        shutil.copyfile(points_file, os.path.join(stat_dir, os.path.basename(points_file)))
        shutil.copyfile(times_file, os.path.join(stat_dir, os.path.basename(times_file)))
        shutil.copyfile(channels_file, os.path.join(stat_dir, os.path.basename(channels_file)))
        
        day_count = fdatas[0].shape[0]
        for result_file in ['fData','oData']:
            day_dfs = []
            for dindex in tqdm( range(day_count) , desc=f'{ test_data_type } {result_file} days', leave=True):
                action_time = time_start + timedelta(days = dindex, hours=-1)
                tstart = dindex*24+24
                for cindex in range(channels_df.shape[0]):
                    if result_file == 'fData':
                        newest_datas[cindex][tstart:tstart+_MAX_FORE_DAY*24,:,0] = fdatas[cindex][dindex,:,:]
                    else:
                        newest_datas[cindex][tstart:tstart+_MAX_FORE_DAY*24,:,0] = odatas[cindex][tstart-24:tstart-24+_MAX_FORE_DAY*24,:]
                    np.save( os.path.join(stat_dir, f'newestData_c{cindex}.npy'), newest_datas[cindex])
                air_stat = AirStat(test_dir, self.source_type, action_time, action_time + timedelta(hours=7*24+1))
                csv_path = air_stat.stat_all()
                day_aqi = pd.read_csv(csv_path)
                day_aqi = day_aqi[day_aqi['stat_name'] == '城市日AQI']
                day_dfs.append(day_aqi)
                os.remove(csv_path)
            result_df = pd.concat(day_dfs)
            result_df.to_csv(os.path.join(test_dir, f'{self.source_type.name}_{result_file}.csv'))
    
    def plot_values(self, fdata, odata, channel_index, channel, test_data_type, test_dir):
        days = fdata.shape[0]
        days_add = range(days)
        time_start = self.test_date
        x = [time_start+timedelta(days=d) for d in days_add]
        y=np.arange(0, fdata.shape[1], 1, dtype=int)
        fig = plt.figure(figsize=(20,4))

        # missing
        if channel_index ==  0:
            plot_datas = ~np.isnan( fdata )
            plot_datas = np.count_nonzero( plot_datas, axis=2)
            fig.clf()
            ax=fig.add_axes([0.1,0.1,0.8,0.8],facecolor='black')
            ax.set_title(f"{test_data_type} data missing")
            ax.set_xlabel("date")
            ax.set_ylabel("forecast hour" )
            cmap = plt.get_cmap('viridis')
            # cmap.set_bad('black')
            im=ax.pcolor(x,y,plot_datas.swapaxes(0,1))#, edgecolor='white')
            fig.colorbar(im, ax=ax, cmap=cmap)
            plt.xticks(rotation = 30)
            fig.savefig(os.path.join(test_dir,f'{self.source_type.name}_{test_data_type}_missing.png'), bbox_inches='tight')

        # error
        #datas [days,T(168 hours), V, C]  , 每一个day的数据为上一天预测的今天至7天的数据
        row = channel
        plot_datas = np.nanmean(  fdata[0:fdata.shape[0],:,:] , axis=2)
        x = [time_start+timedelta(days=d) for d in range(plot_datas.shape[0])]
        # self.calcInoutside(timeStart, plotDatas, row, maxValue, minValue)
        fig.clf()
        ax=fig.add_axes([0.1,0.1,0.8,0.8],facecolor='black')
        ax.set_title(f"{test_data_type} {row['symbol']} data error")
        ax.set_xlabel("date")
        ax.set_ylabel("hour" )
        cmap = plt.get_cmap('viridis')
        # cmap.set_bad('black')
        im=ax.pcolor(x,y,plot_datas.swapaxes(0,1))#, edgecolor='white')
        fig.colorbar(im, ax=ax, cmap=cmap)
        plt.xticks(rotation = 30)
        fig.savefig(os.path.join(test_dir,f'{self.source_type.name}_{test_data_type}_error_c{channel_index}.png'), bbox_inches='tight')


    def assess_hj1130(self, test_data_type, pred_day, channel_name, F1, O1, csv_lines):
        #去掉nan
        coor = ~np.isnan(O1)
        O1 = O1[coor]
        F1 = F1[coor]
        coor = ~np.isnan(F1)
        O1 = O1[coor]
        F1 = F1[coor]
        NMB = np.sum( (F1-O1))/np.sum(O1)
        csv_lines.append({'测试数据':test_data_type,
                         '评估因子':'标准化平均偏差',
                         '预测天数':pred_day+1, 
                         '监测因子':channel_name,
                         '评估值':NMB,
                        })
        NME = np.sum( np.abs(F1-O1))/np.sum(O1)
        csv_lines.append({'测试数据':test_data_type,
                         '评估因子':'标准化平均误差',
                         '预测天数':pred_day+1, 
                         '监测因子':channel_name,
                         '评估值':NME,
                        })        
        MSE = np.square(np.subtract(O1,F1)).mean() 
        RMSE = math.sqrt(MSE)        
        csv_lines.append({'测试数据':test_data_type,
                         '评估因子':'均方根误差',
                         '预测天数':pred_day+1, 
                         '监测因子':channel_name,
                         '评估值':RMSE,
                        })
        CCR = np.corrcoef(O1,F1)
        csv_lines.append({'测试数据':test_data_type,
                         '评估因子':'相关系数',
                         '预测天数':pred_day+1, 
                         '监测因子':channel_name,
                         '评估值':CCR[0,1],
                        })
        return NME

if __name__ == '__main__':
    now = datetime.now()

    ap = argparse.ArgumentParser()
    ap.add_argument("-dp", "--data_dir", required=True,default="data/Air", help="path to the .pkl dataset, e.g. 'data/Dutch/dataset.pkl'")
    ap.add_argument("-e", "--epochs", type=int, default=50, help="Number of epochs")
    ap.add_argument("-ps", "--predict_timesteps", type=str, required=True, default='3,6,12,24', help="number of timesteps (hours) ahead for the prediction")
    ap.add_argument("-bt", "--base_timesteps", required=True, type=int,default=120, help="batch size for dataloader'")
    ap.add_argument("-bat", "--batch_size", required=True, type=int,default=64, help="batch size for dataloader'")
    ap.add_argument("-id", "--input_predays", type=int, required=True, help="number of days to be included in the input")
    ap.add_argument("-c", "--channel_index", required=False,  type=int,default=0, help="channel Index'")
    ap.add_argument("-cr", "--channel_roll", required=False,  type=int,default=0, help="channel_roll'")
    ap.add_argument("-src", "--source_type", required=True, type=str,default=1, help="数据源'")
    ap.add_argument("-ws", "--weather_source", required=False, help="config")
    ap.add_argument("-vs", "--vs_dir", default='/datapool01/shared/vs', required=False, help="vs model Data dir")
    ap.add_argument("-md", "--model_dir", default='', required=False, help="vs model Data dir")
    ap.add_argument("-dc", "--db_config", required=True, help="config")
    ap.add_argument("-db", "--db", required=True, help="config")
    ap.add_argument("-td", "--train_date", required=True, help="")
    ap.add_argument("-ts", "--test_date", required=True, help="")
    ap.add_argument("-ed", "--end_date", required=True, help="")
    ap.add_argument("-pg", "--param_groud_id", required=False,  type=int,default=1, help="param_groud_id")
    ags, unknowns = ap.parse_known_args()
    arguments = vars(ags)

    logger = _setup_logger(os.path.join( arguments["data_dir"],'log') )

    tester = Tester(arguments["data_dir"], 
                    arguments, 
                    SourceType[ arguments["source_type"] ], 
                    SourceType[ arguments["weather_source"] ], 
                    arguments["vs_dir"], 
                    arguments["channel_index"],
                    arguments["model_dir"], 
                    datetime.fromisoformat( arguments["train_date"]),
                    datetime.fromisoformat( arguments["test_date"]),
                    datetime.fromisoformat( arguments["end_date"])
                    )

    tester.all_days_test(arguments["param_groud_id"], arguments["db_config"], arguments["db"])

    # trainTime,lastEpoch, usedMemory, modelFileName = '1:22:21.226327',  96, 5887, None
    # tester.testOne( arguments["predict_timesteps"], arguments["batch_size"], arguments["input_predays"],
    #                arguments["channel_index"], trainTime, lastEpoch, usedMemory, modelFileName)

    # tester.test()
    # tester.plot()
    # tester.showGraph()



    timeUsed = datetime.now()-now
    tester.logger.info(f"test 用时： {str(timeUsed)}")
