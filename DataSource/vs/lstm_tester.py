import argparse
from curses import meta
import datetime
from dis import pretty_flags
import sys,os,glob
import logging
import socket
from time import sleep
import numpy as np
# import tushare as ts
import torch
from torch import nn
import matplotlib.pyplot as plt
import pandas as pd
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset
import tqdm
from torch.utils.data import DataLoader
from itertools import chain
from datetime import datetime, timedelta,time
sys.path.append("DataPrepare") 
from data_file import DataFile, DataStep, SourceType



_MAX_FORE_DAY = 7
_EPOCHS = 388

class LSTM_Regression(nn.Module):
    """
        使用LSTM进行回归
        
        参数：
        - input_size: feature size
        - hidden_size: number of hidden units
        - output_size: number of output
        - num_layers: layers of LSTM to stack
    """
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, _x):
        x, _ = self.lstm(_x)  # _x is input, size (seq_len, batch, input_size)
        s, b, h = x.shape  # x is output, size (seq_len, batch, hidden_size)
        x = x.view(s*b, h)
        x = self.fc(x)
        x = x.view(s, b, -1)  # 把形状改回来
        return x




def create_dataset(data, seq_len, input_size) :#-> (np.array, np.array):
    dataset_x, dataset_y= [], []
    for i in range(len(data)-seq_len-seq_len+1):
        _x = data[i:(i+seq_len), :]
        dataset_x.append(_x)
        _y = data[i+seq_len:i+seq_len+seq_len]
        dataset_y.append(_y)
    return (np.array(dataset_x), np.array(dataset_y))




class LstmTester():
    def __init__(self , data_dir, vs_dir, source_type, train_date, test_date , end_date):
        self.logger = logging.getLogger()
        self.data_dir = data_dir
        self.source_type = source_type
        self.vs_dir = vs_dir
        self.train_date = train_date
        self.test_date = test_date
        self.end_date = end_date
        self.batch_size = 120
        self.data_file = DataFile(data_dir)
        self.test_dir = os.path.join(vs_dir, 'lstm')
        if not os.path.exists(self.test_dir):
            os.mkdir(self.test_dir)
        self.tempDir = os.path.join( self.test_dir, 'temp')
        if not os.path.exists(self.tempDir):
            os.mkdir(self.tempDir)
        self.loss_function = nn.MSELoss()
        self.dev = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.scaler_min = torch.tensor([0.000e+00]).to(self.dev)
        self.scaler_max = torch.tensor([0.000e+00]).to(self.dev)

    
    def train(self):
        metas = self.data_file.read_meta(self.source_type)
        channels_df = metas['channels']
        for cindex in range(channels_df.shape[0]):
            data_close =  self.data_file.read_data(cindex,self.source_type, DataStep.normalized)[:,:,0]
            point_count = data_close.shape[1]
            data_close = np.float32(data_close)
            dataset_x, dataset_y = create_dataset(data_close, _MAX_FORE_DAY*24,point_count)

            # 划分训练集和测试集，70%作为训练集
            train_size = int(len(dataset_x) * 0.7)

            train_x = dataset_x[:train_size]
            train_y = dataset_y[:train_size]
            test_x = dataset_x[train_size:]
            test_y = dataset_y[train_size:]

            # 将数据改变形状，RNN 读入的数据维度是 (seq_size, batch_size, feature_size)


            # 转为pytorch的tensor对象
            
            
            model_filename = os.path.join(self.test_dir, f'{self.source_type.name}_model_{cindex}.pt')
            model = LSTM_Regression(point_count, # `8` is the number of hidden units in the LSTM.
                    8, output_size=point_count, num_layers=2).to(self.dev)
            
            train_x = torch.from_numpy(train_x).permute(1,0,2).to(self.dev)
            train_y = torch.from_numpy(train_y).permute(1,0,2).to(self.dev)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
            min_val_loss = 5
            for epoch in range(_EPOCHS):
                for batchStart in range(0,train_x.shape[1],self.batch_size):
                    out = model(train_x[:,batchStart:batchStart+self.batch_size,:])
                    loss = self.loss_function(out, train_y[:,batchStart:batchStart+self.batch_size,:])
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                if loss < min_val_loss:
                    min_val_loss = loss
                    torch.save(model.state_dict(), model_filename)
                if (epoch+1) % 50 == 0:
                    print(f'c{cindex} Epoch: {epoch+1}, Loss:{loss.item():.7f}')
                    
    def forecast_test_data(self, channelsDf, timesDf):
        end_index = timesDf.index.get_loc(self.end_date+timedelta(hours=-1))+1
        test_index = timesDf.index.get_loc(self.test_date)
        test_start_index  = test_index - _MAX_FORE_DAY*24 
        test_end_index  = end_index - _MAX_FORE_DAY*24 - _MAX_FORE_DAY*24
        result_count = _MAX_FORE_DAY*24
        #做预测

        for cindex in range(channelsDf.shape[0]):
            self.scaler_max[0] = float(channelsDf.iloc[cindex]['maxValue'])
            self.scaler_min[0] = float(channelsDf.iloc[cindex]['minValue'])
            data_close =  self.data_file.read_data(cindex,self.source_type, DataStep.normalized)[:,:,0]
            point_count = data_close.shape[1]
            data_close = np.float32(data_close)
            dataset_x, dataset_y = create_dataset(data_close, _MAX_FORE_DAY*24,point_count)
            

            test_x = dataset_x[test_start_index:test_end_index]
            test_y = dataset_y[test_start_index:test_end_index]
            test_x = torch.from_numpy(test_x).permute(1,0,2).to(self.dev)
            test_y = torch.from_numpy(test_y).permute(1,0,2).to(self.dev)
            # test_y = test_y * (self.scaler_max - self.scaler_min) + self.scaler_min
            model_filename = os.path.join(self.test_dir, f'{self.source_type.name}_model_{cindex}.pt')
            model = LSTM_Regression(point_count, # `8` is the number of hidden units in the LSTM.
                    8, output_size=point_count, num_layers=2).to(self.dev)
            model.load_state_dict(torch.load(model_filename))
            model = model.eval() # 转换成测试模式
            losss = []
            outs = np.empty((_MAX_FORE_DAY*24,0,point_count),dtype=np.float32)
            for batch_start in range(0,test_x.shape[1],self.batch_size):
                out = model(test_x[:,batch_start:batch_start+self.batch_size,:])# 全量训练集的模型输出 (seq_size, batch_size, output_size)
                odata = test_y[:,batch_start:batch_start+self.batch_size,:]
                losss.append(  self.loss_function(out, odata).item())
                out = out * (self.scaler_max - self.scaler_min) + self.scaler_min
                outs = np.concatenate((outs, out.cpu().detach().numpy()), axis=1)
            print(f'c{cindex} test loss { np.array(losss).mean() }')
            
            for tindex in range(0,outs.shape[1], 24):
                out = outs[:,tindex, :].reshape((outs.shape[0], outs.shape[2],1))
                action_time = self.test_date + timedelta(hours=tindex-1)
                result_path = os.path.join(self.tempDir, f'resultData_c{cindex}_{action_time:%Y%m%d_%H}.npy')
                np.save(result_path,np.float64( out))
        


    
if __name__ == '__main__':
    now = datetime.now()

    ap = argparse.ArgumentParser()
    ap.add_argument("-dp", "--data_dir", required=True,default="data/Air", help="path to the .pkl dataset, e.g. 'data/Dutch/dataset.pkl'")
    ap.add_argument("-e", "--epochs", type=int, default=50, help="Number of epochs")
    ap.add_argument("-ps", "--predict_timesteps", type=str, required=True, default='3,6,12,24', help="number of timesteps (hours) ahead for the prediction")
    ap.add_argument("-bat", "--batch_size", required=True, type=int,default=64, help="batch size for dataloader'")
    ap.add_argument("-id", "--input_predays", type=int, required=True, help="number of days to be included in the input")
    ap.add_argument("-c", "--channel_index", required=False,  type=int,default=0, help="channel Index'")
    ap.add_argument("-src", "--source_type", required=True, type=str,default=1, help="数据源'")
    ap.add_argument("-vs", "--vs_dir", default='/datapool01/shared/vs', required=False, help="vs model Data dir")
    ap.add_argument("-md", "--model_dir", default='', required=False, help="vs model Data dir")
    ap.add_argument("-td", "--train_date", required=True, help="")
    ap.add_argument("-ts", "--test_date", required=True, help="")
    ap.add_argument("-ed", "--end_date", required=True, help="")
    ags, unknowns = ap.parse_known_args()
    arguments = vars(ags)

    tester = LstmTester(arguments["data_dir"], 
                    arguments["vs_dir"], 
                    SourceType[ arguments["source_type"] ], 
                    datetime.fromisoformat( arguments["train_date"]),
                    datetime.fromisoformat( arguments["test_date"]),
                    datetime.fromisoformat( arguments["end_date"])
                    )
    tester.train()
    # lstmTester.forecastTestData()

    time_used = datetime.now()-now
    print(f"test 用时： {time_used}")
    