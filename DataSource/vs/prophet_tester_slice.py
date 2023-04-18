import os,sys
from datetime import datetime,date,time,timedelta
import pandas as pd
from prophet import Prophet
import numpy as np

def pre_one_point(data_dir,tempDir, source_type_name, channel_count, src_begin_index, src_end_index,action_time, result_count):
    stdout = os.dup(1)
    stderr = os.dup(2)
    null = os.open(os.devnull, os.O_RDWR)
    os.dup2(null, 1)
    os.dup2(null, 2)
    errors = []
    if source_type_name == 'guoKong':
        country_name = 'CN'
    elif source_type_name == 'ca':
        country_name = 'US'
    else:
        country_name = ''
    src = 'normalized' #这里不能用src，因为有些点位长期缺数，需要用已经空间插值的。因此，结果也要反规格化
    src_count = src_end_index-src_begin_index
    ds = pd.Series( [action_time+timedelta(hours=h - src_count + 1) for h in range(src_count)])  # 时间  [tid, code, name, startTime, endTime]
    channels_file=    os.path.join(data_dir , f'{source_type_name}_meta_channels.csv')
    channels_df = pd.read_csv(channels_file, index_col=0)
    for cindex in range(channel_count):
        result_path = os.path.join(tempDir, f'resultData_c{cindex}_{action_time:%Y%m%d_%H}.npy')
        # if os.path.exists(result_path) and os.path.getsize(result_path)>100*1024 :   
        #     continue
        data_file = os.path.join(data_dir , f'{source_type_name}_{src}_c{cindex}.npy' )
        datas = np.load(data_file) # [T, V, C]
        point_count = datas.shape[1]
        out_datas = np.empty((result_count, point_count , 1),dtype=float)
        scaler_max = float(channels_df.iloc[cindex]['maxValue'])
        scaler_min = float(channels_df.iloc[cindex]['minValue'])

        try:
            for pindex in range(point_count):
                y = pd.Series(datas[src_begin_index:src_end_index, pindex, 0])# [T, V, C]
                # y[y<=0] = np.nan
                data = pd.DataFrame({'ds':ds, 'y':y})
                m = Prophet(yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=True, changepoint_prior_scale=0.01)
                m.add_country_holidays(country_name=country_name)
                m.fit(data)
                future = m.make_future_dataframe(periods=result_count, freq='H',include_history=False)
                forecast = m.predict(future)
                # fig1 = m.plot(forecast)
                out_datas[:,pindex, 0] = forecast.yhat* (scaler_max - scaler_min) + scaler_min
            np.save(result_path, out_datas)
            
        except Exception as e:
            errors.append(f'{action_time} c{cindex} p{pindex} \
                错误：{e}')

    os.dup2(stdout, 1)
    os.dup2(stderr, 2)
    return  errors
