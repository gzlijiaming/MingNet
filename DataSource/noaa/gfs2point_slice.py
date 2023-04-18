from cmath import pi
from importlib import metadata
import ray
import numpy as np
import socket
import os,sys
import pandas as pd
from datetime import datetime, timedelta, date
from pykrige.ok import OrdinaryKriging

_DEBUG_TIME = False
_POINT_SIZE = 0.25

def _simple_interpolation( xmin,ymin,zgrid, xs, ys):
    result = np.empty((xs.shape[0]),dtype=float)
    for pindex in range(result.shape[0]):
        x=xs[pindex]
        y=ys[pindex]
        xindex = int( (x-xmin) // _POINT_SIZE )
        yindex = int( (y-ymin) // _POINT_SIZE )
        value = _calc_idw( np.array([0,0,_POINT_SIZE,_POINT_SIZE]),
                        np.array([0,_POINT_SIZE,0,_POINT_SIZE]),
                        np.array([zgrid[xindex,yindex], zgrid[xindex,yindex+1], zgrid[xindex+1,yindex], zgrid[xindex+1,yindex+1],]),
                        x,
                        y)
        result[pindex] = value
    return result

def _calc_idw(xs,ys,zs,x,y):
    weights = 1/ np.sqrt((xs-x)*(xs-x)+(ys-y)*(ys-y))
    return np.average(zs, weights=weights)

def grid2point_slice( data_dir, grid_file,  inter_hours):
    hostname = socket.gethostname()
    # 通过插值，计算监测点位的气象数据
    # grid第一个点为左上角的点，先lon，后lat
    errors = []
    grid_lonlat = np.load(os.path.join(data_dir, 'gfs_grid_lonlat.npy'))
    xmin = grid_lonlat[:,0].min()
    ymin = grid_lonlat[:,1].min()
    #监测点的坐标s
    point_lonlat = np.load(os.path.join(data_dir, 'gfs_point_lonlat.npy'))
    grid_path = os.path.join(data_dir,'rda_slice',grid_file)
    result_path = os.path.join(data_dir,'zpoint_slice',grid_file)
    #结果文件已经有,而且大过1M，就不用做了
    if os.path.exists(result_path) and os.path.getsize(result_path)>512*1024  and not _DEBUG_TIME:   
        return errors
    if os.path.isfile(grid_path):
        channelsDf = pd.read_csv(os.path.join(data_dir, 'meta_channels.csv'), index_col=0)
        #gridDatas [Day(0),T, lon, lat, C]
        grid_datas = np.load(grid_path)
        grid_datas = np.flip(grid_datas, axis=3)
        #resultDatas [T, V, C]， 最后一个channel用于测试，不用，所以减一。
        result_datas= np.empty((grid_datas.shape[1] * inter_hours,point_lonlat.shape[0],grid_datas.shape[4]), dtype=float)
        if _DEBUG_TIME:
            ##### 调试时间值，将时间放进去，后面用的时候看时间是否准确
            time_zone = 8
            slice_begin_time = datetime(int(grid_file[:4]), int(grid_file[4:6]), int(grid_file[6:8]), int(grid_file[9:11]) + time_zone+1)
            for i in range(result_datas.shape[0]):
                current_time = slice_begin_time+timedelta(hours=i)
                # currentInt =int( currentTime.strftime('%y%m%d%H'))
                current_int =int( current_time.strftime('%m%d%H'))
                result_datas[i,:,:] = current_int/ 1e6
        else:
            ##### 正常值
            result_datas.fill(np.nan)
            for channel_index in range(result_datas.shape[2]):
                for grid_tindex in range(grid_datas.shape[1]):
                    point_begin = grid_tindex*inter_hours 
                    point_end = point_begin +inter_hours
                    # split都做，join 的时候再挑
                    try:
                        zgrid = grid_datas[0,grid_tindex,:,:,channel_index]
                        #将异常值去除
                        if np.max(zgrid)>np.min(zgrid):
                            #有差值才需要做，有数值的话，max>min
                            pointz = _simple_interpolation(xmin,ymin, zgrid,point_lonlat[:,0], point_lonlat[:,1] )
                        else:
                            #都是一个值的话，填满都是这个值
                            pointz = np.empty((point_lonlat.shape[0]),dtype=result_datas.dtype)
                            pointz.fill(np.max(zgrid))
                        for dst_index in range(point_begin, point_end):
                            result_datas[dst_index,:,channel_index] = pointz
                    except Exception as e:
                        err_msg = f'inner error@{hostname} {grid_file} tIndex={grid_tindex} channelIndex={channel_index} \
                            错误：{e}'
                        print(err_msg)
                        errors.append(err_msg)
        try:
            np.save(result_path,result_datas)
        except Exception as e:
            errors.append(f'grid2point_slice @{hostname} save file {result_path} \
                错误：{e}')
    return  errors

