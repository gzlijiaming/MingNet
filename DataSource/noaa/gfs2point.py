# conda install cartopy

import argparse
from asyncio.log import logger
from locale import normalize
import logging
import math
from sched import scheduler
import shutil
import socket
from xml.etree.ElementPath import iterfind
import os,sys

from matplotlib.pyplot import grid
from zmq import PROTOCOL_ERROR_ZAP_INVALID_METADATA
sys.path.append("DataPrepare") 
from data_file import DataFile, DataStep,SourceType,ChannelSplit
# import args
# from DataSource.gd.db_reader import DbReader
import numpy as np
import json
from datetime import datetime, timedelta, date
from tqdm import tqdm
import ray
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import gfs2point_slice as Gfs2pointSliceModule
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy

_MAX_FORE_DAYS = 7
_DEBUG_TIME = False


def _setup_logger(log_dir):
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
    fhlr = logging.FileHandler(os.path.join( log_dir,'log',f'gfs2point_{hostname}.log')) # 输出到文件的handler
    fhlr.setFormatter(formatter)
    fhlr.setLevel('DEBUG')
    logger.addHandler(chlr)
    logger.addHandler(fhlr)
    return logger


@ray.remote(num_cpus=1,memory=15*1073741824)
def grid2_point_slice_remote( data_dir, gridFile,  interHours):
    import gfs2point_slice as Gfs2pointSliceModule
    err = Gfs2pointSliceModule.grid2point_slice( data_dir, gridFile,  interHours)
    return err

#让ray任务每完成一个叫一次
def to_iterator(obj_ids):
    while obj_ids:
        done, obj_ids = ray.wait(obj_ids)
        yield  ray.get(done[0])




def plot_grid( data_dir, grid_file):

    grid_path = os.path.join(data_dir,'rda_slice',grid_file)
    slice_datas = np.load(grid_path)     #[day, foreHour/3, lon, lat, channel]
    channels_df = pd.read_csv(os.path.join(data_dir, f'meta_channels.csv'), index_col=0)
    #gridDatas [Day(0),T, lon, lat, C]
    grid_lonlat = np.load(os.path.join(data_dir, 'gfs_grid_lonlat.npy'))
    # x=grid_lonlat[:,0]
    # y=grid_lonlat[:,1]
    plt.ioff()
    fig = plt.figure(figsize=(10,4))
    
    #只查有问题的channel
    for channel_index in range (0, channels_df.shape[0]):  
        channel_data=slice_datas[0,:,:,:,channel_index]
        plot_datas = channel_data.reshape((channel_data.shape[0],channel_data.shape[1]*channel_data.shape[2]))
        # plotDatas = channelData[:,:,0]
        fig.clf()
        ax=fig.add_axes([0.1,0.1,0.8,0.8],facecolor='black')
        ax.set_title(f"{grid_file}  {channels_df.iloc[channel_index]['name']} ")
        ax.set_xlabel("点位")
        ax.set_ylabel("预测时间" )
        cmap = plt.get_cmap('viridis')
        # cmap.set_bad('black')
        im=ax.pcolor(plot_datas)
        fig.colorbar(im, ax=ax, cmap=cmap)
        plt.xticks(rotation = 30)
        fig.savefig(os.path.join(data_dir,'rda_plot',f'{grid_file[0:11]}_c{channel_index}.png'), bbox_inches='tight')





class Gfs2Point():
    def __init__(self, data_dir, weather_source, gfs_root_dir, source_type, production_dir) :
        # plt.rcParams["font.sans-serif"]=["HYJunhei"] #设置中文字体
        plt.rcParams["axes.titlesize"]=10 #设置标题字体
        plt.rcParams["axes.unicode_minus"]=False #该语句解决图像中的“-”负号的乱码问题
        # plt.rcParams["figure.figsize"]=(8,4.5) #设置图大小
        plt.rcParams["figure.dpi"]=300 #设置图大小
        plt.rcParams["xtick.labelsize"]=6
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
            
        self.data_dir = data_dir
        self.gfs_root_dir = gfs_root_dir
        self.weather_source = weather_source
        self.data_file = DataFile(data_dir)
        self.source_type = source_type
        self.production_dir = production_dir
        self.logger = logging.getLogger()
        dirname, filename = os.path.split(os.path.abspath(__file__))
        with open(os.path.join(dirname, f'rda_tds_{weather_source.name}.json'), 'r') as f:
            self.config = json.load(f)
        self.gfs_dir = os.path.join(gfs_root_dir, weather_source.name)
        self.newest_data_path = os.path.join(data_dir, f'{weather_source.name}_newestWeatherData.npy')
        self.newest_time_path = os.path.join(data_dir, f'{weather_source.name}_newestWeatherTime.csv')
        self.min_date =datetime(1999,1,1)

        self.channel_inv_names =[var[0] for var in self.config['vars']]
        self.slice_size = 24
        self.gd_lon_lat = self.config['gridLonLat']
        self.point_size = self.config['pointSize']
        self.gd_block={'y':{'start':int((90-self.gd_lon_lat['north'])/self.point_size),
                    'end': int((90-self.gd_lon_lat['south'])/self.point_size)+1},
                'x': {'start':int((self.gd_lon_lat['west'])/self.point_size),
                    'end': int((self.gd_lon_lat['east'])/self.point_size)+1}}
        self.lon_count = self.gd_block['x']['end']-self.gd_block['x']['start']
        self.lat_count = self.gd_block['y']['end']-self.gd_block['y']['start']
        self.time_zone = self.config['timeZone']
        self.start_time = datetime.fromisoformat(self.config['startDate'])
        self.end_time = datetime.fromisoformat(self.config['endDate'])
        # self.initNewestData()        


    def plot_map(self, gfs_dir, mon_air, slice_name):
        grid_datas = np.load(os.path.join(gfs_dir, 'rda_slice', slice_name))     #[day, foreHour/3, lon, lat, channel]
        # gridDatas = np.flip(gridDatas, axis=3)
        point_datas = np.load(os.path.join(gfs_dir, 'zpoint_slice', slice_name))     #[day, foreHour/3, lon, lat, channel]
        point_metas = DataFile(mon_air).read_meta(self.source_type)
        inter_hours = self.config['interHours']
        
        points_df = point_metas['points']
        plot_dir = os.path.join( gfs_dir,'plot')
        if not os.path.exists(plot_dir):
            os.mkdir(plot_dir)
        x=np.arange(self.gd_lon_lat['west'],self.gd_lon_lat['east']+self.point_size,self.point_size)
        y=np.arange(self.gd_lon_lat['north'],self.gd_lon_lat['south']-self.point_size,-self.point_size)
        plt.ioff()
        fig = plt.figure()

        #只查有问题的channel
        # for cIndex in [1]: 
        for cindex in range (0, grid_datas.shape[4]):  
            for hindex3 in tqdm(range(12), desc=f"{slice_name} c{cindex}：", leave=False):
                block_data = grid_datas[0,hindex3,:,:,cindex]
                # logger.info(f"{slicename} grid c{cIndex} h{hIndex3}:min={ blockData.min():.2f} max={blockData.max():.2f}")
                fig.clf()
                ax_position = [0.1,0.1,0.8,0.8]
                ax = fig.add_subplot(1,1,1, projection=ccrs.PlateCarree())
                fig.subplots_adjust(ax_position[0],ax_position[1],ax_position[2],ax_position[3])
                ax.coastlines()
                # zorder can be used to arrange what is on top
                ax.add_feature(cartopy.feature.COASTLINE, zorder=4)   # land is specified to plot above ...
                ax.add_feature(cartopy.feature.OCEAN, zorder=1)  # ... the ocean
                # ax=fig.add_axes(axPosition)
                ax.set_title(f"{slice_name} c{cindex} h{hindex3}\n{self.config['vars'][cindex][0]} ({self.config['vars'][cindex][1]})")
                ax.set_xticks(np.arange(-180, 180 + 60, 1))
                ax.set_yticks(np.arange(-90, 90 + 30, 1))
                cmap = plt.get_cmap('viridis')
                im=ax.pcolormesh(x,y, block_data.T,vmin=block_data.min(),vmax=block_data.max(), cmap=cmap, zorder=3, edgecolor='white', linewidth=4)

                lon = points_df['lon']
                lat = points_df['lat']
                c = point_datas[hindex3* inter_hours,:,cindex]
                # logger.info(f"{slicename} point c{cIndex} h{hIndex3}:min={ c.min():.2f} max={c.max():.2f}")
                # c[:] = 0.8
                ax.scatter(lon, lat, s=2, c=c,vmin=block_data.min(),vmax=block_data.max(), label='point', cmap=cmap, zorder=5)
                
                fig.colorbar(im, ax=ax)
                fig.savefig( os.path.join( plot_dir,f'{slice_name}_c{cindex:02}_h{hindex3:02}.png'), bbox_inches='tight')


        

    def write_metas(self):
        #保存metaData
        channels=[]
        # [id, 名称，代号，最大值，最小值]
        for index,var in enumerate( self.config['vars']):
            channels.append([var[4],var[0],var[1],var[2],var[3]])
        # 最后一个只是用来debug，不用加进去，"Land_cover_0__sea_1__land_surface","陆地" 
        # channelsDf = pd.DataFrame(data=channels[0:-1], columns=['cid',  "name", 'symbol',  "maxValue",  "minValue"])
        channels_df = pd.DataFrame(data=channels, columns=['cid',  "name", 'symbol',  "maxValue",  "minValue"])
        channels_df.set_index('cid',inplace=True)

        times =[]
        #时间  [tid, code, name, startTime, endTime]
        one_hour = timedelta(days=1)
        current_time = self.start_time
        while current_time < self.end_time:
            code = f"{current_time.strftime('%Y%m%d')}_{current_time.strftime('%H')}"
            tid =code
            times.append([tid,code,f"{current_time.strftime('%Y/%m/%d %H:%M:%S')}",current_time,current_time+one_hour])
            current_time += one_hour
        times_df = pd.DataFrame(data=times, columns=['tid', 'code', 'name', 'startTime', 'endTime'])
        times_df.set_index('tid',inplace=True)

        point_metas = self.data_file.read_meta(self.source_type)
        points_df = point_metas['points']

        metas={'channels':channels_df, 'times':times_df, 'points':points_df}
        self.data_file.write_meta(metas, self.weather_source)

    def split_make(self):
        #用ray把插值任务分出去
        object_ids = []
        results =[]
        errors = []
        dirname, filename = os.path.split(os.path.abspath(__file__))
        # ray.init(address='ray://localhost:10001', include_dashboard=False, 
                #  runtime_env={"working_dir": dirname , "excludes":['*/']})
        point_metas = self.data_file.read_meta(self.source_type)
        points_df = point_metas['points']
        grid_metas = self.data_file.read_meta(self.weather_source)
        channels_df = grid_metas['channels']
        self.logger.info(f'begin split')
        # 保存坐标数据，让多个worker可用
        #气象点的坐标s
        grid_lon=np.zeros((self.lon_count*self.lat_count),dtype=float)
        grid_lat=np.zeros((self.lon_count*self.lat_count),dtype=float)
        for lon_index in range(0,self.lon_count):
            for lat_index in range(0,self.lat_count):
                grid_lon[lon_index*self.lat_count+lat_index] = self.gd_lon_lat['west']+self.point_size*lon_index
                grid_lat[lon_index*self.lat_count+lat_index] = self.gd_lon_lat['north']-self.point_size*lat_index
        np.save(os.path.join(self.gfs_dir, 'gfs_grid_lonlat.npy'), 
            np.c_[grid_lon,grid_lat])
        #监测点的坐标s
        point_lon =np.array(points_df['lon'].astype(float))
        point_lat =np.array(points_df['lat'].astype(float))
        np.save(os.path.join(self.gfs_dir, f'gfs_point_lonlat.npy'), 
            np.c_[point_lon,point_lat])
        #channel 最大最小值
        channels_df.to_csv(os.path.join(self.gfs_dir, f'meta_channels.csv'))

        slice_dir = os.path.join(self.gfs_dir,'rda_slice')
        inter_hours = self.config['interHours']
        split_dst_dir = os.path.join(self.gfs_dir,'zpoint_slice')
        if not os.path.exists(split_dst_dir):
            os.mkdir(split_dst_dir)
        files=os.listdir(slice_dir)
        files.sort()
        file_bar = tqdm(files, desc='split file: ', leave=False)
        for file in file_bar:
            file_bar.set_postfix_str(file,refresh=True)
            date_part = file[0:8]
            cycle_part = file[9:11]
            if file.endswith('.npy') and date_part.isnumeric and cycle_part.isnumeric:
                Gfs2pointSliceModule.grid2point_slice(self.gfs_dir, file,  inter_hours )
                # object_ids.append( grid2PointSliceRemote.remote(self.gfs_dir, file,  interHours))

        #所有任务完成后，重新组装
        for x in tqdm(to_iterator(object_ids), total=len(object_ids)):
            if len(x)>0 :
                self.logger.error(x)
                errors.extend(x)

        errorstr = '\n'.join( errors)
        self.logger.info(f"error count={len(errors)}, errors:\n{errorstr}")
        return 

    def join_result(self ):
        grid_start_date = self.start_time
        grid_end_date = self.end_time
        grid_metas = self.data_file.read_meta(self.weather_source)
        point_metas = self.data_file.read_meta(self.source_type)
        points_df = grid_metas['points']
        channels_df = grid_metas['channels']
        # pointBeginDate =date.fromisoformat( pointMetas['times'].iloc[0]['startTime'][0:10])
        day_count = (grid_end_date - grid_start_date).days+1
        # resultDatas [days,T(168 hours), V, C]  , 每一个day的数据为上一天预测的今天至7天的数据
        result_datas= np.empty((day_count, _MAX_FORE_DAYS*24,points_df.shape[0], channels_df.shape[0] ), dtype=float)
        result_datas.fill(np.nan)
        slice_dir = os.path.join(self.gfs_dir,'zpoint_slice')
        files=os.listdir(slice_dir)
        files.sort()
        fill_bar = tqdm(files, desc='join file: ', leave=False)
        for file in fill_bar:
            fill_bar.set_postfix_str(file,refresh=True)
            if file.endswith('.npy') and file[0:8].isnumeric and file[9:11].isnumeric:
                slice_date = datetime(int(file[0:4]), int(file[4:6]), int(file[6:8]))
                model_cycle_runtimes = int(file[9:11])
                grid_begin_tindex = (24 - (model_cycle_runtimes+self.time_zone)) % 24 -1
                #因为气象数据是预测明天，所以加一
                day_index = (slice_date - grid_start_date).days + 1
                if day_index>=0 and day_index<result_datas.shape[0]:
                    point_slice_data = np.load(os.path.join(slice_dir,file))
                    # pointSliceData [T, V, C]
                    result_datas[day_index,:,:,:] = point_slice_data[grid_begin_tindex:grid_begin_tindex+_MAX_FORE_DAYS*24,:,:]
                    # self.slice2Newest(pointSliceData, sliceDate+timedelta(hours=model_cycle_runtimes+self.timeZone) )

        self.data_file.write_data(0,result_datas, self.weather_source, DataStep.src)

        # write back min max
        for row_index,row in channels_df.iterrows():
            channel_index = channels_df.index.get_loc(row_index)
            channel_datas = result_datas[:,:,:,channel_index]
            hour_max = np.nanmax(channel_datas)
            hour_min = np.nanmin(channel_datas)
            hour_avg = np.nanmean(channel_datas)
            normal_max = hour_max*2-hour_avg
            normal_min = hour_min*2-hour_avg
            channels_df.at[row_index,'minValue']  = normal_min
            channels_df.at[row_index,'maxValue']  = normal_max
            # 显示均值
            self.logger.info( f"\t\tavg:{hour_avg:.2f}  max:{hour_max:.2f}  min:{hour_min:.2f}, 归一化max,min： {normal_max:.2f}, {normal_min:.2f}")
        # write back min max
        self.data_file.write_meta( grid_metas, self.weather_source)

    def slice2newest(self, slice_datas, forecast_time):
        if forecast_time< self.start_time:
            return
        #datas [days,T(168 hours), V, C]  , 每一个day的数据为上一天预测的今天至7天的数据
        dst_start = self.newest_time.index.get_loc(forecast_time) + 1
        for tindex in range(slice_datas.shape[0]):
            monitor_time = forecast_time + timedelta(hours=tindex+1)
            dstIndex = dst_start + tindex
            if dstIndex >=0 and dstIndex<self.newest_datas.shape[0]:
                self.newest_time.loc[monitor_time,'forecast_time'] = forecast_time
                for pIndex in range(slice_datas.shape[1]):
                    for cIndex in range(slice_datas.shape[2]):
                        if not np.isnan(slice_datas[tindex, pIndex, cIndex]):
                            self.newest_datas[dstIndex,:,:] = slice_datas[tindex, pIndex, cIndex]

    def channel2newest(self, cIndex, channel_datas):
        for day in range(channel_datas.shape[0]):
            slice_datas = channel_datas[day,:,:]
            dst_start = day * 24
            for tindex in range(slice_datas.shape[0]):
                dst_index = dst_start + tindex
                if dst_index >=0 and dst_index<self.newest_datas.shape[0]:
                    for pindex in range(slice_datas.shape[1]):
                        if not np.isnan(slice_datas[tindex, pindex]):
                            self.newest_datas[dst_index,pindex,cIndex] = slice_datas[tindex, pindex]


    def init_newest_data(self):
        point_metas = self.data_file.read_meta(self.source_type)
        points_df = point_metas['points']
        # gridMetas = self.dataFile.readMeta(self.weather_source)
        channels_df = pd.read_csv(os.path.join(self.gfs_dir, 'meta_channels.csv'))
        # pointBeginDate =date.fromisoformat( pointMetas['times'].iloc[0]['startTime'][0:10])
        hours = ((self.end_time- self.start_time).days + _MAX_FORE_DAYS +1)* 24
        # newestDatas [T, V, C]
        self.newest_datas= np.empty((hours, points_df.shape[0], channels_df.shape[0] ), dtype=float)
        self.newest_datas.fill(np.nan)
        
        add_time_data = [self.min_date for _ in range(hours)]
        add_time_index = [self.start_time + timedelta(hours=hour) for hour in range(hours)]
        self.newest_time = pd.DataFrame(data=add_time_data, index=add_time_index,columns=['forecast_time'], dtype='datetime64[ns]')



    def plot_slice(self, file, inter_hours):
        fig = plt.figure(figsize=(20,8))
        datas = np.load(os.path.join(self.gfs_dir,"rda_slice", file))

        # missing
        cindex = 1
        plot_datas =datas[0, :, 0, :, cindex]
        x=np.arange(0, plot_datas.shape[0], 1, dtype=int)
        y=np.arange(0, plot_datas.shape[1], 1, dtype=int)
        fig.clf()
        ax=fig.add_axes([0.1,0.1,0.8,0.8],facecolor='black')
        ax.set_title(f"{file} c{cindex} data error")
        ax.set_xlabel("time")
        ax.set_ylabel("grid point" )
        cmap = plt.get_cmap('viridis')
        # cmap.set_bad('black')
        im=ax.pcolor(x,y,plot_datas.swapaxes(0,1))
        fig.colorbar(im, ax=ax, cmap=cmap)
        plt.xticks(rotation = 30)
        fig.savefig(os.path.join(self.gfs_dir,'plot',f'{file}_error_c{cindex}.png'), bbox_inches='tight')


    def plot(self):
        fig = plt.figure(figsize=(20,4))
        channels=[]
        # [id, 名称，代号，最大值，最小值]
        for index,var in enumerate( self.config['vars']):
            channels.append([f'weather{index:03}',var[0],var[1],var[2],var[3]])
        # 最后一个只是用来debug，不用加进去，"Land_cover_0__sea_1__land_surface","陆地" 
        channels_df = pd.DataFrame(data=channels, columns=['cid',  "name", 'symbol',  "maxValue",  "minValue"])
        channels_df.set_index('cid',inplace=True)
        metas = self.data_file.read_meta(self.weather_source)
        time_start = metas['times'].iloc[0]['startTime']

        # newest missing
        newest_datas = np.load(self.newest_data_path )
        # newestDatas = np.load(os.path.join(self.data_dir, f'{self.source_type.name}_newest_worktime.npy') )
        plot_datas = np.logical_not( np.isnan( newest_datas ))
        hours = plot_datas.shape[0]
        days = hours//24
        if hours>days*24:
            plot_datas = np.delete(plot_datas,range(days*24,hours),axis=0)
        plot_datas = np.reshape(plot_datas,[days,24,plot_datas.shape[1],plot_datas.shape[2]])  #把同一天的24小时竖起来，变成T*24*V
        plot_datas = np.sum(plot_datas,axis=1)    #压缩了24小时到一天
        plot_datas = np.sum(plot_datas,axis=2)    #压缩了channels
        # plotDatas[np.where(plotDatas<180)] = 180    #让问题更显眼
        y=np.arange(0, newest_datas.shape[1], 1, dtype=int)
        x = [time_start+timedelta(days=d) for d in range(days)]
        fig.clf()
        ax=fig.add_axes([0.1,0.1,0.8,0.8],facecolor='black')
        ax.set_title(f"{self.weather_source.name} newest missing")
        ax.set_xlabel("date")
        ax.set_ylabel("point" )
        cmap = plt.get_cmap('viridis')
        im=ax.pcolor(x,y,plot_datas.swapaxes(0,1))#, edgecolor='white')
        fig.colorbar(im, ax=ax, cmap=cmap)
        plt.xticks(rotation = 30)
        fig.savefig(os.path.join(self.data_dir,'plot',f'{self.weather_source.name}_newest_missing.png'), bbox_inches='tight')

        # newest error
        plot_datas = newest_datas[0:newest_datas.shape[0],:,:].copy()
        hours = plot_datas.shape[0]
        days = hours//24
        if hours>days*24:
            plot_datas = np.delete(plot_datas,range(days*24,hours),axis=0)
        plot_datas = np.reshape(plot_datas,[days,24,plot_datas.shape[1],plot_datas.shape[2]])  #把每的3小时竖起来，变成T*24*V
        plot_datas = np.average(plot_datas,axis=1)    #压缩到1/3
        x = [self.start_time+timedelta(hours=h*24) for h in range(plot_datas.shape[0])]
        for rowIndex,row in tqdm( channels_df.iterrows(), total=channels_df.shape[0], leave=True, desc=f'{DataStep.normalized.name} plot newest error'):
            channelIndex = channels_df.index.get_loc(rowIndex)
            if channelIndex >= plot_datas.shape[2]:
                continue
            vmin =0
            vmax =1
            #datas [days,T(168 hours), V, C]  , 每一个day的数据为上一天预测的今天至7天的数据
            plotData = plot_datas[:,:,channelIndex] 
            fig.clf()
            ax=fig.add_axes([0.1,0.1,0.8,0.8],facecolor='black')
            ax.set_title(f"{self.source_type.name} {row['name']} newest data error")
            ax.set_xlabel("date")
            ax.set_ylabel("point" )
            cmap = plt.get_cmap('viridis')
            # cmap.set_bad('black')
            im=ax.pcolor(x,y,plotData.swapaxes(0,1))#,vmin=vmin, vmax=vmax)#, edgecolor='white')
            fig.colorbar(im, ax=ax, cmap=cmap)
            plt.xticks(rotation = 30)
            fig.savefig(os.path.join(self.data_dir,'plot',f'{self.weather_source.name}_newest_error_c{channelIndex}.png'), bbox_inches='tight')



    def calc_inoutside(self, time_start, plot_datas, row, max_value, min_value):
        coords = np.logical_and(plot_datas <= max_value , plot_datas >= min_value)
        inside_count = np.count_nonzero(coords)
        outside_count= plot_datas.size-inside_count
        if outside_count>0:
            self.logger.info(f'{row["name"]} {plot_datas.size} - {inside_count} = {outside_count}')
                    # print(plotDatas[~coords])
            outside_coors = np.where(~coords)
            self.logger.info(f'[{outside_coors[0][0]},{outside_coors[1][0]}] {time_start+timedelta(days=int(outside_coors[0][0]))}')


    # 填缺失
    def pretreat_missing( self):
        metas = self.data_file.read_meta(self.weather_source)
        time_start = metas['times'].iloc[0]['startTime']
        #datas [days,T(168 hours), V, C]  , 每一个day的数据为上一天预测的今天至7天的数据
        datas = self.data_file.read_data(0, self.weather_source, DataStep.src)
        # 使用yesterday方法
        for i in tqdm( range(0, datas.shape[0]), desc='pretreatMissing', total=datas.shape[0]):
            if np.isnan( datas[i,:,:,:] ).any():
                print(f'#{i} {time_start + timedelta(days= i-1)} missing!')
                datas[i,0:-24,:,:] = datas[i-1,24:,:,:]
                if i+1<datas.shape[0]:
                    datas[i,-24:,:,:] = datas[i+1,-48:-24,:,:]
        self.data_file.write_data(0,datas,  self.weather_source,DataStep.time_treated)



    # 规格化
    def normalize(self):
        metas = self.data_file.read_meta(self.weather_source)
        channels_df = metas['channels']
        #datas [days,T(168 hours), V, C]  , 每一个day的数据为上一天预测的今天至7天的数据
        datas = self.data_file.read_data(0, self.weather_source, DataStep.time_treated)
        result_datas= np.empty((datas.shape[0],datas.shape[1], datas.shape[2],datas.shape[3]), dtype=datas.dtype)

        for row_index,row in channels_df.iterrows():
            channel_index = channels_df.index.get_loc(row_index)
            max_value=float( row['maxValue'])
            min_value=float( row['minValue'])
            if _DEBUG_TIME:
                ##### 调试时间，
                channel_datas = datas[:,:,:,channel_index]
            else:
                ##### 正常值
                self.logger.info(f"normalize {row['name']} using: {min_value} -> {max_value }")

                channel_datas = (datas[:,:,:,channel_index] - min_value)/(max_value-min_value)
                channel_datas[ np.where(channel_datas<0) ]=0
                channel_datas[ np.where(channel_datas>1) ]=1

            result_datas[:,:,:,channel_index] = channel_datas
            self.channel2newest(channel_index, channel_datas )

        self.data_file.write_data(0,result_datas,  self.weather_source,DataStep.normalized)

        np.save(self.newest_data_path,self.newest_datas )
        self.newest_time.to_csv(self.newest_time_path)
        # self.writeWorktime(  self.newestDatas , self.startTime , False)


    def write_production_data(self):
        dst_dir = os.path.join(self.production_dir,self.weather_source.name)
        if not os.path.exists(dst_dir):
            os.mkdir(dst_dir)
        shutil.copyfile(os.path.join(self.gfs_dir, 'gfs_grid_lonlat.npy'), os.path.join(self.production_dir,self.weather_source.name, 'gfs_grid_lonlat.npy'))
        shutil.copyfile(os.path.join(self.gfs_dir, 'gfs_point_lonlat.npy'), os.path.join(self.production_dir,self.weather_source.name, 'gfs_point_lonlat.npy'))
        shutil.copyfile(os.path.join(self.data_dir, f'{self.weather_source.name }_meta_channels.csv'), os.path.join(self.production_dir, self.weather_source.name, 'meta_channels.csv'))
        shutil.copyfile(os.path.join(self.data_dir, f'{self.source_type.name }_meta_points.csv'), os.path.join(self.production_dir, self.weather_source.name, 'meta_points.csv'))


if __name__ == '__main__':
    # pretreatSpatialSlice(self.gfs_dir, 9936, 9960 )
    program_begin = datetime.now()
    ap = argparse.ArgumentParser()
    ap.add_argument("-ro", "--data_dir", required=True, help="root")
    ap.add_argument("-gfs", "--gfs_root_dir", default='/datapool01/shared/ucar', required=True, help="GFS Data dir")
    ap.add_argument("-pd", "--production_dir", default='/datapool01/shared/production', required=False, help="GFS Data dir")
    ap.add_argument("-ws", "--weather_source", required=False, help="config")
    ap.add_argument("-st", "--source_type", required=False, help="config")

    args, unknowns = ap.parse_known_args()
    arguments = vars(args)

    data_dir = arguments['data_dir']
    logger = _setup_logger(data_dir)

    g2p = Gfs2Point(data_dir,SourceType[arguments['weather_source']], arguments['gfs_root_dir'],
                    SourceType[arguments['source_type']],arguments['production_dir'] )

    g2p.write_metas()
    g2p.split_make()
    g2p.join_result()
    g2p.init_newest_data()

    g2p.pretreat_missing()
    g2p.normalize()

    g2p.write_production_data()
    g2p.plot()  #需半小时，慎用



    logger.info(f"gfs read {arguments['weather_source']}用时： {str(datetime.now() - program_begin)}")
