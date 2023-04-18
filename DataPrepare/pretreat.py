from datetime import datetime,date
import argparse
import socket
import numpy as np 
import pickle
import logging
from tqdm import tqdm
import ray
import os
from pykrige.ok import OrdinaryKriging
from data_file import DataFile, DataStep,SourceType,ChannelSplit
import pandas as pd
from geopy.distance import geodesic
import math

DEBUG_TIME = False
_SLICE_DAYS = 7
_SLICE_SIZE = _SLICE_DAYS*24



#广东中点大概是北纬23度，这个是经纬度变形的比例，在用经纬度坐标算空间插值时，需要通过这个比例，把经度压缩一下，使经纬度比较接近长度的比例
#但是，若coordinates_type='geographic'，则插值模块自己会当成地理坐标考虑，这里设置成1
_LAT_LON_RATIO = 1#math.cos(23*math.pi/180)

#让ray任务每完成一个叫一次
def _to_iterator(obj_ids):
    while obj_ids:
        done, obj_ids = ray.wait(obj_ids)
        yield ray.get(done[0])

#找数组中的连续元素
def _consecutive(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)        

@ray.remote(num_cpus=1,memory=15*1073741824)
def pretreat_spatial_slice_remote( data_dir, channel_index, points_metafile, channels_metafile, slice_path):
    return pretreat_spatial_slice( data_dir, channel_index,  points_metafile, channels_metafile, slice_path)

def pretreat_spatial_slice( data_dir, channel_index, points_metafile, channels_metafile, slice_path, dst_path = None):
    if dst_path is None:
        dst_path = slice_path
    errors = []
    points_df = pd.read_csv(points_metafile)
    channels_df = pd.read_csv(channels_metafile)
    subpoints_df = points_df[points_df['cid'].str.contains(ChannelSplit+str( channels_df.iloc[channel_index]['cid'] )+ChannelSplit)]
    datas = np.load(slice_path)

    kcount = 0
    for time_index in range(0, datas.shape[0]):
        one_data = datas[time_index,:,0]
        has_data_coor = np.nonzero(~np.isnan(one_data))
        # 如果有数据的，少于总数，则要补足数据
        data_count = has_data_coor[0].shape[0]
        if data_count>3 and data_count < one_data.shape[0]:
            kcount +=1
            try:
                non_data_coor = np.nonzero(np.isnan(one_data))[0]
                lon = subpoints_df.iloc[has_data_coor]['lon'].astype(float)*_LAT_LON_RATIO  #经度要压缩一定比例，使经纬度比例与实际长度比例接近
                lat = subpoints_df.iloc[has_data_coor]['lat'].astype(float)
                z = one_data[has_data_coor]
                non_lon =subpoints_df.iloc[non_data_coor]['lon'].astype(float)*_LAT_LON_RATIO #经度要压缩一定比例，使经纬度比例与实际长度比例接近
                non_lat =subpoints_df.iloc[non_data_coor]['lat'].astype(float)
                # Create ordinary kriging object:
                OK = OrdinaryKriging(
                    lon,
                    lat,
                    z,
                    variogram_model="linear",
                    # variogram_model="power",
                    # verbose=True,
                    # enable_plotting=True,
                    # drift_terms=["regional_linear"],
                    coordinates_type='geographic',
                    pseudo_inv=True,
                )
                # Execute on points:
                z1, ss1 = OK.execute("points", non_lon, non_lat)
                one_data[non_data_coor] = z1
            except Exception as e:
                errors.append(f'{slice_path}: timeIndex={time_index} channelIndex={channel_index} \
                    hasData:{has_data_coor[0].shape[0]} nonData: {non_data_coor.shape[0]}\
                    错误：{e}')
            # print( datas[timeIndex,:,channelIndex])
    np.save(dst_path, datas)
    return  errors

@ray.remote(num_cpus=8,memory = 30*1073741824)
def pretreat_time_slice_remote( data_dir, channel_index, point_index):
    return pretreat_time_slice( data_dir, channel_index, point_index)

def pretreat_time_slice( data_dir, slice_path, dst_path = None):
    if dst_path is None:
        dst_path = slice_path
    errors = []
    datas = np.load(slice_path)
    # 使用yesterday方法
    for i in range(0, datas.shape[0]):
        if np.isnan( datas[i,0] ):
            if i-24>=0:
                datas[i,0] = datas[i-24,0]
            else:
                datas[i,0] = datas[i-1,0]

    if np.isnan(datas).any():
        errors.append(f'{slice_path} stil has nan ')
    np.save(dst_path, datas)
    return  errors

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
    fhlr = logging.FileHandler(os.path.join( log_dir,f'pretreat_{hostname}.log')) # 输出到文件的handler
    fhlr.setFormatter(formatter)
    fhlr.setLevel('DEBUG')
    logger.addHandler(chlr)
    logger.addHandler(fhlr)
    return logger 

class pretreat():
    def __init__(self, data_dir,source_type, use_ray = False):
        self.data_dir = data_dir
        self.source_type = source_type
        self.use_ray = use_ray
        self.full_data_dir = os.path.abspath(data_dir)
        self.data_file = DataFile(data_dir)
        self.logger = logging.getLogger()


        # datas =datas[0:24*30,:,:]  #选取部分做测试
        self.metas = self.data_file.read_meta(source_type)
        # self.srcMetas = metas


    # 空间补足
    def pretreat_spatial(self):
        channels_df = self.metas['channels']
        slice_dir = os.path.join(self.data_dir,'ps_slice')
        if not os.path.exists(slice_dir):
            os.mkdir(slice_dir)
        for row_index,row in channels_df.iterrows():
            channel_index = channels_df.index.get_loc(row_index)

            datas = self.data_file.read_data(channel_index, self.source_type, DataStep.src)

            object_ids = []
            results =[]
            errors = []
            #分任务，每块10天，即240小时
            points_metafile, _, channels_metafile = self.data_file.get_meta_filename(self.source_type)
            # shutil.copy(pointsMetaFile, os.path.join(self.data_dir , 'ps_slice', 'meta_points.csv'))
            # shutil.copy(channelsMetaFile, os.path.join(self.data_dir , 'ps_slice', 'meta_channels.csv'))
            for time_index in tqdm( range(0, datas.shape[0], _SLICE_SIZE),unit_scale=_SLICE_DAYS,desc=f'空间插值 {row["name"]} days'):
                stop_index = min(time_index+_SLICE_SIZE, datas.shape[0])
                slice_data = datas[time_index:stop_index, :,:]
                slice_path = os.path.join(self.data_dir, 'ps_slice',f'c{channel_index}_{time_index:05}-{stop_index:05}.npy')
                np.save(slice_path,slice_data)
                if self.use_ray:                    
                    object_ids.append( pretreat_spatial_slice_remote.remote(self.full_data_dir, channel_index, points_metafile, channels_metafile, slice_path ))
                else:
                    x = pretreat_spatial_slice(self.full_data_dir,channel_index, points_metafile, channels_metafile, slice_path )
                    if len(x)>0 :
                        errors.extend(x)

            if self.use_ray:                    
                #所有任务完成后，重新组装
                for x in tqdm(_to_iterator(object_ids), total=len(object_ids),unit_scale=_SLICE_DAYS,desc=f'ray 空间插值 {row["name"]} days'):
                    if len(x)>0 :
                        errors.extend(x)

            errorstr = '\n'.join( errors)
            self.logger.info(f"{row['name']} error count={len(errors)}, errors:\n{errorstr}")
            result_datas= np.empty((0, datas.shape[1], datas.shape[2]), dtype=datas.dtype)
            for time_index in range(0, datas.shape[0], _SLICE_SIZE):
                stop_index = min(time_index+_SLICE_SIZE, datas.shape[0])
                slice_path = os.path.join(self.data_dir, 'ps_slice',f'c{channel_index}_{time_index:05}-{stop_index:05}.npy')
                slice_data=np.load(slice_path)
                result_datas = np.concatenate( (result_datas,slice_data),axis=0)
            
            self.data_file.write_data(channel_index, result_datas,self.source_type,DataStep.spatial_traeted)


    # 时间补足
    def pretreat_time(self):
        #T*V*C
        # pointsDf = self.metas['points']
        channels_df = self.metas['channels']
        for row_index,row in channels_df.iterrows():
            # subpointsDf = pointsDf[(pointsDf['cid']==row.name)]
            channel_index = channels_df.index.get_loc(row_index)

            datas = self.data_file.read_data(channel_index, self.source_type, DataStep.spatial_traeted)


            object_ids = []
            results =[]
            errors = []
            slice_dir = os.path.join(self.data_dir,'pt_slice')
            if not os.path.exists(slice_dir):
                os.mkdir(slice_dir)
            
            #分任务，每块一个点
            bar = tqdm(range(0, datas.shape[1]), desc=f'c{channel_index} 时间插值 point: ', leave=False)
            for point_index in bar:
                slice_data = datas[:, point_index,:]
                slice_path = os.path.join(self.data_dir, 'pt_slice',f'c{channel_index}_p{point_index:05}.npy')
                np.save(slice_path, slice_data)
                # object_ids.append( pretreatTimeSliceRemote.remote(self.fullDataDir, slicePath ))
                x = pretreat_time_slice(self.full_data_dir, slice_path )
                if len(x)>0 :
                    errors.extend(x)

            #所有任务完成后，重新组装
            # for x in tqdm(to_iterator(object_ids), total=len(object_ids)):
            #     if len(x)>0 :
            #         errors.extend(x)

            errorstr = '\n'.join( errors)
            self.logger.info(f"{row['name']} error count={len(errors)}, errors:\n{errorstr}")
            result_datas= np.empty((datas.shape[0],0, datas.shape[2]), dtype=datas.dtype)
            for point_index in range(0, datas.shape[1]):
                slice_path = os.path.join(self.data_dir, 'pt_slice',f'c{channel_index}_p{point_index:05}.npy')
                slice_data=np.load(slice_path)
                slice_data= slice_data.reshape(datas.shape[0],1, datas.shape[2])
                result_datas = np.concatenate( (result_datas,slice_data),axis=1)
            
            self.data_file.write_data(channel_index,result_datas, self.source_type,DataStep.time_treated)


    # 规格化
    def pretreat_normalizd(self):
        #T*V*C
        channels_df = self.metas['channels']

        for row_index,row in channels_df.iterrows():
            # subpointsDf = pointsDf[(pointsDf['cid']==row.name)]
            channel_index = channels_df.index.get_loc(row_index)
            datas = self.data_file.read_data(channel_index, self.source_type, DataStep.time_treated)
            max_value=float( row['maxValue'])
            min_value=float( row['minValue'])
            self.logger.info(f"normalize {row['name']} using: {min_value} -> {max_value }")
            result_datas= np.empty((datas.shape[0],datas.shape[1], 0), dtype=datas.dtype)
            if DEBUG_TIME:
                channel_datas = datas
            else:
                channel_datas = (datas - min_value)/(max_value-min_value)
            channel_datas[ np.where(channel_datas<0) ]=0
            channel_datas[ np.where(channel_datas>1) ]=1
            # channelDatas= channelDatas.reshape(datas.shape[0], datas.shape[1],1)
            result_datas = np.concatenate( (result_datas,channel_datas),axis=2)

            self.data_file.write_data(channel_index,result_datas,  self.source_type,DataStep.normalized)




    #全部做
    def pretreat_all(self):
        # self.calc_adj()
        # return
        if self.use_ray:
            ray.init(address='ray://localhost:10001', include_dashboard=False)
            # ray.init(address='auto', include_dashboard=False)#, runtime_env={"working_dir": os.path.dirname(os.path.abspath(__file__)), "excludes":[os.path.dirname(os.path.abspath(__file__))+'/__pycache__']})

    
        now = datetime.now()
        self.pretreat_spatial()
        time_used = datetime.now()-now
        print(f"空间补足 用时： {str(time_used)}")

        now = datetime.now()
        self.pretreat_time()
        time_used = datetime.now()-now
        print(f"时间补足 用时： {str(time_used)}")

        now = datetime.now()
        self.pretreat_normalizd()
        time_used = datetime.now()-now
        print(f"归一化 用时： {str(time_used)}")



    def calc_adj(self):
        std_deviation = 200
        std_deviation2 = std_deviation**2
        points_df = self.metas['points']
        point_count = points_df.shape[0]
        wam = np.full((point_count, point_count), np.nan)
        for i in range(point_count):
            for j in range(point_count):
                if i== j :
                    w = 1
                else:
                    iPoint = points_df.iloc[i]
                    jPoint = points_df.iloc[j]
                    dist = geodesic((iPoint['lat'], iPoint['lon']), (jPoint['lat'], jPoint['lon'])).km
                    w = math.exp(-dist**2/(std_deviation2))
                wam[i,j]= w
        adj_path = os.path.join(self.data_dir, f'{self.source_type.name }_points_adj.npy')
        np.save(adj_path, wam)
                    
                    
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-ro", "--data_dir", required=True, help="root")
    ap.add_argument("-ed", "--epa_dir", required=True, help="root")
    ap.add_argument("-sta", "--start", required=False, help="config")
    ap.add_argument("-end", "--end", required=False, help="config")
    ap.add_argument("-src", "--source_type", required=True, type=str,default=1, help="数据源'")
    ap.add_argument("-pd", "--production_dir", default='/datapool/shared/production', required=False, help="GFS Data dir")
    args, unknowns = ap.parse_known_args()
    arguments = vars(args)

    data_dir = arguments['data_dir']  
    logger = _setup_logger(os.path.join( data_dir,'log') )
    begin_date= date.fromisoformat( arguments['start'])
    end_date = date.fromisoformat( arguments['end'])
    pt = pretreat(data_dir, SourceType[ arguments["source_type"]],False)
    pt.pretreat_all()


