from datetime import datetime
from datetime import timedelta
import argparse
import socket
import numpy as np 
import pandas as pd
import os
import logging
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
# import args
from data_file import DataFile, DataStep,SourceType,ChannelSplit
import cartopy.crs as ccrs
import cartopy
from tqdm import tqdm

plt.rcParams['font.family'] = 'sans-serif' 
plt.rcParams["font.sans-serif"]=["方正黑体"] #设置中文字体
plt.rcParams["axes.titlesize"]=10 #设置标题字体
plt.rcParams["axes.unicode_minus"]=False #该语句解决图像中的“-”负号的乱码问题
# plt.rcParams["figure.figsize"]=(8,4.5) #设置图大小
plt.rcParams["figure.dpi"]=400 #设置图大小
plt.rcParams["xtick.labelsize"]=6
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
# plt.rc('axes', labelsize=8)





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
    fhlr = logging.FileHandler(os.path.join( log_dir,f'ploter_{hostname}.log')) # 输出到文件的handler
    fhlr.setFormatter(formatter)
    fhlr.setLevel('DEBUG')
    logger.addHandler(chlr)
    logger.addHandler(fhlr)
    return logger 



class Ploter():
    def __init__(self, data_dir, data_step, source_type):
        self.data_dir = data_dir
        self.data_step = data_step
        self.source_type = source_type
        self.data_file = DataFile(data_dir)
        metas = self.data_file.read_meta(source_type)
        self.metas = metas
        self.fig = plt.figure()
        self.logger = logging.getLogger()
        plot_dir = os.path.join(data_dir,'plot')
        if not os.path.exists(plot_dir):
            os.mkdir(plot_dir)

   

    # 点位分布
    def plot_points(self):
        plt.rcParams["figure.figsize"]=(8,8) #设置图大小
        s = self.metas['points']['cid']
        showChannel = 1
        fig = self.fig
        fig.clf()
        axPosition = [0.1,0.1,0.8,0.8]
        ax = fig.add_subplot(1,1,1, projection=ccrs.PlateCarree())
        # fig.subplots_adjust(axPosition[0],axPosition[1],axPosition[2],axPosition[3])
        ax.coastlines()
        # zorder can be used to arrange what is on top
        ax.add_feature(cartopy.feature.STATES, zorder=4)   # land is specified to plot above ...
        ax.add_feature(cartopy.feature.OCEAN, zorder=1)  # ... the ocean
        ax.set_title(f'{self.source_type.name}  points')
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")
        points_df = self.metas['points']
        for rowIndex,row in self.metas['channels'].iterrows():
            sub_points_df = points_df[points_df['cid'].str.contains(ChannelSplit+str(row.name)+ChannelSplit)]
            sub_points_df.set_index('vid' ,inplace=True)

            lon = sub_points_df['lon']
            lat = sub_points_df['lat']
            ax.scatter(lon, lat,label=f"{row['symbol']}({sub_points_df.shape[0]})", zorder=3)
        ax.legend()
        dot_file =os.path.join( self.data_dir ,'plot', f'{self.source_type.name}_points.png')
        fig.savefig(dot_file, bbox_inches='tight')




    # 数据缺失
    def plot_missing(self, xrange = 'day'):
        fig = plt.figure(figsize=(10,4))
        # pointsDf = self.metas['points']
        channels_df = self.metas['channels']
        for row_index,row in channels_df.iterrows():
            # subPointsDf = pointsDf[pointsDf['cid'].str.contains(ChannelSplit+str(row.name)+ChannelSplit)]
            # subPointsDf.set_index('vid' ,inplace=True)
            channel_index = channels_df.index.get_loc(row_index)
            #T*V*C
            dot_file =os.path.join( self.data_dir ,'plot', f'{self.source_type.name}_missing_{self.data_step.name}_c{channel_index}.png')
            datas = self.data_file.read_data(channel_index,self.source_type, self.data_step)
            coors = np.isnan(datas)
            datas.fill(1)
            datas[coors]=0
            datas = np.sum(datas,axis=2)    #压缩了channel，变成T*V

            # 找空数据小时数
            hour_sum = np.sum(datas,axis=1)  #压缩了点位，变成另一个一维数组， T
            coors = np.nonzero(hour_sum)
            zero_count = hour_sum.shape[0]- coors[0].shape[0]
            #继续压缩
            time_start = self.metas['times'].iloc[0]['startTime']
            hours = datas.shape[0]
            if xrange == 'day':
                days = hours//24
                x = [time_start+timedelta(days=d) for d in range(days)]
                if hours>days*24:
                    datas = np.delete(datas,range(days*24,hours),axis=0)
                datas = np.reshape(datas,[days,24,datas.shape[1]])  #把同一天的24小时竖起来，变成T*24*V
                datas = np.sum(datas,axis=1)    #压缩了24小时到一天，变回成T*V
            else:
                x = [time_start+timedelta(hours=h) for h in range(hours)]
            

            # trainData=standardization(np.abs(trainData))*255
            # 直接从数据生成图片，虽然快，但不好看已弃用
            # datas = datas.astype(np.uint8)
            # image = Image.fromarray(datas.swapaxes(0,1), mode="L")
            # image.save(dotFile) #成品图保存

            y=np.arange(0, datas.shape[1], 1, dtype=int)
            
            # plt.gcf().autofmt_xdate(bottom=0.2, rotation=30)
            plt.xticks(rotation = 30)
            fig.clf()
            ax=fig.add_axes([0.1,0.1,0.8,0.8])
            ax.set_title(f"{self.source_type.name} {self.data_step.name} {row['symbol']} data missing")
            ax.set_xlabel("time")
            ax.set_ylabel("point")
            levels = np.arange(0,24*1+1,1)
            cmap = plt.get_cmap('viridis')
            # cmap.set_bad('black')
            norm = matplotlib.colors.BoundaryNorm(levels, ncolors=cmap.N, clip=True)
            im=ax.pcolormesh( x,y,datas.swapaxes(0,1), cmap=cmap, norm=norm)
            fig.colorbar(im, ax=ax)
            # fig.show()
            fig.savefig(dot_file, bbox_inches='tight')
            # 显示全0的天
            datas = np.sum(datas,axis=1)
            zero_days = np.where(datas==0)
            str_out = f"无数据小时数：{zero_count}\t{self.source_type.name}_{self.data_step.name}_c{channel_index}_{row['symbol']} 无数据日期："
            if xrange == 'day':
                for day in zero_days[0]:
                    hour = day*24
                    str_out += self.metas['times'].iloc[hour][2][0:10] +"\t"
                self.logger.info(str_out)

    # 数据异常
    def plot_error(self, xrange = 'day'):
        fig = plt.figure(figsize=(16,5))
        channels_df = self.metas['channels']
        normal_string = ''
        time_start = self.metas['times'].iloc[0]['startTime']
        # timeStart = datetime.strptime(timeStart, "%Y-%m-%d %H:%M:%S")
        for row_index,row in channels_df.iterrows():
            channel_index = channels_df.index.get_loc(row_index)
            #T*V*C
            dot_file =os.path.join( self.data_dir ,'plot', f'{self.source_type.name}_error_{self.data_step.name}_c{channel_index}.png')
            datas = self.data_file.read_data(channel_index,self.source_type, self.data_step)
            max_value = np.nanmax( datas)
            avg_value = np.nanmean( datas)
            hours = datas.shape[0]
            if xrange == 'day':
                days = hours//24
                x = [time_start+timedelta(days=d) for d in range(days)]
                if hours>days*24:
                    datas = np.delete(datas,range(days*24,hours),axis=0)
                datas = np.reshape(datas,[days,24,datas.shape[1]])
                datas = np.nanmean(datas,axis=1)
            else:
                x = [time_start+timedelta(hours=h) for h in range(hours)]
                datas = datas[:,:,0]
                
            y=np.arange(0, datas.shape[1], 1, dtype=int)


            fig.clf()
            ax=fig.add_axes([0.1,0.1,0.8,0.8],facecolor='black')

            # plt.gcf().autofmt_xdate(bottom=0.2, rotation=30)
            ax.set_title(f"{self.source_type.name} {self.data_step.name} c{channel_index} {row['symbol']} data error\n\
                avg={avg_value:.2f} max={max_value:.2f}")
            ax.set_xlabel("time")
            ax.set_ylabel("point" )
            cmap = plt.get_cmap('viridis')
            # cmap.set_bad('black')
            im=ax.pcolor(x,y,datas.swapaxes(0,1))
            fig.colorbar(im, ax=ax, cmap=cmap)
            plt.xticks(rotation = 30)
            fig.savefig(dot_file, bbox_inches='tight')
                # plt.show()
        self.logger.info(normal_string)
    
    
    # 数据异常
    def plot_newest(self, start_time, end_time, action_time = None, xrange = 'day'):
        fig = plt.figure(figsize=(10,4))
        channels_df = self.metas['channels']
        normal_string = ''
        newest_time = pd.read_csv(os.path.join( self.data_dir ,'newestTime.csv'), index_col=0, parse_dates=[0,1], infer_datetime_format=True)
        start_time_index = newest_time.index.get_loc(start_time)
        end_time_index = newest_time.index.get_loc(end_time)
        for row_index,row in channels_df.iterrows():
            channel_index = channels_df.index.get_loc(row_index)
            #T*V*C
            dot_file =os.path.join( self.data_dir ,'plot', f'newest_error_c{channel_index}.png')
            datas = np.load(os.path.join( self.data_dir ,f'newestData_c{channel_index}.npy'))[start_time_index:end_time_index, :, :]
            max_value = np.nanmax( datas)
            avg_value = np.nanmean( datas)
            hours = datas.shape[0]
            if xrange == 'day':
                days = hours//24
                x = [start_time+timedelta(days=d) for d in range(days)]
                if hours>days*24:
                    datas = np.delete(datas,range(days*24,hours),axis=0)
                datas = np.reshape(datas,[days,24,datas.shape[1]])
                datas = np.nanmean(datas,axis=1)
            else:
                x = [start_time+timedelta(hours=h) for h in range(hours)]
                # x = range(startTimeIndex, endTimeIndex)
                datas = datas[:,:,0]
                
            y=np.arange(0, datas.shape[1], 1, dtype=int)


            fig.clf()
            ax=fig.add_axes([0.1,0.1,0.8,0.8],facecolor='black')

            # plt.gcf().autofmt_xdate(bottom=0.2, rotation=30)
            ax.set_title(f"{self.source_type.name} {self.data_step.name} c{channel_index} {row['symbol']} data error\n\
                avg={avg_value:.2f} max={max_value:.2f} @{action_time}")
            ax.set_xlabel("time")
            ax.set_ylabel("point" )
            cmap = plt.get_cmap('viridis')
            # cmap.set_bad('black')
            im=ax.pcolor(x,y,datas.swapaxes(0,1))
            fig.colorbar(im, ax=ax, cmap=cmap)
            plt.xticks(rotation = 30)
            fig.savefig(dot_file, bbox_inches='tight')
                # plt.show()
        self.logger.info(normal_string)
        
            
    # 数据区间
    def find_min_max(self):
        fig = plt.figure(figsize=(10,4))
        channels_df = self.metas['channels']
        self.logger.info(self.source_type.name )
        normal_string = ''
        for row_index,row in channels_df.iterrows():
            channel_index = channels_df.index.get_loc(row_index)
            #T*V*C
            datas = self.data_file.read_data(channel_index,self.source_type, self.data_step)
            # hours = datas.shape[0]
            # days = hours//24
            # daysAdd=range(0, days, 1)
            # timeStart = self.metas['times'].iloc[0]['startTime']
            # timeStart = datetime.strptime(timeStart, "%Y-%m-%d %H:%M:%S")
            # x = [timeStart+timedelta(days=d) for d in daysAdd]
            # y=np.arange(0, datas.shape[1], 1, dtype=int)
            
            hour_max = np.nanmax(datas)
            hour_min = np.nanmin(datas)
            hour_avg = np.nanmean(datas)
            # 显示均值
            self.logger.info( f"{row['name']} avg:{hour_avg:.2f}  max:{hour_max:.2f}  min:{hour_min:.2f}, 建议归一化max,min： {hour_max*2-np.nanmean(datas):.2f}, {hour_min*2-hour_avg:.2f}")
            normal_string += f"{hour_max*2-np.nanmean(datas):.2f}\t{hour_min*2-hour_avg:.2f}\n"
            

        self.logger.info(normal_string)
        
    def plot_all(self, xrange = 'day'):
        if not self.source_type in ( SourceType.gfs, SourceType.gfs_ca, SourceType.gfs_gd):
            # 气象数据不需要管点位和缺失，只需要查最大最小值
            if self.data_step == DataStep.src:
                self.plot_points()
            self.plot_missing()
            pass

        self.plot_error(xrange)
        
    def plot_marker(self):
        # fig = plt.figure(figsize=(10,8))
        # pointsDf = self.metas['points']
        channels_df = self.metas['channels']
        data_step = DataStep.pre_audited
        for rowIndex,row in tqdm( channels_df.iterrows(), total=channels_df.shape[0]):
            fig, axs = plt.subplots(4, 1, sharey=False, figsize=(20,10))            
            ax_valid = axs[0]
            ax_marker = axs[1]
            ax_has_data = axs[2]
            ax_dst_has_data = axs[3]

            channelIndex = channels_df.index.get_loc(rowIndex)
            #T*V*C   C: valid, wsh, hasData, markers
            dot_file =os.path.join( self.data_dir ,'plot', f'{self.source_type.name}_marker_c{channelIndex}_{self.data_step.value:02}{self.data_step.name}.png')
            datas = self.data_file.read_data(channelIndex,self.source_type, data_step)
            hours = datas.shape[0]
            hours_add=range(0, hours, 1)
            time_start = self.metas['times'].iloc[0]['startTime']
            time_start = datetime.strptime(time_start, "%Y-%m-%d %H:%M:%S")
            x = [time_start+timedelta(hours=d) for d in hours_add]
            y=np.arange(0, datas.shape[1], 1, dtype=int)
            
            
            valid_datas = datas[:,:,0]
            coors = np.isnan(valid_datas)
            valid_datas[coors]=0
            title = f"{self.source_type.name} {row['symbol']} predict to remove?"
            self.draw_one_part(ax_valid,fig,title,x,y,valid_datas)
            
            
            hasDatas = datas[:,:,2]
            title = f"{self.source_type.name} {row['symbol']} data pre audit"
            self.draw_one_part(ax_has_data,fig,title,x,y,hasDatas)
            
            
            marker_datas = datas[:,:,3:]
            coors = np.isnan(marker_datas)
            marker_datas[coors]=0
            marker_datas = np.sum(marker_datas,axis=2)    #压缩了channel，变成T*V
            title = f"{self.source_type.name} {row['symbol']} marker"
            self.draw_one_part(ax_marker,fig,title,x,y,marker_datas)
            
            dst_datas = self.data_file.read_data(channelIndex,self.source_type, DataStep.pos_audited)[:,:,0]
            coors = np.isnan(dst_datas)
            dst_datas.fill(1)
            dst_datas[coors]=0
            title = f"{self.source_type.name} {row['symbol']} data pre audit"
            self.draw_one_part(ax_dst_has_data,fig,title,x,y,dst_datas)

            
            fig.savefig(dot_file, bbox_inches='tight')

    def draw_one_part(self, ax,fig ,title, x, y, z):
        ax.set_title(title)
        # ax.set_xlabel("时间")
        ax.set_ylabel("point")
        # levels = np.arange(0,24*1+1,1)
        cmap = plt.get_cmap('viridis')
        # cmap.set_bad('black')
        # norm = matplotlib.colors.BoundaryNorm(levels, ncolors=cmap.N, clip=True)
        im=ax.pcolormesh( x,y,z.swapaxes(0,1), cmap=cmap, )
        fig.colorbar(im, ax=ax)            
        
    def plot_points_adj(self):
        adjPath = os.path.join(self.data_dir, f'{self.source_type.name }_points_adj.npy')
        datas = np.load(adjPath)
        fig = plt.figure(figsize=(10,8))
        y=np.arange(0, datas.shape[0], 1, dtype=int)


        fig.clf()
        ax=fig.add_axes([0.1,0.1,0.8,0.8],facecolor='black')

        # plt.gcf().autofmt_xdate(bottom=0.2, rotation=30)
        ax.set_title(f"{self.source_type.name} points adj")
        ax.set_xlabel("point")
        ax.set_ylabel("point" )
        cmap = plt.get_cmap('viridis')
        # cmap.set_bad('black')
        im=ax.pcolor(y,y,datas.swapaxes(0,1))
        fig.colorbar(im, ax=ax, cmap=cmap)
        plt.xticks(rotation = 30)
        dot_file = os.path.join(self.data_dir,'plot', f'{self.source_type.name }_points_adj.png')
        fig.savefig(dot_file, bbox_inches='tight')
            
            
            
            
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-ro", "--data_dir", required=True, help="root")
    ap.add_argument("-ed", "--epa_dir", required=True, help="root")
    ap.add_argument("-sta", "--start", required=False, help="config")
    ap.add_argument("-end", "--end", required=False, help="config")
    ap.add_argument("-src", "--source_type", required=True, type=str,default=1, help="数据源'")
    ap.add_argument("-pd", "--production_dir", default='/datapool01/shared/production', required=False, help="GFS Data dir")
    args, unknowns = ap.parse_known_args()
    arguments = vars(args)

    data_dir = arguments['data_dir']  
    logger = _setup_logger(os.path.join( data_dir,'log') )
    plot = Ploter(data_dir, DataStep.src, SourceType[ arguments["source_type"]])
    # plot.plot_points_adj()
    plot.plot_all('day')
    plot = Ploter(data_dir, DataStep.normalized, SourceType[ arguments["source_type"]])
    plot.plot_all('day')

