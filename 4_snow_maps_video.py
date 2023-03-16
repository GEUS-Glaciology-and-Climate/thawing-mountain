# -*- coding: utf-8 -*-
"""
@author: bav@geus.dk

tip list:
    %matplotlib inline
    %matplotlib qt
    import pdb; pdb.set_trace()
"""
import xarray as xr
import rioxarray
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import geopandas as gpd
import cv2
import os

def make_video(image_folder, video_name):
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    # images.sorted()
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, 15, (width,height))
    
    for image in images:
        frame = cv2.imread(os.path.join(image_folder, image))
        video.write(frame)
    
    cv2.destroyAllWindows()
    video.release()
    
plt.close('all')

df_fa = gpd.read_file('data/GIS/FocusAreas.shp')


for area in df_fa.name[2:3]:
    print('===',area,'===')
    print('opening dataset')
    ds_sm = xr.open_mfdataset(['data/S2/snowmap_'+area+'_'+str(y)+'.nc' \
                               for y in range(2017, 2023) \
                               if 'snowmap_'+area+'_'+str(y)+'.nc' in os.listdir('data/S2')])
    ds_sm = ds_sm.where(ds_sm.elevation>0).drop('elevation')
    ds_sm = ds_sm.where((ds_sm.NDSI+ds_sm.B04)!=0)
    ds_sm['snow'] = ds_sm.snow.where(ds_sm.snow <= 100).where(ds_sm.snow >= 0)
    tmp = ds_sm.snow.copy()
    del ds_sm
    # post processing (filling gaps with bbfill, ffill and climatology)

    print('> filling with ffill and bfill')   
    # pixels for which there is at least one good value
    msk = tmp.notnull().any(dim='time').copy()
    
    for year in np.unique(tmp.time.dt.year):
        # filling the gaps in the first image of the year (assuming snow)
        time_first = tmp.sel(time=str(year)).time[0]
        tmp.loc[{'time': time_first}] = \
            tmp.sel(time=time_first).fillna(100).where(msk)
        # filling the gaps in the last image of the year (assuming snow)
        time_last = tmp.sel(time=str(year)).time[-1]
        tmp.loc[{'time': time_last}] = \
            tmp.sel(time=time_last).fillna(100).where(msk)

        tmp_y = tmp.sel(time=str(year)).copy()

        # then creating a replicate of the first image with time stamp yyyy-01-01
        init = tmp_y.isel(time=[0, 0]).copy()
        init['time']=[pd.to_datetime(str(year)+'-01-01'), time_first.values]
        init = init.resample(time='D').nearest()
        # then creating a replicate of the first image with time stamp yyyy-12-31
        finit = tmp_y.isel(time=[-1, -1]).copy()
        finit['time']=[time_last.values, pd.to_datetime(str(year)+'-12-31')]
        finit = finit.resample(time='D').nearest()
        
        print('Adding',init.time[0].values,finit.time[-1].values)
        tmp = xr.concat((init,finit,tmp),dim='time')
    del tmp_y, init, finit
    tmp = tmp.sortby('time')
    tmp = tmp.resample(time='D').asfreq()
    ds_sm_filled = tmp.ffill('time')
    ds_sm_b = tmp.bfill('time')
    ds_sm_filled = ds_sm_filled.where(ds_sm_filled==ds_sm_b)
    ds_sm_filled['doy'] = ds_sm_filled.time.dt.dayofyear
    print('> filling with climatology')
    ds_sm_clim = ds_sm_filled.groupby('doy').mean(skipna=True)
    ds_sm_clim_ext = ds_sm_clim.sel(doy = ds_sm_filled['doy'] )
    ds_sm_filled = ds_sm_filled.fillna(ds_sm_clim_ext)
    print('loading data')
    ds_sm_clim = ds_sm_clim.load()
    
    # %% Generating maps
    def plotting_all_maps(ds_sm_clim, folder):
        try:
            os.mkdir(folder)
        except:
            pass
        [os.remove(folder+f) for f in os.listdir(folder)]
        time_dim = list(ds_sm_clim.dims)
        time_dim.remove('x')
        time_dim.remove('y')
        time_dim=time_dim[0]
        for doy in ds_sm_clim[time_dim]:

            fig = plt.figure()
            ax = plt.gca()
            ax.set_facecolor("lightgray")
    
            if time_dim == 'time':
                ds_sm_clim.isel(time=doy).plot(vmin=0, vmax = 100, 
                                                      cmap='BuPu_r',
                                                      cbar_kwargs={'label': 'Snow probability'})                
                plt.title(ds_sm_clim.isel(time=doy).time.dt.strftime('%Y-%m-%d').values)
            elif time_dim == 'doy':
                ds_sm_clim.sel(doy=doy).plot(vmin=0, vmax = 100, 
                                                      cmap='BuPu_r',
                                                      cbar_kwargs={'label': 'Snow probability'})                
                plt.title('DOY: '+str(doy.values).zfill(3))
            fig.savefig(folder+str(doy.values).zfill(3)+'.png', dpi = 300)
            plt.close(fig)
            print(doy.values,' /', len(ds_sm_clim[time_dim]))
    print('plotting climatology')

    plotting_all_maps(ds_sm_clim, folder='plots/Climatology/'+area+'/')
    print('making video')
    
    make_video('plots/climatology videos/'+area+'_climatology.avi')

