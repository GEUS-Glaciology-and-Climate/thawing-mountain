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
df = pd.read_csv('data/coordinates_all.csv', sep=';').set_index('ID')
# plt.close('all')
# notes = pd.read_csv('data/time lapse/notes.txt', skipinitialspace=True)
# notes['time'] = pd.to_datetime(notes.time)
# notes = notes.set_index(['site', 'time']).sort_index()

df_gst = pd.read_csv('data/GST_data.csv', skipinitialspace=True)
df_gst['time'] = pd.to_datetime(df_gst.time)
df_gst['site'] = df_gst.site.astype(str)
df_gst = df_gst.set_index(['site', 'time']).sort_index()

for area in df_fa.name[-1:]:
    
    print('===',area,'===')
    # #%%
    # area = 'Vaigat'
    # y = 2021
    print('opening dataset')
    ds_sm = xr.open_mfdataset(['data/S2/snowmap_'+area+'_'+str(y)+'.nc' \
                               for y in range(2016, 2023) \
                               if 'snowmap_'+area+'_'+str(y)+'.nc' in os.listdir('data/S2')]
                              ).snow
        
    # post processing (filling gaps with bbfill, ffill and climatology)
    # ds_sm = ds_sm.rio.write_crs("EPSG:3413")
    # ds_sm = ds_sm.rio.reproject("EPSG:4326", nodata=-999)
    print('> filling with ffill and bfill')
    tmp = ds_sm.where(ds_sm <= 100)
    tmp = tmp.where(tmp >= 0)
    tmp = tmp.resample(time='D').mean()
    ds_sm_filled = tmp.ffill('time')
    ds_sm_b = tmp.bfill('time')
    ds_sm_filled = ds_sm_filled.where(ds_sm_filled==ds_sm_b)
    ds_sm_filled['doy'] = ds_sm_filled.time.dt.dayofyear
    print('> filling with climatology')
    ds_sm_clim = ds_sm_filled.groupby('doy').mean(skipna=True)
    ds_sm_clim_ext = ds_sm_clim.sel(doy = ds_sm_filled['doy'] )
    ds_sm_filled = ds_sm_filled.fillna(ds_sm_clim_ext)
    
    print('loading climatology')
    ds_sm_clim = ds_sm_clim.load()
    
    # %% Generating maps
    def plotting_all_maps(ds_sm_clim, folder):
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
    try:
        os.mkdir('plots/'+area+'/')
        os.mkdir('out/videos/')
    except:
        pass
    plotting_all_maps(ds_sm_clim, folder='plots/'+area+'/')
    print('making video')
    make_video('plots/'+area+'/', 'out/videos/'+area+'_climatology.avi')
    
    # print('plotting gap-filled maps')
    # plotting_all_maps(ds_sm_filled, folder='plots/'+area+'/')
    # print('making video')
    # make_video('plots/'+area+'/', area+'_gap-filled.avi')
    
