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
plt.close('all')

df_fa = gpd.read_file('data/GIS/FocusAreas.shp')[-1:]
for area in df_fa.name:
    print('===',area,'===')
    ds_sm_filled = xr.open_mfdataset(['out/S2 snow maps/snowmap_'+area+'_'+str(y)+'_fbfill.nc' \
                               for y in range(2017, 2023) \
                               if 'snowmap_'+area+'_'+str(y)+'_fbfill.nc' in os.listdir('out/S2 snow maps')])
        
    print('> filling with climatology')
    ds_sm_filled['doy'] = ds_sm_filled.time.dt.dayofyear
    ds_sm_clim = ds_sm_filled.groupby('doy').mean(skipna=True)
    del ds_sm_filled
    # ds_sm_clim_ext = ds_sm_clim.sel(doy = ds_sm_filled['doy'] )
    # ds_sm_filled = ds_sm_filled.fillna(ds_sm_clim_ext)
    # ds_sm_filled['doy'] = ds_sm_filled.time.dt.dayofyear
    print('Computing climatology')
    ds_sm_clim = ds_sm_clim.snow.load()
    print('saving climatology to file')
    ds_sm_clim.to_netcdf('out/snowmap_'+area+'_'+'_clim.nc',
                         encoding={'snow': dict(zlib=True, complevel=5)})
    
    # Generating maps
    print('plotting climatology')
    plotting_all_maps(ds_sm_clim, folder='plots/Climatology/'+area+'/')
    
    print('making video')
    make_video('plots/Climatology/'+area+'/', 'plots/climatology videos/'+area+'_climatology.avi')

