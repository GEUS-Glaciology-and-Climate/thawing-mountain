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
plt.close('all')

ds_sm = xr.open_dataset('data/S2/1_Sisimiut/snowmap_2017-02-14_2022-11-01.nc')

# post processing (filling gaps with bbfill, ffill and climatology)
ds_sm = ds_sm.rio.write_crs("EPSG:3413")
ds_sm = ds_sm.rio.reproject("EPSG:4326", nodata=-999)
print('> filling with ffill and bfill')
tmp = ds_sm.where(ds_sm <= 100)
tmp = tmp.where(tmp >= 0)
tmp = tmp.resample(time='D').mean()
ds_sm_filled = tmp.ffill('time')
ds_sm_b = tmp.bfill('time')
ds_sm_filled = ds_sm_filled.where(ds_sm_filled==ds_sm_b)
ds_sm_filled['doy'] = ds_sm_filled.time.dt.dayofyear
print('> filling with climatology')
ds_sm_clim = ds_sm_filled.groupby('doy').mean(skipna=True).sel(doy = ds_sm_filled['doy'] )
ds_sm_filled = ds_sm_filled.fillna(ds_sm_clim)

# %% Generating maps
for doy in range(0, len(ds_sm_filled.time),1):
    fig = plt.figure()
    ax = plt.gca()
    ax.set_facecolor("lightgray")
    ds_sm_filled.isel(time=doy).snow.plot(vmin=0, vmax = 100, cmap='BuPu_r', cbar_kwargs={'label': 'Snow probability'})
    plt.title(ds_sm_filled.isel(time=doy).time.dt.strftime('%Y-%m-%d').values)
    
    fig.savefig('plots/Sisimiut/'+ds_sm_filled.isel(time=doy).time.dt.strftime('%Y-%m-%d').values+'.png')
    plt.close(fig)
    print(doy,' /', len(ds_sm_filled.time) )

# %% Making video
import cv2
import os
def make_video(image_folder, video_name):
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, 15, (width,height))
    
    for image in images:
        frame = cv2.imread(os.path.join(image_folder, image))
        video.write(frame)
    
    cv2.destroyAllWindows()
    video.release()
print('making video')
make_video('plots/Sisimiut/', 'Sisimiut.avi')
    # ds_sm_filled.sel(time=str(year)).snow.where(lambda x: x<50).idxmax(dim='time').doy.plot(vmin=0, vmax = 365)
    # plt.figure()
    # ds_sm_filled.sel(time=str(year)).snow.where(lambda x: x<50).idxmin(dim='time').doy.plot(vmin=0, vmax = 365)
# %% 
ds_sm_filled.loc[ds_sm_filled.snow>50, ['year', 'snow']].groupby('year').first()

df = pd.read_csv('data/coordinates_all.csv', sep=';').set_index('ID')
# plt.close('all')
# notes = pd.read_csv('data/time lapse/notes.txt', skipinitialspace=True)
# notes['time'] = pd.to_datetime(notes.time)
# notes = notes.set_index(['site', 'time']).sort_index()

df_gst = pd.read_csv('data/GST_data.csv', skipinitialspace=True)
df_gst['time'] = pd.to_datetime(df_gst.time)
df_gst['site'] = df_gst.site.astype(str)
df_gst = df_gst.set_index(['site', 'time']).sort_index()
#%%
plt.close('all')

for site in df.index:
    # site = '6436FF'
    print(site)
    site_flag = ds_sm_filled.sel(x=df.loc[site].longitude, 
                          y=df.loc[site].latitude, 
                          method='nearest').reset_coords(['x', 'y'], 
                                                         drop=True).to_dataframe()['snow']
    site_flag_clim = ds_sm_clim.sel(x=df.loc[site].longitude, 
                          y=df.loc[site].latitude, 
                          method='nearest').reset_coords(['x', 'y'], 
                                                         drop=True).to_dataframe()['snow']
    if site_flag.isnull().all():
        print('> outside of snow map')
        continue

    if site in df_gst.index.get_level_values(0):
        fig,ax = plt.subplots(1,1, figsize=(7,3.5))
        plt.subplots_adjust(top=0.6)
        
    try:
        df_gst.loc[site,'GST'].plot(ax=ax, label='_nolegend_')
    except:
        print('> not in df_gst')
        continue

    cmap = cm.get_cmap('coolwarm_r')
    
    for i in range(len(site_flag)-1):
        ax.axvspan(site_flag.index[i].to_pydatetime(),
               site_flag.index[i+1].to_pydatetime(),
               alpha=0.5,color=cmap(site_flag.iloc[i]/100),
               lw=0)
    
    # plt.legend(title=site, ncol=3, loc='lower center', bbox_to_anchor=(0.5,1.05))
    plt.ylabel('GST ($^o$C)')
    
    color = 'k'
    ax2 = plt.gca().twinx()  # instantiate a second axes that shares the same x-axis
    ax2.plot(site_flag.index, site_flag, color='k')
    ax2.plot(site_flag.index, site_flag_clim, color='gray')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title(site)
    fig.savefig('plots/'+site+'_GST_labeled.png',bbox_inches='tight', dpi=300)
    