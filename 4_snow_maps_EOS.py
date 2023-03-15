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

    
plt.close('all')

df_fa = gpd.read_file('data/GIS/FocusAreas.shp')
df = pd.read_csv('data/coordinates_all.csv', sep=';').set_index('ID')
# plt.close('all')
# notes = pd.read_csv('data/time lapse/notes.txt', skipinitialspace=True)
# notes['time'] = pd.to_datetime(notes.time)
# notes = notes.set_index(['site', 'time']).sort_index()

# df_gst = pd.read_csv('data/GST_data.csv', skipinitialspace=True)
# df_gst['time'] = pd.to_datetime(df_gst.time)
# df_gst['site'] = df_gst.site.astype(str)
# df_gst = df_gst.set_index(['site', 'time']).sort_index()

for area in df_fa.name[2:3]:
    # # %% 
    # area = 'DiskoNE'
    print('===',area,'===')
    print('opening dataset')
    ds_sm = xr.open_mfdataset(['data/S2/snowmap_'+area+'_'+str(y)+'.nc' \
                               for y in range(2016, 2023) \
                               if 'snowmap_'+area+'_'+str(y)+'.nc' in os.listdir('data/S2')])
    ds_sm = ds_sm.where(ds_sm.elevation>0).drop('elevation')
    ds_sm = ds_sm.where((ds_sm.NDSI+ds_sm.B04)!=0)
    ds_sm['snow'] = ds_sm.snow.where(ds_sm.snow <= 100).where(ds_sm.snow >= 0)
    tmp = ds_sm.snow.copy()
    # post processing (filling gaps with bbfill, ffill and climatology)

    print('> filling with ffill and bfill')   
    # pixels for which there is at least one good value
    msk = tmp.notnull().any(dim='time').copy()
    
    for year in np.unique(tmp.time.dt.year):
        # filling the gaps in the first image of the year (assuming snow)
        time_first = tmp.sel(time=str(year)).time[0]
        tmp.loc[{'time': time_first}] = \
            tmp.sel(time=time_first).fillna(100).where(msk).values
        # filling the gaps in the last image of the year (assuming snow)
        time_last = tmp.sel(time=str(year)).time[-1]
        tmp.loc[{'time': time_last}] = \
            tmp.sel(time=time_last).fillna(100).where(msk).values

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

    # ds_sm_filled.to_netcdf('out/snowmap_'+area+'_'+'_filled.nc')
    # ds_sm_clim.to_netcdf('out/snowmap_'+area+'_'+'_clim.nc')
    # # %% Plotting the gap-filled and climatalogical snow presence at GST measurement sites   
    # for site in df.index:
    #     # site = '6436FF'
    #     site_flag = ds_sm_filled.sel(x=df.loc[site].longitude, 
    #                           y=df.loc[site].latitude, 
    #                           method='nearest').reset_coords(['x', 'y'], 
    #                                                          drop=True).to_dataframe()['snow']
    #     site_flag_clim = ds_sm_clim_ext.sel(x=df.loc[site].longitude, 
    #                           y=df.loc[site].latitude, 
    #                           method='nearest').reset_coords(['x', 'y'], 
    #                                                          drop=True).to_dataframe()['snow']
    #     if site_flag.isnull().all():
    #         continue
    
    #     if site in df_gst.index.get_level_values(0):
    #         fig,ax = plt.subplots(1,1, figsize=(7,3.5))
    #         plt.subplots_adjust(top=0.6)
            
    #     try:
    #         df_gst.loc[site,'GST'].plot(ax=ax, label='_nolegend_')
    #     except:
    #         continue
    #     # print(site)
    
    #     cmap = cm.get_cmap('coolwarm_r')
        
    #     for i in range(len(site_flag)-1):
    #         ax.axvspan(site_flag.index[i].to_pydatetime(),
    #                site_flag.index[i+1].to_pydatetime(),
    #                alpha=0.5,color=cmap(site_flag.iloc[i]/100),
    #                lw=0)
        
    #     # plt.legend(title=site, ncol=3, loc='lower center', bbox_to_anchor=(0.5,1.05))
    #     plt.ylabel('GST ($^o$C)')
        
    #     color = 'k'
    #     ax2 = plt.gca().twinx()  # instantiate a second axes that shares the same x-axis
    #     ax2.plot(site_flag.index, site_flag, color='k')
    #     ax2.plot(site_flag.index, site_flag_clim, color='gray')
    #     ax2.tick_params(axis='y', labelcolor=color)
        
    #     plt.title(site)
    #     fig.savefig('plots/'+site+'_GST_labeled.png',bbox_inches='tight', dpi=300)

    # %% 
    print('Snow onset and end day from climatology')
    thr= 65
    bare_area = xr.where(ds_sm_clim>thr, 0,1)
    bare_area =bare_area.where(ds_sm_clim.notnull())    
    SED = bare_area.idxmax(dim='doy')
    SED = SED.where(ds_sm_clim.notnull().any(dim='doy'))
    SED = SED.rename('SED')
    bare_area2 =bare_area.copy()
    bare_area2['doy'] = 365-bare_area2.doy
    bare_area2 = bare_area2.sortby('doy')
    SOD = 365 - bare_area2.idxmax(dim='doy')
    SOD = SOD.where(ds_sm_clim.notnull().any(dim='doy'))
    SOD = SOD.rename('SOD')
    plt.close('all')
    #%%
    x = int(len(bare_area.x)*0.345)
    y = int(len(bare_area.y)*0.2)
    plt.figure()
    ds_sm_clim.isel(x=x,y=y).plot(marker='o')
    plt.plot(SED.isel(x=x,y=y),1,marker='o',markersize=10)
    plt.plot(SOD.isel(x=x,y=y),1,marker='o',markersize=10)
    
    import matplotlib.colors as cm
    c = cm.Colormap('Spectral')
    c.set_under('magenta')
    
    plt.figure()
    h = SED.plot(vmin=1)
    plt.plot(bare_area.isel(x=x,y=y).x,bare_area.isel(x=x,y=y).y, 'r',marker='o')
    cmap = h.get_cmap()
    cmap.set_under('red')
    h.set_cmap(cmap)
    plt.title('Melt out day')
    #%%
    SED.rio.to_raster('out/'+area+'_SED.tif')

    plt.figure()
    SOD.plot()
    SOD.rio.to_raster('out/'+area+'_SOD.tif')

