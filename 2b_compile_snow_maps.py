# -*- coding: utf-8 -*-
"""
@author: bav@geus.dk

tip list:
    %matplotlib inline
    %matplotlib qt
    import pdb; pdb.set_trace()
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# The following is not a package. It is a file utils.py which should be in the same folder as this notebook.
from osgeo import gdal
gdal.PushErrorHandler('CPLQuietErrorHandler')
import xarray as xr
import rioxarray
import rasterio
import pandas as pd
import geopandas as gpd
import matplotlib.cm as cm
from matplotlib.colors import from_levels_and_colors                  
from  rasterio.enums import Resampling
import warnings
from sentinelhub import (
    CRS,
    BBox,
    bbox_to_dimensions,
    Geometry,
)
plotting = 0
# Conclusion:
# SCL	Defective. Mark shaddows as saturated or water
# SNW misses the snowy parts in shaddowed areas.
# CLD	Cloud probability,bare rock as cloud
# CLP does not work. Flags the entire area as snow covered.
# CLM works ok. It is a binary map.
# NDSI works fine in shaddowed areas.
CLM_th = 0.5  

# Let-it-snow approach
r_f = 12
r_d = 0.3
n1 = 0.4
n2 = 0.150
r1 = 0.2
r2 = 0.40
dz = 100
fs = 0.1
fct = 0.1
ft = 0.001
rb = 0.1

# snowflag
isnow_label = {0: 'no snow',
               100: 'snow',
               205: 'cloud',
               254: 'no data'}
col_name = {1: 'NDSI',
            2: 'B04',
            4: 'CLM',
            5: 'elevation'}
resolution = 20

df_fa = gpd.read_file('data/GIS/FocusAreas.shp')
df_fa.index = df_fa.index + 1
gdal.UseExceptions()
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')   
    
    
    for k in df_fa.index[-1:]:
        dir_name= 'data/S2/'+str(k) + '_' + df_fa.loc[k,'name']
        try:
            os.mkdir(dir_name)
        except:
            pass
        print(' ')
        print(dir_name)
        
        geometry = Geometry(geometry={"type":"Polygon",
                                      "coordinates":
                                          [[[np.round(x,4), np.round(y,4)] \
                                            for (x,y) in df_fa.loc[k,'geometry'].exterior.coords]]},
                            crs=CRS.WGS84)
        betsiboka_coords_wgs84 = [np.round(x,4) for x in df_fa.loc[k,'geometry'].bounds]
        betsiboka_bbox = BBox(bbox=betsiboka_coords_wgs84, crs=CRS.WGS84)
        betsiboka_size = bbox_to_dimensions(betsiboka_bbox, resolution=resolution)
            
        for year in range(2016,2023):
            print(year)
# %% 

            list_files = os.listdir(dir_name)
            list_files = [f for f in list_files if f.endswith('.tif')]
            list_files = [f for f in list_files if f[15:19] == str(year)]
            list_files = [dir_name+'/'+f for f in list_files]
            list_bbox = ['_'.join(f.split('_')[-6:]) for f in list_files]
            if len(np.unique(np.array(list_bbox)))>1:
                print(np.unique(np.array(list_bbox)))
                print(wtf)
            if len(list_files)==0:
                print('no data for', df_fa.loc[k,'name'],year)
                continue
            print('loading data')
            snow_map = xr.open_mfdataset(list_files, engine="rasterio", 
                                     combine='nested',
                                     concat_dim='time')
            print('preparing data')
            snow_map['time'] = [pd.to_datetime(f.split('_')[3]) for f in list_files]
            snow_map=snow_map.band_data.to_dataset('band').rename(col_name)
            snow_map['elevation'] = snow_map.elevation - 12000
            snow_map['NDSI'] = snow_map['NDSI'].where(snow_map.elevation.isel(time=0)>0)
            snow_map['B04'] = snow_map['B04'].where(snow_map.elevation.isel(time=0)>0)
            snow_map['elevation'] = snow_map['elevation'].where(snow_map.elevation.isel(time=0)>0)

            snow_map = snow_map.drop_duplicates('time')
            mask = snow_map['NDSI'].notnull().any(('x','y'))
            good_times = mask.time.loc[mask]
            if good_times.size == 0:
                print('No usable data for', df_fa.loc[k,'name'], year)
                continue
            snow_map = snow_map.loc[{'time': good_times.time}]

            snow_map['snow'] = xr.zeros_like(snow_map.B04)
            snow_map['snow'] = xr.where(snow_map.B04.isnull(), 254, snow_map['snow'])       
            
            print('first pass')
            msk = (snow_map.NDSI > n1) & (snow_map.B04 > r1)
            snow_map['snow'] = xr.where(msk, 100, snow_map.snow)
            
            zs = snow_map.elevation.copy()*np.nan

            for time in snow_map.time:
                snow_map_d = snow_map.sel(time=time).copy()
                # elevation of snowy pixels
                elev_snow = snow_map_d.elevation.where(snow_map_d.snow==100)
                zs.loc[{'time': time}] = elev_snow.quantile(0.01, skipna=True)
                
            print('second pass')
            msk = ((snow_map.NDSI > n2) & (snow_map.B04 > r2)) & (snow_map.elevation > zs)
            time_remove = (snow_map.snow==100).sum(('x','y')) / snow_map.elevation.notnull().sum(('x','y')) <= ft
            msk.loc[{'time': time_remove}] = False
            snow_map['snow'] = xr.where(msk, 100, snow_map.snow)
            
            print('third pass')
            msk = (snow_map.NDSI > 0.8) & (snow_map.snow != 205)
            snow_map['snow'] = xr.where(msk, 100, snow_map.snow)
                
            # clouds
            snow_map['snow'] = xr.where(snow_map.CLM>CLM_th, 205, snow_map.snow)
            snow_map['snow'] = xr.where(snow_map.NDSI.isnull(), 254, snow_map.snow)       
            
            snow_map['snow'] = snow_map.snow.astype(np.int8)
            snow_map = snow_map.drop('CLM')
            snow_map = snow_map.drop(3)
            snow_map['elevation'] = snow_map['elevation'].isel(time=0)

            # # plotting
            # if plotting:
            #     snow_map_3413 = snow_map.rio.reproject("EPSG:3413", resampling = Resampling.nearest, nodata=999)
            #     snow_map_3413 = snow_map_3413.where(snow_map_3413.NDSI != 999)
                
            #     fig,ax = plt.subplots(1,3,figsize=(18,8), sharex=True,sharey=True)
            #     plt.subplots_adjust(left=0.05, right=0.93, wspace=0.3)
            #     if snow_map_3413.NDSI.notnull().any():
            #         snow_map_3413.NDSI.plot(cmap='magma', ax=ax[0])
            #     # snow_map.CLM.where(snow_map.CLM>CLM_th).plot(cmap = 'Resnow_map', vmin=0, vmax=1, alpha=0.6, ax=ax[0])
            #     X, Y = np.meshgrid(snow_map_3413.x, snow_map_3413.y)
            #     ax[0].hexbin(X.reshape(-1), Y.reshape(-1), 
            #                snow_map_3413.CLM.where(snow_map_3413.CLM>CLM_th).values.reshape(-1), 
            #                grisnow_mapize=(50,50), cmap='Greens_r', alpha=0.5)
                
            #     snow_map_3413.B04.plot(cmap='magma', ax=ax[1])
            #     # snow_map.CLM.where(snow_map.CLM>CLM_th).plot(cmap = 'Resnow_map', vmin=0, vmax=1, alpha=0.6, ax=ax[0])
            #     X, Y = np.meshgrid(snow_map_3413.x, snow_map_3413.y)
            #     ax[1].hexbin(X.reshape(-1), Y.reshape(-1), 
            #                snow_map_3413.CLM.where(snow_map_3413.CLM>CLM_th).values.reshape(-1), 
            #                grisnow_mapize=(50,50), cmap='Greens_r', alpha=0.5)
    
            #     isnow_col = {0: 'darkred',
            #                    100: 'dodgerblue',
            #                    205: 'lightgray',
            #                    254: 'black'}
            #     vals = np.sort(np.unique(snow_map_3413.snow))
            #     vals = np.array([0 , 100, 205, 254, 255])
            #     cmap, norm = from_levels_and_colors(
            #         vals, [isnow_col[i] for i in [0 , 100, 205, 254]]
            #     )
            #     mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
            #     mapper.set_array([vals[0] - 0.5, vals[-1] + 1])
            #     param = {"cmap": cmap, "norm": norm, "add_colorbar": False}
            #     snow_map_3413.snow.plot(ax=ax[2], **param)
            #     cbar1 = fig.colorbar(mapper, ax=ax[2])
            #     cbar1.set_ticks(np.array(vals[:-1]) + np.diff(np.array(vals)) / 2)
            #     cbar1.ax.set_yticklabels([isnow_label[v] for v in vals[:-1]])
            #     for i in range(3):
            #         ax[i].set_title('')
            #         ax[i].axes.get_xaxis().set_ticks([])
            #         ax[i].axes.get_yaxis().set_ticks([])
            #         ax[i].set_xlabel('')
            #         ax[i].set_ylabel('')
            #     fig.suptitle(filename[15:25]+' '+filename[26:36])
            #     fig.savefig('plots/snow_maps/'+filename[:-4]+'.png')
            #     plt.close()
            
            snow_map = snow_map.rio.write_crs("EPSG:4326")
            print('printing file')
            snow_map.sortby('time').to_netcdf('data/S2/snowmap_' + df_fa.loc[k,'name']+'_'+ str(year) + '.nc')
