# -*- coding: utf-8 -*-
"""
@author: bav@geus.dk

tip list:
    %matplotlib inline
    %matplotlib qt
    import pdb; pdb.set_trace()
"""

from sentinelhub import SHConfig

config = SHConfig()
config.instance_id = '0f5116b4-2067-40eb-9db0-54e24ea143ab'
config.sh_client_id = '199668de-bc44-4c42-832f-83d77da0284f'
config.sh_client_secret = '@0xC+r~kke4n&8pk/TC-bPb2z9bmf^&:L;*|djHy'
config.save()

# Niels ID:
# config = SHConfig()
# config.instance_id = 'c406a27c-67ad-4e6b-b1a6-32020f396b05' #Instance ID needed (User ID)
# config.sh_client_id = '27afa300-f205-4468-86f3-9e22f97f3cf7' # OAuth Client ID needed
# config.sh_client_secret = '|iPVXn-AD8is6MR/LXL?FaHVpFXL+j6(F9FWKaF;' #Client secret needed
# config.save()  #Save client login so not needed again   

import os
import matplotlib.pyplot as plt
import numpy as np

from sentinelhub import (
    CRS,
    BBox,
    DataCollection,
    DownloadRequest,
    MimeType,
    MosaickingOrder,
    SentinelHubDownloadClient,
    SentinelHubRequest,
    bbox_to_dimensions,
    Geometry,
)

# The following is not a package. It is a file utils.py which should be in the same folder as this notebook.
from utils import plot_image
import xarray as xr
import rioxarray
import rasterio
import pandas as pd
import json
import shutil
import matplotlib.cm as cm
from matplotlib.colors import from_levels_and_colors
       
def get_request_info(request_file):
    with open(request_file, 'r') as req:
        request = json.load(req)['request']
        bbox = '_'.join([str(s).replace('.','_') for s in request['payload']['input']['bounds']['bbox']])
        date_start = request['payload']['input']['data'][0]['dataFilter']['timeRange']['from'][:10]
        date_end = request['payload']['input']['data'][0]['dataFilter']['timeRange']['to'][:10]
        source = request['payload']['input']['data'][0]['type']
    return source, date_start, date_end, bbox

def clear_process_folder(request_all_bands):
    for folder in next(os.walk(request_all_bands.data_folder))[1]:
        path = request_all_bands.data_folder + '/' + folder+'/'
        info = get_request_info(path+'request.json')
        shutil.move(path+'/response.tiff',  
                    request_all_bands.data_folder+'\\' +  \
                        ('_'.join(info)).replace(':','')+'.tif')
        shutil.move(path+'/request.json',  
                    request_all_bands.data_folder+'\\' +  \
                        ('_'.join(info)).replace(':','')+'.json')
        os.rmdir(path)
        print('Removed', folder)


betsiboka_coords_wgs84 = [-53.9, 66.8, -53.1, 67.2]


geometry = Geometry(geometry={"type":"Polygon",
                              "coordinates":[[[-53.84304533658620073, 67.10460657094161263],
                                              [-53.21463270213931906, 67.13861532259072362],
                                              [-53.10591115636343318, 66.90549066894622854],
                                              [-53.74562323102417594, 66.86494959291036366],
                                              [-53.84304533658620073, 67.10460657094161263]]]},
                    crs=CRS.WGS84)

resolution = 10
betsiboka_bbox = BBox(bbox=betsiboka_coords_wgs84, crs=CRS.WGS84)
betsiboka_size = bbox_to_dimensions(betsiboka_bbox, resolution=resolution)

print(f"Image shape at {resolution} m resolution: {betsiboka_size} pixels")

# %% all bands
evalscript_all_bands = """
    //VERSION=3
    function setup() {
        return {
            input: [{
                datasource: "l2a",
                bands: ["B03","B04","B11", "CLM"],
                units: ["reflectance","reflectance","reflectance","DN"]
            },
            {
                datasource: "dem",
                bands: ["DEM"]
            }],
            output: [{
                bands: 5,
                sampleType: SampleType.FLOAT32
            }]
        };
    }

function evaluatePixel(samples, inputData, inputMetadata, customData, outputMetadata) {
    sample = samples.l2a[0]
    dem = samples.dem[0]
    if (dem.DEM > 0) {
    return [
            (sample.B03-sample.B11)/(sample.B03+sample.B11),
            sample.B04,
            sample.CLD,
            sample.CLM,
            dem.DEM + 12000]
    }
    }

"""
# Conclusion:
# SCL	Defective. Mark shaddows as saturated or water
# SNW misses the snowy parts in shaddowed areas.
# CLD	Cloud probability,bare rock as cloud
# CLP does not work. Flags the entire area as snow covered.
# CLM works ok. It is a binary map.
# NDSI works fine in shaddowed areas.

# test all areas on single date
import pandas as pd
import geopandas as gpd
df_fa = gpd.read_file('data/GIS/FocusAreas.shp')
df_fa.index = df_fa.index + 1
time_stamps = pd.date_range('2016-10-01', '2022-11-01', freq='1D')
resolution = 20

for k in df_fa.index:
    print(k)
    geometry = Geometry(geometry={"type":"Polygon",
                                  "coordinates":[[[np.round(x,4), 
                                                   np.round(y,4)] for (x,y) in df_fa.loc[k,'geometry'].exterior.coords]]},
                        crs=CRS.WGS84)
    betsiboka_coords_wgs84 = [np.round(x,4) for x in df_fa.loc[k,'geometry'].bounds]
    betsiboka_bbox = BBox(bbox=betsiboka_coords_wgs84, crs=CRS.WGS84)
    betsiboka_size = bbox_to_dimensions(betsiboka_bbox, resolution=resolution)
        
    request_all_bands = SentinelHubRequest(
        data_folder="snow_maps",
        evalscript=evalscript_all_bands,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A,
                time_interval=("2021-03-08", "2021-03-15"),
                identifier="l2a",  # has to match Sentinel input datasource id in evalscript
                mosaicking_order=MosaickingOrder.LEAST_CC,
            ),
            SentinelHubRequest.input_data(
                data_collection=DataCollection.DEM_COPERNICUS_30,
                identifier="dem",  # has to match Sentinel input datasource id in evalscript
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=betsiboka_bbox,
        size=betsiboka_size,
        config=config,
        geometry=geometry,
    )
    all_bands_response = request_all_bands.get_data(save_data=True)
clear_process_folder(request_all_bands)
# all_bands_img_redownload = request_all_bands.get_data(redownload=True)
# Image showing the SWIR band B12
# Factor 1/1e4 due to the DN band values in the range 0-10000
# Factor 3.5 to increase the brightness
plot_image(all_bands_response[0][:, :, -1])

#%% Multi-time step request
import pandas as pd
import geopandas as gpd
df_fa = gpd.read_file('data/GIS/FocusAreas.shp')
df_fa.index = df_fa.index + 1
time_stamps = pd.date_range('2016-10-01', '2022-11-01', freq='1D')
resolution = 20

for k in df_fa.index[2:3]:
    dir_name= str(k) + '_' + df_fa.loc[k,'name']
    try:
        os.mkdir(dir_name)
    except:
        pass
    print(dir_name)
    geometry = Geometry(geometry={"type":"Polygon",
                                  "coordinates":[[[np.round(x,4), np.round(y,4)] for (x,y) in df_fa.loc[k,'geometry'].exterior.coords]]},
                        crs=CRS.WGS84)
    betsiboka_coords_wgs84 = [np.round(x,4) for x in df_fa.loc[k,'geometry'].bounds]
    betsiboka_bbox = BBox(bbox=betsiboka_coords_wgs84, crs=CRS.WGS84)
    betsiboka_size = bbox_to_dimensions(betsiboka_bbox, resolution=resolution)
    
    for i in range(1614, len(time_stamps)-1):
        print(time_stamps[i])
        request_all_bands = SentinelHubRequest(
            data_folder=dir_name,
            evalscript=evalscript_all_bands,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=DataCollection.SENTINEL2_L2A,
                    identifier="l2a",  # has to match Sentinel input datasource id in evalscript
                    time_interval=(time_stamps[i], time_stamps[i+1]),
                ),
                SentinelHubRequest.input_data(
                    data_collection=DataCollection.DEM_COPERNICUS_30,
                    identifier="dem",  # has to match Sentinel input datasource id in evalscript
                )
            ],
            responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
            bbox=betsiboka_bbox,
            size=betsiboka_size,
            geometry=geometry,
            config=config,
        )
        data_with_cloud_mask = request_all_bands.get_data(save_data=True)
        
        plot_image(data_with_cloud_mask[0][:,:,1], factor = 4)
    
    clear_process_folder(request_all_bands)
    
# %% Plotting data
from osgeo import gdal
gdal.PushErrorHandler('CPLQuietErrorHandler')
from  rasterio.enums import Resampling
import pandas as pd
import geopandas as gpd
import warnings

df_fa = gpd.read_file('data/GIS/FocusAreas.shp')
df_fa.index = df_fa.index + 1
# gdal.UseExceptions()
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')   
    for k in df_fa.index[2:]:
        dir_name= 'data/S2/'+str(k) + '_' + df_fa.loc[k,'name']
        try:
            os.mkdir(dir_name)
        except:
            pass
        print(dir_name)
        
        geometry = Geometry(geometry={"type":"Polygon",
                                      "coordinates":[[[np.round(x,4), np.round(y,4)] for (x,y) in df_fa.loc[k,'geometry'].exterior.coords]]},
                            crs=CRS.WGS84)
        betsiboka_coords_wgs84 = [np.round(x,4) for x in df_fa.loc[k,'geometry'].bounds]
        betsiboka_bbox = BBox(bbox=betsiboka_coords_wgs84, crs=CRS.WGS84)
        betsiboka_size = bbox_to_dimensions(betsiboka_bbox, resolution=resolution)
            
        snow_map = xr.DataArray()
        for filename in os.listdir(dir_name):
            if filename.endswith('tif'):
                print('%i/%i %s'% (os.listdir(dir_name).index(filename), 
                                     len(os.listdir(dir_name)),
                                     filename))
                col_name = {1: 'NDSI',
                            2: 'B04',
                            4: 'CLM',
                            5: 'elevation'}
                ds = xr.open_dataarray(dir_name+'/'+filename).to_dataset('band').rename(col_name)
                ds['elevation'] = ds.elevation - 12000
                ds['NDSI'] = ds['NDSI'].where(ds.elevation>0)
                ds['B04'] = ds['B04'].where(ds.elevation>0)
                ds['elevation'] = ds['elevation'].where(ds.elevation>0)
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
                
                # no data
                ds['snow'] = xr.where(ds.B04.isnull(), 254, 0)       
                
                # first pass:
                msk = (ds.NDSI > n1) & (ds.B04 > r1)
                ds['snow'] = xr.where(msk, 100, ds.snow)
                
                # elevation of snowy pixels
                elev_snow = ds.elevation.where(ds.snow==100)
                
                # plt.figure()
                # ds.elevation.plot.hist()
                # elev_snow.plot.hist()
                zs = elev_snow.quantile(0.01,skipna=True)
                
                # second pass:
                if (ds.snow==100).sum() / ds.elevation.notnull().sum() > ft:
                    msk = ((ds.NDSI > n2) & (ds.B04 > r2)) & (ds.elevation > zs)
                    ds['snow'] = xr.where(msk, 100, ds.snow)
                    
                # thrid pass for shadows:
                msk = (ds.NDSI > 0.8) & (ds['snow'] != 205)
                ds['snow'] = xr.where(msk, 100, ds.snow)
                    
                # clouds
                ds['snow'] = xr.where(ds.CLM>CLM_th, 205, ds.snow)
        
        
                ds['snow'] = xr.where(ds.NDSI.isnull(), 254, ds.snow)       
        
                if (ds['snow']==254).all():
                    print('no data')
                    continue
                # plotting
                ds_3413 = ds.rio.reproject("EPSG:3413", resampling = Resampling.nearest, nodata=999)
                ds_3413 = ds_3413.where(ds_3413.NDSI != 999)
                
                fig,ax = plt.subplots(1,3,figsize=(18,8), sharex=True,sharey=True)
                plt.subplots_adjust(left=0.05, right=0.93, wspace=0.3)
                if ds_3413.NDSI.notnull().any():
                    ds_3413.NDSI.plot(cmap='magma', ax=ax[0])
                # ds.CLM.where(ds.CLM>CLM_th).plot(cmap = 'Reds', vmin=0, vmax=1, alpha=0.6, ax=ax[0])
                X, Y = np.meshgrid(ds_3413.x, ds_3413.y)
                ax[0].hexbin(X.reshape(-1), Y.reshape(-1), 
                           ds_3413.CLM.where(ds_3413.CLM>CLM_th).values.reshape(-1), 
                           gridsize=(50,50), cmap='Greens_r', alpha=0.5)
                
                ds_3413.B04.plot(cmap='magma', ax=ax[1])
                # ds.CLM.where(ds.CLM>CLM_th).plot(cmap = 'Reds', vmin=0, vmax=1, alpha=0.6, ax=ax[0])
                X, Y = np.meshgrid(ds_3413.x, ds_3413.y)
                ax[1].hexbin(X.reshape(-1), Y.reshape(-1), 
                           ds_3413.CLM.where(ds_3413.CLM>CLM_th).values.reshape(-1), 
                           gridsize=(50,50), cmap='Greens_r', alpha=0.5)
    
                isnow_col = {0: 'darkred',
                               100: 'dodgerblue',
                               205: 'lightgray',
                               254: 'black'}
                vals = np.sort(np.unique(ds_3413.snow))
                vals = np.array([0 , 100, 205, 254, 255])
                cmap, norm = from_levels_and_colors(
                    vals, [isnow_col[i] for i in [0 , 100, 205, 254]]
                )
                mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
                mapper.set_array([vals[0] - 0.5, vals[-1] + 1])
                param = {"cmap": cmap, "norm": norm, "add_colorbar": False}
                ds_3413.snow.plot(ax=ax[2], **param)
                cbar1 = fig.colorbar(mapper, ax=ax[2])
                cbar1.set_ticks(np.array(vals[:-1]) + np.diff(np.array(vals)) / 2)
                cbar1.ax.set_yticklabels([isnow_label[v] for v in vals[:-1]])
                for i in range(3):
                    ax[i].set_title('')
                    ax[i].axes.get_xaxis().set_ticks([])
                    ax[i].axes.get_yaxis().set_ticks([])
                    ax[i].set_xlabel('')
                    ax[i].set_ylabel('')
                fig.suptitle(filename[15:25]+' '+filename[26:36])
                fig.savefig('plots/snow_maps/'+filename[:-4]+'.png')
                plt.close()
                
                tmp = ds['snow'].expand_dims(dim={"time": [pd.to_datetime(filename[26:36])]})
                tmp = tmp.rio.write_crs("EPSG:4326")
                tmp = tmp.rio.reproject("EPSG:3413",nodata=-999)
                if len(snow_map.shape) == 0:
                    snow_map = tmp.astype(int)
                else:
                    snow_map = xr.concat((snow_map, tmp.astype(int)), dim='time')
    
        date_start = pd.to_datetime(snow_map.time.min().values).strftime('%Y-%m-%d')
        date_end = pd.to_datetime(snow_map.time.max().values).strftime('%Y-%m-%d')
        snow_map.to_netcdf(dir_name+'/snowmap_' + date_start +'_' \
                           + date_end + '.nc')

