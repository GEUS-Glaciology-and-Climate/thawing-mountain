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
    filter_times,
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
import datetime as dt

# Getting catalogue
from sentinelhub import SentinelHubCatalog
catalog = SentinelHubCatalog(config=config)
# catalog.get_info()
collections = catalog.get_collections()
collections = [collection for collection in collections if not collection["id"].startswith(("byoc", "batch"))]
     
def get_request_info(request_file):
    with open(request_file, 'r') as req:
        request = json.load(req)['request']
        bbox = '_'.join([str(s).replace('.','_') for s in request['payload']['input']['bounds']['bbox']])
        date_start = request['payload']['input']['data'][0]['dataFilter']['timeRange']['from'][:13]
        date_end = request['payload']['input']['data'][0]['dataFilter']['timeRange']['to'][:13]
        source = request['payload']['input']['data'][0]['type']
    return source, date_start, date_end, bbox

def clear_process_folder(request_all_bands):
    for folder in next(os.walk(request_all_bands.data_folder))[1]:
        path = request_all_bands.data_folder + '/' + folder+'/'
        try:
            info = get_request_info(path+'request.json')
        except:
            print('Did not find request.json in',path)
            continue
        shutil.move(path+'/response.tiff',  
                    request_all_bands.data_folder+'\\' +  \
                        ('_'.join(info)).replace(':','')+'.tif')
        shutil.move(path+'/request.json',  
                    request_all_bands.data_folder+'\\' +  \
                        ('_'.join(info)).replace(':','')+'.json')
        os.rmdir(path)
        print('Removed', folder)

evalscript_all_bands = """
    //VERSION=3
    function setup() {
        return {
            input: [{
                datasource: "l2a",
                bands: ["B03","B04","B11", "CLM","dataMask"],
                units: ["reflectance","reflectance","reflectance","DN","DN"]
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
    if (sample.dataMask==1) {
            if (dem.DEM > 0) {
    return [
            (sample.B03-sample.B11)/(sample.B03+sample.B11),
            sample.B04,
            sample.CLD,
            sample.CLM,
            dem.DEM + 12000]
    }
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

# %% all bands, multi-time
import geopandas as gpd
df_fa = gpd.read_file('data/GIS/FocusAreas.shp')
resolution = 20
for year in range(2020,2022):
    time_interval = str(year)+"-01-01", str(year+1)+"-01-01"
    
    for k in df_fa.index[-1:]:
        dir_name= 'data/S2/'+str(k+1) + '_' + df_fa.loc[k,'name']
        try:
            os.mkdir(dir_name)
        except:
            pass
        print(dir_name)
        geometry = Geometry(geometry={"type":"Polygon",
                                      "coordinates":[[[np.round(x,4), np.round(y,4)] \
                            for (x,y) in df_fa.loc[k,'geometry'].exterior.coords]]},
                            crs=CRS.WGS84)
        ROI_coords_wgs84 = [np.round(x,4) for x in df_fa.loc[k,'geometry'].bounds]
        ROI_bbox = BBox(bbox=ROI_coords_wgs84, crs=CRS.WGS84)
        ROI_size = bbox_to_dimensions(ROI_bbox, resolution=resolution)
        
        search_iterator = catalog.search(
            DataCollection.SENTINEL2_L2A,
            bbox=ROI_bbox,
            time=time_interval,
            filter="eo:cloud_cover < 90",
            fields={"include": ["id", "properties.datetime", "properties.eo:cloud_cover"], "exclude": []},
        )
        
        results = list(search_iterator)
        print("Total number of results:", len(results))
        time_difference = dt.timedelta(hours=1)
        all_timestamps = search_iterator.get_timestamps()
        unique_acquisitions = filter_times(all_timestamps, time_difference)
        
        process_requests = []
        for timestamp in unique_acquisitions:
            request = SentinelHubRequest(
                data_folder=dir_name,
                evalscript=evalscript_all_bands,
                input_data=[
                    SentinelHubRequest.input_data(
                        data_collection=DataCollection.SENTINEL2_L2A,
                        time_interval=(timestamp - time_difference, timestamp + time_difference),
                        identifier="l2a",  # has to match Sentinel input datasource id in evalscript
                        mosaicking_order=MosaickingOrder.LEAST_CC,
                    ),
                    SentinelHubRequest.input_data(
                        data_collection=DataCollection.DEM_COPERNICUS_30,
                        identifier="dem",  # has to match Sentinel input datasource id in evalscript
                    )
                ],
                responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
                bbox=ROI_bbox,
                size=ROI_size,
                config=config,
                geometry=geometry,
            )
            process_requests.append(request)
            
        client = SentinelHubDownloadClient(config=config)
        download_requests = [request.download_list[0] for request in process_requests]
        data = client.download(download_requests)
        data[0].shape
        # all_bands_response = request_all_bands.get_data(save_data=True)
    clear_process_folder(request)
    # all_bands_img_redownload = request_all_bands.get_data(redownload=True)
    # Image showing the SWIR band B12
    # Factor 1/1e4 due to the DN band values in the range 0-10000
    # Factor 3.5 to increase the brightness
    plot_image(data[0][:, :,0])
