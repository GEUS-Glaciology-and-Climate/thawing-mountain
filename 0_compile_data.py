# -*- coding: utf-8 -*-
"""
@author: bav@geus.dk

tip list:
    %matplotlib inline
    %matplotlib qt
    import pdb; pdb.set_trace()
"""


import os
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pytz

# Get files in folder
fold = 'data/Marco'
files = os.listdir(fold)[:-2]
print(files)
folder_plots = "plots"

# Iterate through all points in the folder
df_all = pd.DataFrame()
for logger in files:    
    print("Working on", logger)    

    # Geoprecision loggers files start with letter A                          
    if logger[0] == "A":
        # print("Geoprecision exception")
        df = pd.read_csv(fold+"/"+ logger, comment = '<')
        if len(df.columns) == 5:
            df.columns = ['NO','time','GST','T2','V']       
        if len(df.columns) == 4:
            df.columns = ['NO','time','GST','V']
        df.time = pd.to_datetime(df.time, format='%d.%m.%Y %H:%M:%S') #.dt.tz_localize('Etc/GMT+2')
        df=df.set_index('time')
    else :          
        # print("iButton exception")
        df = pd.read_csv(fold+"/"+ logger, header = 18).reset_index()
        df['time'] = pd.to_datetime(df['index'], format='%d-%m-%y %H:%M:%S')
        df = df.set_index('time')
        df['GST'] = df.Unit + df.Value/1000
    
    df['site'] = logger.split('_')[0][:-4]
    df_all = df_all.append(df.reset_index())
    # Plot data and save figure
    fig = plt.figure()
    df.GST.plot()
    plt.suptitle(logger)
    plt.savefig(folder_plots + "/sensors/" + logger.split('.')[0].split('_')[0]+ ".png")
    plt.close()
    
# %% loading michele's data
import xarray as xr 

ds = xr.open_dataset('data/Michele/t_strings.nc')
#%%
for var in list(ds.keys()):
    dims = ds[var].dims
    df = pd.DataFrame()
    df['GST'] = ds[var][:,0].to_dataframe().iloc[:,-1]
    df['site'] = var
    df.index = pd.to_datetime(df.index)
    df_all = df_all.append(df.reset_index())
    print('added ', var,'to dataset')

#%% 
df_all = df_all.loc[df_all['GST'].notnull(),:]
df_all.set_index(['time','site'])['GST'].to_csv('data/GST_data.csv', float_format='%0.3f')
