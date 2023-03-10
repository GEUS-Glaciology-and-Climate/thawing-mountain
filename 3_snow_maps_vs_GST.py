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
import numpy as np
plt.close('all')
import os
import geopandas as gpd

df_fa = gpd.read_file('data/GIS/FocusAreas.shp')
df = pd.read_csv('data/coordinates_all.csv', sep=';').set_index('ID')
# plt.close('all')
# notes = pd.read_csv('data/time lapse/notes.txt', skipinitialspace=True)
# notes['time'] = pd.to_datetime(notes.time)
# notes = notes.set_index(['site', 'time']).sort_index()

df_gst = pd.read_csv('data/GST_data.csv', skipinitialspace=True)
df_gst['time'] = pd.to_datetime(df_gst.time)
df_gst['site'] = df_gst.site.astype(str)
df_gst['snow_flag'] = 254
df_gst_first = df_gst.set_index('time').groupby('site').resample('D').first()
df_gst =  df_gst.set_index('time').groupby('site').resample('D').mean(numeric_only=True)
df_gst = df_gst.sort_index()


# plt.close('all')
isnow_label = {0: 'no snow',
               100: 'snow',
               205: 'cloud',
               254: 'no data'}
isnow_color = {0: 'red',
               100: 'blue',
               205: 'lightgreen',
               254: 'gray'}

for area in df_fa.name[:1]:
    print(area)
    # file_list = ['data/S2/snowmap_'+area+'_'+str(y)+'_sorted.nc' \
    #                            for y in range(2016, 2023) \
    #                            if 'snowmap_'+area+'_'+str(y)+'_sorted.nc' in os.listdir('data/S2')]
    # ds_sm = xr.open_mfdataset(file_list)
    ds_sm = xr.open_dataset('data/S2/snowmap_'+area+'_'+'_filled.nc')
    ds_sm['snow'] = ds_sm.snow.sortby('time')
    ds_sm = ds_sm.rio.write_crs("EPSG:3413")
    ds_sm = ds_sm.rio.reproject("EPSG:4326",nodata=-999)
    

    for site in df.index:
    # site = '779A72'
    
        # all_sites=df_gst.index.get_level_values('site').unique()
        # if len(all_sites[all_sites.str.lower().str.contains(site_id)])==0:
        #     continue
        # site = all_sites[all_sites.str.lower().str.contains(site_id)].values[0]
        site_flag = ds_sm.sel(x=df.loc[site].longitude, 
                              y=df.loc[site].latitude, 
                              method='nearest').reset_coords(['x', 'y'], 
                                                             drop=True).to_dataframe()['snow']
        if (site_flag == -999).all():
            continue
        
        if site in df_gst.index.get_level_values(0):
            fig,ax = plt.subplots(1,1, figsize=(7,3.5))
            plt.subplots_adjust(top=0.6)
        try:
            df_gst.loc[site,'GST'].plot(ax=ax, label='_nolegend_')
        except:
            continue
        print(site)
        # site_notes = notes.loc[site_id,'note']
        # cmap = cm.get_cmap('tab20', len(site_notes.unique()))
        # for i in range(len(site_notes)-1):
        #     ax.axvspan(site_notes.index[i].to_pydatetime(),
        #            site_notes.index[i+1].to_pydatetime(),
        #            alpha=0.5,color=color_code[site_notes.iloc[i]], 
        #                                          label='_nolegend_')
        # labels = [ '\n'.join(wrap(l, 20)) for l in site_notes.unique()]
        # for i in range(len(site_notes.unique())):
        #     ax.axvspan(np.nan, np.nan,
        #            alpha=0.5,color=color_code[site_notes.unique()[i]],
        #            label=labels[i])
        
        overlap = df_gst.loc[site,:].index[np.isin(df_gst.loc[site,:].index,
                          site_flag.index)]
        df_gst.loc[(site, overlap),'snow_flag'] = site_flag.loc[overlap].values
        for i in range(len(site_flag)-1):
            ax.axvspan(site_flag.index[i].to_pydatetime(),
                   site_flag.index[i+1].to_pydatetime(),
                   alpha=0.5,color=isnow_color[site_flag.iloc[i]], 
                                                 label='_nolegend_')
        for flag in site_flag.unique():
            ax.axvspan(np.nan, np.nan,
                   alpha=0.5,color=isnow_color[flag],
                   label=isnow_label[flag],
                   edgecolor=None)
        plt.legend(title=site, ncol=3, loc='lower center', bbox_to_anchor=(0.5,1.05))
        plt.ylabel('GST ($^o$C)')
        # plt.xlim(site_flag.index[0], site_flag.index[-1])
        # plt.tight_layout()
        fig.savefig('plots/'+site+'_GST_labeled.png',bbox_inches='tight', dpi=300)
    