# -*- coding: utf-8 -*-
"""
@author: bav@geus.dk

tip list:
    %matplotlib inline
    %matplotlib qt
    import pdb; pdb.set_trace()
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from textwrap import wrap


plt.close('all')
notes = pd.read_csv('data/time lapse/notes.txt', skipinitialspace=True)
notes['time'] = pd.to_datetime(notes.time)
notes = notes.set_index(['site', 'time']).sort_index()
df_gst = pd.read_csv('data/GST_data.csv', skipinitialspace=True)
df_gst['time'] = pd.to_datetime(df_gst.time)
df_gst = df_gst.set_index(['site', 'time']).sort_index()

color_code = {'snow free': 'wheat',
              'left side picture snow free': 'wheat',
              'patchy snow on rest of picture': 'lightcyan',
              'base snow free': 'lightcyan',
              'logger melted out': 'lightcyan',
              'logger out rest of picture snow covered': 'lightcyan',
              'partial snow cover': 'lightblue',
              'partially snow covered': 'lightblue',
              'left side picture partially melted': 'lightblue',
              'left side picture almost melted': 'lightblue',
              'left side picture partially snow cover': 'lightblue',
              'lower part snow covered': 'lightblue',
              'snow cover': 'blue',
              'snow covered': 'blue',
              'snow cover except wall behind logger': 'blue',
              'left side picture covered with snow while logger still exposed': 'blue',
              'thick snow cover': 'steelblue',
              'polar night': 'gray',
              'camera covered': 'gray',
              }
for site_id in notes.index.get_level_values('site').unique():
# site_id = 'ae1'
    all_sites=df_gst.index.get_level_values('site').unique()
    if len(all_sites[all_sites.str.lower().str.contains(site_id)])==0:
        continue
    site = all_sites[all_sites.str.lower().str.contains(site_id)].values[0]
    print(site)
    fig,ax = plt.subplots(1,1, figsize=(7,3.5))
    plt.subplots_adjust(top=0.6)
    df_gst.loc[site,'GST'].plot(ax=ax, label='_nolegend_')
    
    site_notes = notes.loc[site_id,'note']
    cmap = cm.get_cmap('tab20', len(site_notes.unique()))
    for i in range(len(site_notes)-1):
        ax.axvspan(site_notes.index[i].to_pydatetime(),
               site_notes.index[i+1].to_pydatetime(),
               alpha=0.5,color=color_code[site_notes.iloc[i]], 
                                             label='_nolegend_')
    labels = [ '\n'.join(wrap(l, 20)) for l in site_notes.unique()]
    for i in range(len(site_notes.unique())):
        ax.axvspan(np.nan, np.nan,
               alpha=0.5,color=color_code[site_notes.unique()[i]],
               label=labels[i])
    plt.legend(title=site, ncol=3, loc='lower center', bbox_to_anchor=(0.5,1.05))
    plt.ylabel('GST ($^o$C)')
    plt.xlim(site_notes.index[0], site_notes.index[-1])
    # plt.tight_layout()
    fig.savefig('plots/'+site+'_GST_labeled.png',bbox_inches='tight', dpi=300)