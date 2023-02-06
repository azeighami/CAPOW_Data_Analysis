# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 15:42:33 2021

@author: mzeigha
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import geopandas as gpd
from shapely.geometry import Point
from matplotlib.colors import TwoSlopeNorm
from textwrap import wrap



##### Reading Data 

stream_annual = pd.read_csv('Results/stream_annual.csv').T
stream_index = stream_annual.index
stream_annual=stream_annual[stream_annual.sum(axis=1)>0]


wind_annual = pd.read_csv('Results/wind_annual.csv').T
temp_annual = pd.read_csv('Results/temp_annual.csv').T
irradiance_annual = pd.read_csv('Results/irradiance_annual.csv').T
emission = pd.read_csv('Results/emission_no_tax.csv',header=0, index_col = "name").set_index(pd.RangeIndex(220))
# county_damage = pd.read_csv('Crosswalk/county_damage.csv', dtype={"fips": str})
county_damage = pd.read_csv('Results/county_damage_per_generator.csv', index_col = 0)

production_no_tax = pd.read_csv('Results/production_df_no_tax.csv').groupby('Year').sum().T

no_tax = production_no_tax[0:220]

########## Reading the Capacity factors
df = pd.DataFrame( index = emission.index, columns = range(0,500))
for year in range (0,500):
    df.iloc[:,year] = (emission['PMdol{}'.format(str(year))]/emission['PMTax($/kWh)'])/(8760*emission['netcap'])
    
#%%
damage = pd.DataFrame()
for i in range (0,500):
    for j in county_damage.columns:
        damage.loc[j,i] = (county_damage.loc[:,j]*no_tax.loc[:,i]).sum()

#%%

stream_percentile = pd.DataFrame([])
wind_percentile = pd.DataFrame([])
temp_percentile = pd.DataFrame([])
irradiance_percentile = pd.DataFrame([])
capacity_factor_deviation = pd.DataFrame([])
county_damage_deviation = pd.DataFrame([])

for j in [100,151,495]:

    capacity_factor_deviation[j] = df.iloc[:,j] - df.iloc[:,0:500].mean(axis = 1)
    county_damage_deviation.loc[:,j] = (damage.loc[:,j] - damage.mean(axis = 1))/damage.mean(axis = 1)

    
    for i in range (len(stream_annual)):
        stream_percentile.loc[i,j] = stats.percentileofscore(stream_annual.values[i,:], stream_annual.values[i,j])


    for i in range (len(wind_annual)):
        wind_percentile.loc[i,j] = stats.percentileofscore(wind_annual.values[i,:], wind_annual.values[i,j])
    

    for i in range (len(temp_annual)):
        temp_percentile.loc[i,j] = stats.percentileofscore(temp_annual.values[i,:], temp_annual.values[i,j])
    

    for i in range (len(irradiance_annual)):
        irradiance_percentile.loc[i,j] = stats.percentileofscore(irradiance_annual.values[i,:], irradiance_annual.values[i,j])
        
#%%

county_damage_deviation['fips'] = damage.index
county_damage_deviation = county_damage_deviation.reset_index()


stream_coordinates = pd.read_csv('Results/streamflow_sites.csv',header=0, index_col = 'cap').set_index(stream_index)
stream_coordinates = stream_coordinates[stream_coordinates.index.isin(stream_annual.index)]

weather_coordinates = pd.read_csv('Results/weather_stations.csv',header=0)
shape = gpd.read_file('Results/USA_Counties/shape.shp').to_crs('EPSG:4326')
state_map = gpd.read_file('Results/USA_Counties/USA_States_Generalized.shp').to_crs('EPSG:4326')

crs = {'init':'epsg:4326'}

geometry_stream = [Point(xy) for xy in zip(stream_coordinates['Longitude'],stream_coordinates['Latitude'])]
geometry_weather = [Point(xy) for xy in zip(weather_coordinates['Longitude'],weather_coordinates['Latitude'])]
geometry = [Point(xy) for xy in zip(emission['LON'],emission['LAT'])]


####### Making GeoPandas dataframe for the maps
#%%

stream_geo = gpd.GeoDataFrame(stream_percentile , crs = crs, geometry= geometry_stream)
wind_geo = gpd.GeoDataFrame(wind_percentile, crs = crs,geometry= geometry_weather[0:17])
temp_geo = gpd.GeoDataFrame(temp_percentile, crs = crs,geometry= geometry_weather[0:17])
irradiance_geo = gpd.GeoDataFrame(irradiance_percentile, crs = crs,geometry= geometry_weather[17:])
geo_df = gpd.GeoDataFrame(capacity_factor_deviation, crs = crs,geometry=geometry)
geo_df ['netcap'] = emission['netcap']
geo_df = geo_df.sort_values( by = 'netcap', ascending=False)



for y in [100,151,495]:
    for i in range(len(shape)):
        for j in range(len(county_damage_deviation)):
            if str(shape.loc[i,'FIPS']) == county_damage_deviation.loc[j,'fips']:
                shape.loc[i,y] = county_damage_deviation.loc[j,y]

best = 495
worst = 151
ywwd = 100

#%%

# fig = plt.figure( figsize = (15.5,11))
# #constrained_layout=True,

# gs = fig.add_gridspec(3,6, width_ratios=[1, 1, 1, 1, 1.5, 1.5], height_ratios = [1, 1, 1.25])
csfont = {'fontname':'arial'}

fig = plt.figure( figsize = (15,10))
#constrained_layout=True,

gs = fig.add_gridspec(3,6, width_ratios=[1, 1, 1, 1, 1.40, 1.45], height_ratios = [1, 1, 1] , wspace=0, hspace=0.1 )

############# Wind 
cf_ax1 = fig.add_subplot(gs[0,0])
cf_ax2 = fig.add_subplot(gs[1,0])
cf_ax3 = fig.add_subplot(gs[2,0])


############# Temp 
cf_ax4 = fig.add_subplot(gs[0,1])
cf_ax5 = fig.add_subplot(gs[1,1])
cf_ax6 = fig.add_subplot(gs[2,1])


############# Irradiance 
cf_ax7 = fig.add_subplot(gs[0,2])
cf_ax8 = fig.add_subplot(gs[1,2])
cf_ax9 = fig.add_subplot(gs[2,2])


############# Streamflow 
cf_ax10 = fig.add_subplot(gs[0,3])
cf_ax11 = fig.add_subplot(gs[1,3])
cf_ax12 = fig.add_subplot(gs[2,3])


############# Capacity Factor 
cf_ax13 = fig.add_subplot(gs[0,4])
cf_ax14 = fig.add_subplot(gs[1,4])
cf_ax15 = fig.add_subplot(gs[2,4])


############# Damage Map
cf_ax16 = fig.add_subplot(gs[0,5])
cf_ax17 = fig.add_subplot(gs[1,5])
cf_ax18 = fig.add_subplot(gs[2,5])


############ Right hand side 
# cf_ax19 = fig.add_subplot(gs[0,6])
# cf_ax20 = fig.add_subplot(gs[1,6])
# cf_ax21 = fig.add_subplot(gs[2,6])


# # ############# Color bar
# cf_ax22 = fig.add_subplot(gs[3,4:6])
# cf_ax23 = fig.add_subplot(gs[4,4:6])

# ############# Legend 
# cf_ax24 = fig.add_subplot(gs[3,0:4])
# cf_ax25 = fig.add_subplot(gs[4,0:4])



################# Worst year Wind map
vmin, vmax, vcenter = 0 , 100, 50
norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
cmap = 'PiYG_r'
cbar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)


state_map.plot(ax=cf_ax1,color='gray',alpha=0.6,edgecolor='white',linewidth=0.5)

cf_ax1.set_xlim(-126,-108)
cf_ax1.set_ylim(30,53)

cf_ax1.set_xticks([])
cf_ax1.set_yticks([])
cf_ax1.set_title("Wind Speed", fontsize= 18,**csfont)

wind_geo.plot(ax=cf_ax1 , markersize = 200 ,cmap=cmap , norm=norm , column=worst , marker="^" , edgecolor='black' ,linewidth=0.8 ,legend=False)



################## Best year Wind map
state_map.plot(ax=cf_ax2,color='gray',alpha=0.6,edgecolor='white',linewidth=0.5)
cf_ax2.set_xlim(-126,-108)
cf_ax2.set_ylim(30,53)

cf_ax2.set_xticks([])
cf_ax2.set_yticks([])

wind_geo.plot(ax=cf_ax2 , markersize = 200 ,cmap=cmap , norm=norm , column=best , marker="^" , edgecolor='black' ,linewidth=0.8 ,legend=False)


################## Year with the worst day Wind map
state_map.plot(ax=cf_ax3,color='gray',alpha=0.6,edgecolor='white',linewidth=0.5)
cf_ax3.set_xlim(-126,-108)
cf_ax3.set_ylim(30,53)
cf_ax3.set_xticks([])
cf_ax3.set_yticks([])

wind_geo.plot(ax = cf_ax3 , markersize = 200 ,cmap = cmap , norm = norm , column = ywwd , marker = "^" , edgecolor = 'black' ,linewidth = 0.8 ,legend = False)
# cbar = plt.colorbar(mappable=cbar,ax = cf_ax3, orientation='horizontal', pad = 0.04, shrink= 0.8 ,aspect=8, ticks=[vmin,vmax])
# cbar.ax.tick_params(labelsize=13)

################# Worst year Temp map
vmin, vmax, vcenter = 0, 100, 50
norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
cmap = 'PiYG_r'
cbar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)


state_map.plot(ax=cf_ax4,color='gray',alpha=0.6,edgecolor='white',linewidth=0.5)
cf_ax4.set_xlim(-126,-108)
cf_ax4.set_ylim(30,53)
cf_ax4.set_xticks([])
cf_ax4.set_yticks([])
cf_ax4.set_title("Temperature", fontsize= 18,**csfont)

temp_geo.plot(ax=cf_ax4 , markersize = 200 ,cmap=cmap , norm=norm , column=worst , marker="^" , edgecolor='black' ,linewidth=0.8 ,legend=False)


################## Best year Temp map
state_map.plot(ax=cf_ax5,color='gray',alpha=0.6,edgecolor='white',linewidth=0.5)
cf_ax5.set_xlim(-126,-108)
cf_ax5.set_ylim(30,53)
cf_ax5.set_xticks([])
cf_ax5.set_yticks([])

temp_geo.plot(ax=cf_ax5 , markersize = 200 ,cmap=cmap , norm=norm , column=best , marker="^" , edgecolor='black' ,linewidth=0.8 ,legend=False)


################## Year with the worst day Temp map
state_map.plot(ax=cf_ax6,color='gray',alpha=0.6,edgecolor='white',linewidth=0.5)
cf_ax6.set_xlim(-126,-108)
cf_ax6.set_ylim(30,53)
cf_ax6.set_xticks([])
cf_ax6.set_yticks([])

temp_geo.plot(ax = cf_ax6 , markersize = 200 ,cmap = cmap , norm = norm , column = ywwd , marker = "^" , edgecolor = 'black' ,linewidth = 0.8 ,legend = False)
# cbar = plt.colorbar(mappable=cbar,ax = cf_ax6, orientation='horizontal', pad = 0.04, shrink= 0.8 ,aspect=8, ticks=[vmin,vmax])
# cbar.ax.tick_params(labelsize=13)

################## Worst year Irradiance map
vmin, vmax, vcenter =0, 100, 50 
norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
cmap = 'PiYG_r'
cbar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)


state_map.plot(ax=cf_ax7,color='gray',alpha=0.6,edgecolor='white',linewidth=0.5)
cf_ax7.set_xlim(-126,-108.25)
cf_ax7.set_ylim(30,53.55)
cf_ax7.set_xticks([])
cf_ax7.set_yticks([])
cf_ax7.set_title("Irradiance", fontsize= 18,**csfont)

irradiance_geo.plot(ax=cf_ax7 , markersize = 200 ,cmap=cmap , norm=norm , column=worst , marker="^" , edgecolor='black' ,linewidth=0.8 ,legend=False)


################## Best year Irradiance map
state_map.plot(ax=cf_ax8,color='gray',alpha=0.6,edgecolor='white',linewidth=0.5)
cf_ax8.set_xlim(-126,-108.25)
cf_ax8.set_ylim(30,53.55)
cf_ax8.set_xticks([])
cf_ax8.set_yticks([])

irradiance_geo.plot(ax=cf_ax8 , markersize = 200 ,cmap=cmap , norm=norm , column=best , marker="^" , edgecolor='black' ,linewidth=0.8 ,legend=False)


################## Year with the worst day Irradiance map
state_map.plot(ax=cf_ax9,color='gray',alpha=0.6,edgecolor='white',linewidth=0.5)
cf_ax9.set_xlim(-126,-108.25)
cf_ax9.set_ylim(30,53.55)
cf_ax9.set_xticks([])
cf_ax9.set_yticks([])

irradiance_geo.plot(ax = cf_ax9 , markersize = 200 ,cmap = cmap , norm = norm , column = ywwd , marker = "^" , edgecolor = 'black' ,linewidth = 0.8 ,legend = False)
# cbar = plt.colorbar(mappable=cbar,ax = cf_ax9, orientation='horizontal', pad = 0.04, shrink= 0.8 ,aspect=8, ticks=[vmin,vmax])
# cbar.ax.tick_params(labelsize=13)


################## Worst year Streamflow map
vmin, vmax, vcenter =0, 100 , 50 
norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
cmap = 'PiYG_r'
cbar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

state_map.plot(ax=cf_ax10,color='gray',alpha=0.6,edgecolor='white',linewidth=0.5)
cf_ax10.set_xlim(-126,-108)
cf_ax10.set_ylim(31.5,53.2)
cf_ax10.set_xticks([])
cf_ax10.set_yticks([])
cf_ax10.set_title("Streamflow", fontsize= 18,**csfont)

# stream_geo = stream_geo.sort_values( by = worst, ascending=True)
stream_geo.plot(ax=cf_ax10 , markersize = 200 ,cmap=cmap , norm=norm , column=worst , marker="^" , edgecolor='black' ,linewidth=0.8 ,legend=False)


################## Best year Streamflow map
state_map.plot(ax=cf_ax11,color='gray',alpha=0.6,edgecolor='white',linewidth=0.5)
cf_ax11.set_xlim(-126,-108)
cf_ax11.set_ylim(31.5,53.2)
cf_ax11.set_xticks([])
cf_ax11.set_yticks([])

# stream_geo = stream_geo.sort_values( by = best, ascending=True)
stream_geo.plot(ax=cf_ax11 , markersize = 200 ,cmap=cmap , norm=norm , column=best , marker="^" , edgecolor='black' ,linewidth=0.8 ,legend=False)


################## Year with the worst day Streamflow map
state_map.plot(ax=cf_ax12,color='gray',alpha=0.6,edgecolor='white',linewidth=0.5)
cf_ax12.set_xlim(-126,-108)
cf_ax12.set_ylim(31.5,53.2)
cf_ax12.set_xticks([])
cf_ax12.set_yticks([])

# stream_geo = stream_geo.sort_values( by = ywwd, ascending=True)
stream_geo.plot(ax = cf_ax12 , markersize = 200 ,cmap = cmap , norm = norm , column = ywwd , marker = "^" , edgecolor = 'black' ,linewidth = 0.8 ,legend = False)
# cbar = plt.colorbar(mappable=cbar,ax = cf_ax12, orientation='horizontal', pad = 0.04 , shrink= 0.8, aspect=8, ticks=[vmin,vmax])
# cbar.ax.tick_params(labelsize=13)

################## Worst year Capacity Factor map 
vmin, vmax, vcenter = -0.4, 0.4, 0
# vmin, vmax, vcenter = np.round(capacity_factor_deviation.iloc[:,0:3].min().min(),decimals=1),np.round(capacity_factor_deviation.iloc[:,0:3].max().max(),decimals=1) , 0
norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
cmap = 'PiYG_r'
cbar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)


state_map.plot(ax=cf_ax13,color='gray',alpha=0.6,edgecolor='white',linewidth=0.5)
cf_ax13.set_xlim(-125,-113)
cf_ax13.set_ylim(31.25,42)
cf_ax13.set_xticks([])
cf_ax13.set_yticks([])
cf_ax13.set_title("Capacity Factor", fontsize= 18,**csfont)

geo_df2=geo_df[geo_df.loc[:,worst]>-1]
geo_df2.plot(ax=cf_ax13,markersize=geo_df2['netcap'],cmap=cmap,norm=norm,column=worst ,alpha= 0.7, marker='o',edgecolor='black',linewidth=0.8,legend=False)


################## Best year Capacity Factor map 
state_map.plot(ax=cf_ax14,color='gray',alpha=0.6,edgecolor='white',linewidth=0.5)
cf_ax14.set_xlim(-125,-113)
cf_ax14.set_ylim(31.25,42)
cf_ax14.set_xticks([])
cf_ax14.set_yticks([])

geo_df3=geo_df[geo_df.loc[:,best]>-1]
geo_df3.plot(ax=cf_ax14,markersize=geo_df3['netcap'],cmap=cmap,norm=norm,column=best ,alpha= 0.7, marker='o',edgecolor='black',linewidth=0.8,legend=False)


################## Year with the worst day Capacity Factor map
state_map.plot(ax=cf_ax15,color='gray',alpha=0.6,edgecolor='white',linewidth=0.5)
cf_ax15.set_xlim(-125,-113)
cf_ax15.set_ylim(31.25,42)
cf_ax15.set_xticks([])
cf_ax15.set_yticks([])

geo_df4=geo_df[geo_df.loc[:,ywwd]>-1]
geo_df4.plot(ax=cf_ax15,markersize=geo_df4['netcap'],cmap=cmap,norm=norm,column=ywwd ,alpha= 0.7, marker='o',edgecolor='black',linewidth=0.8,legend=False)
# cbar = plt.colorbar(mappable=cbar,ax = cf_ax15, orientation='horizontal', pad = 0.04, aspect=11 , shrink= 0.8 , ticks=[vmin,vcenter,vmax])
# cbar.ax.tick_params(labelsize=13)

################# Worst Year Damage Maps  

vmin, vmax, vcenter = -0.6 , 0.6 , 0
# vmin, vmax, vcenter =  np.round(county_damage_deviation_normal.min().min(),decimals=1),  np.round(county_damage_deviation_normal.max().max(),decimals=1), 0
norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
cmap = 'PiYG_r'
cbar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)


shape.plot(ax=cf_ax16 , cmap=cmap , norm=norm , column=worst, alpha=1,edgecolor='black',linewidth=0.5)
state_map.boundary.plot(ax=cf_ax16, edgecolor='black',linewidth=1)
cf_ax16.set_xlim(-125,-109)
cf_ax16.set_ylim(30.5,43)
cf_ax16.set_xticks([])
cf_ax16.set_yticks([])
cf_ax16.set_title("Annual Damage", fontsize= 18,**csfont)
cf_ax16.set_ylabel('Highest Damage Year', labelpad = -200, fontsize= 18,**csfont)


################## Best Year Damage Maps 
shape.plot(ax=cf_ax17 , cmap=cmap , norm=norm , column=best, alpha=1,edgecolor='black',linewidth=0.5)
state_map.boundary.plot(ax=cf_ax17, edgecolor='black',linewidth=1)
cf_ax17.set_xlim(-125,-109)
cf_ax17.set_ylim(30.5,43)
cf_ax17.set_xticks([])
cf_ax17.set_yticks([])
cf_ax17.set_ylabel('Lowest Damage Year', labelpad = -200, fontsize= 18,**csfont)


################## YWWD Damage Maps 
shape.plot(ax=cf_ax18 , cmap=cmap , norm=norm , column=ywwd, alpha=1,edgecolor='black',linewidth=0.5)
state_map.boundary.plot(ax=cf_ax18, edgecolor='black',linewidth=1)
cf_ax18.set_xlim(-125,-109)
cf_ax18.set_ylim(30.5,43)
cf_ax18.set_xticks([])
cf_ax18.set_yticks([])
cf_ax18.set_ylabel('\n'.join(wrap("Year With The Highest Damage Day",20)), labelpad = -210, fontsize= 18,**csfont)
# '\n'.join(wrap("Anomaly (m/s)",
# cbar = plt.colorbar(mappable=cbar,ax = cf_ax18, orientation='horizontal', pad = 0.04 , shrink= 0.8 , aspect=11 , ticks=[vmin,vcenter,vmax])
# cbar.ax.tick_params(labelsize=13)


norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
cbar = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.PiYG_r)
cax1 = plt.axes([0.642, .060, 0.240, 0.025])
cax2 = plt.axes([0.3905, .060, 0.240, 0.025])
cax3 = plt.axes([0.139, .060, 0.240, 0.025])

cbar1 = plt.colorbar(cbar, cax= cax1 , orientation="horizontal" )#.set_ticklabels([3.13,8.58])
cbar1.set_ticklabels([])
cax1.xaxis.set_tick_params(length = 0)
cbar1.ax.tick_params(labelsize=15)


cbar2 = plt.colorbar(cbar, cax= cax2 , orientation="horizontal" )#.set_ticklabels([3.13,8.58])
cbar2.set_ticklabels([])
cax2.xaxis.set_tick_params(length = 0)
cbar2.ax.tick_params(labelsize=15)


cbar3 = plt.colorbar(cbar, cax= cax3 , orientation="horizontal" )#.set_ticklabels([3.13,8.58])
cbar3.set_ticklabels([])
cax3.xaxis.set_tick_params(length = 0)
cbar3.ax.tick_params(labelsize=15)



plt.savefig('Plots/Fig_2_maps.pdf',dpi=250)

