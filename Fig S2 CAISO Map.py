# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 13:19:55 2021

@author: mzeigha
"""

import matplotlib.pyplot as plt
import geopandas as gpd
import cartopy.crs as ccrs
import cartopy


CAISO = gpd.read_file('Results/mapping/region2.shp').to_crs('EPSG:4326')
MID_C = gpd.read_file('Results/mapping/region1.shp').to_crs('EPSG:4326')
Columbia_basin = gpd.read_file('Results/mapping/Basins/Columbia_basin.shp').to_crs('EPSG:4326')
Sacramento = gpd.read_file('Results/mapping/Basins/CA/Sacramento/WBDHU4.shp').to_crs('EPSG:4326')
San_Joaquin = gpd.read_file('Results/mapping/Basins/CA/SanJoaquin/WBDHU4.shp').to_crs('EPSG:4326')
Tulare = gpd.read_file('Results/mapping/Basins/CA/Tulare/WBDHU4.shp').to_crs('EPSG:4326')
# state_map = gpd.read_file('Crosswalk/USA_Counties/USA_States_Generalized.shp').to_crs('EPSG:4326')



plt.figure(figsize = (14,10))
ax = plt.axes(projection=ccrs.PlateCarree())
# ax.coastlines(resolution='110m', color='black')
ax.add_feature(cartopy.feature.OCEAN)
# ax.add_feature(cartopy.feature.LAND,edgecolor='b',facecolor="#d8dcd6",alpha=0.5)
# ax.add_feature(cartopy.feature.RIVERS)
ax.add_feature(cartopy.feature.STATES, edgecolor='gray' )



CAISO.plot(ax = ax, facecolor='#ffa756',edgecolor='#ffa756', alpha=0.5)
MID_C.plot(ax = ax,facecolor='#6fc276',edgecolor='#6fc276', alpha=0.5)
Columbia_basin.boundary.plot(ax = ax,edgecolor='firebrick',linewidth=2, linestyle='dashed')
Sacramento.boundary.plot(ax = ax,edgecolor='black', linestyle='dashed')
San_Joaquin.boundary.plot(ax = ax,edgecolor='black', linestyle='dashed')
Tulare.boundary.plot( ax = ax , edgecolor='black', linestyle='dashed')
# state_map.boundary.plot(ax=ax, edgecolor='black',linewidth=0.5)



ax.set_xlim(-125, -107)
ax.set_ylim(31, 53)

ax.set_xticks([])
ax.set_yticks([])


# plt.savefig('Plots/CAISO_Map.png' , bbox_inches='tight',dpi=250)