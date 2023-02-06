# -*- coding: utf-8 -*-
"""
Created on Tue May 18 00:33:29 2021

@author: mzeigha
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats 
from textwrap import wrap

stream = pd.read_csv('Results/Average values/stream_annual.csv')
wind = pd.read_csv('Results/Average values/wind_annual.csv')
temp = pd.read_csv('Results/Average values/temp_annual.csv')
irradiance = pd.read_csv('Results/Average values/irradiance_annual.csv')



best = 495
worst = 151
ywwd = 100
ywld = 81

day = pd.DataFrame([])
day['day'] = pd.RangeIndex(365)

stream_yavg = stream.groupby(['DOY'])['avg'].mean()

wind_yavg = wind.groupby(['DOY'])['avg'].mean()

temp_yavg = temp.groupby(['DOY'])['avg'].mean()

irradiance_yavg = irradiance.groupby(['DOY'])['avg'].mean()

cooling = temp['avg'] - 18.33
cooling_yavg = temp_yavg-18.33

cooling [cooling<0] = 0
cooling_yavg [cooling_yavg<0] = 0


cooling = pd.concat ( [cooling ,temp.loc[:,'Year']], axis = 1)

#%%
csfont = {'fontname':'Arial'}
fig = plt.figure(figsize = (9,7))
#constrained_layout=True, 

gs = fig.add_gridspec(4,2, height_ratios = [1,1,1,1], width_ratios=[1,2], wspace=0.30, hspace=0.5)

############# stream 
cf_ax1 = fig.add_subplot(gs[0,0])
cf_ax2 = fig.add_subplot(gs[0,1])

############# wind 
cf_ax3 = fig.add_subplot(gs[1,0])
cf_ax4 = fig.add_subplot(gs[1,1])


############# Temp 
cf_ax5 = fig.add_subplot(gs[2,0])
cf_ax6 = fig.add_subplot(gs[2,1])


############# Irradiance 
cf_ax7 = fig.add_subplot(gs[3,0])
cf_ax8 = fig.add_subplot(gs[3,1])



cf_ax1.plot(day['day'] , wind_yavg, color='black', lw=0.7)
cf_ax1.set_ylabel('\n'.join(wrap("Wind Speed (m/s)", 12)),labelpad=2, fontsize=12,**csfont)
cf_ax1.set_xticks([0,90,181,273])
cf_ax1.set_xticklabels(['Jan', 'April', 'July', 'Oct'],fontsize=11,**csfont)
# cf_ax1.set_title('Average',fontsize=14)

# cf_ax2.set_title('Anomaly',fontsize=14)
cf_ax2.plot(day['day'] , -(wind_yavg-wind[wind['Year'] == best].reset_index()['avg']),linewidth=1, color='royalblue')
cf_ax2.plot(day['day'] , -(wind_yavg-wind[wind['Year'] == worst].reset_index()['avg']),linewidth=1, color='maroon')
cf_ax2.plot(day['day'] , -(wind_yavg-wind[wind['Year'] == ywwd].reset_index()['avg']),linewidth=1, color='goldenrod')
cf_ax2.plot(day['day'] , (wind_yavg-wind_yavg), color='black', linewidth=0.5)
cf_ax2.set_xticks([0,90,181,273])
cf_ax2.set_xticklabels(['Jan', 'April', 'July', 'Oct'],fontsize=11,**csfont)
cf_ax2.set_ylabel('\n'.join(wrap("Anomaly (m/s)", 12)),labelpad=5,fontsize=12,**csfont)
# cf_ax2.legend(['Best','Worst','YWWD'], loc=1)



# cf_ax3.plot(day['day'] , temp_yavg, color='black', lw=0.7)
# cf_ax3.set_ylabel("Temperature",fontsize=12)
# cf_ax3.set_xticks([0,90,181,273,365])
# cf_ax3.set_xticklabels(['Jan', 'April', 'July', 'Oct', 'Dec'],fontsize=11)
# cf_ax4.plot(day['day'] , -(temp_yavg - temp[temp['Year'] == best].reset_index()['avg']), color='royalblue',linewidth=1)
# cf_ax4.plot(day['day'] , -(temp_yavg - temp[temp['Year'] == worst].reset_index()['avg']), color='maroon',linewidth=1)
# cf_ax4.plot(day['day'] , -(temp_yavg - temp[temp['Year'] == ywwd].reset_index()['avg']), color='goldenrod',linewidth=1)
# cf_ax4.plot(day['day'] , (wind_yavg-wind_yavg), color='black', linewidth=0.5)
# # cf_ax4.legend(['Best','Worst','YWWD'], loc=1)
# cf_ax4.set_xticks([0,90,181,273,365])
# cf_ax4.set_xticklabels(['Jan', 'April', 'July', 'Oct', 'Dec'],fontsize=11)



cf_ax3.plot(day['day'] , cooling_yavg, color='black', lw=0.7)
cf_ax3.set_ylabel('\n'.join(wrap("Cooling Degrees (C)", 12)),labelpad=13,fontsize=12,**csfont)
cf_ax3.set_xticks([0,90,181,273])
cf_ax3.set_xticklabels(['Jan', 'April', 'July', 'Oct'],fontsize=11,**csfont)
cf_ax4.plot(day['day'] , -(cooling_yavg - cooling[cooling['Year'] == best].reset_index()['avg']), color='royalblue',linewidth=1)
cf_ax4.plot(day['day'] , -(cooling_yavg - cooling[cooling['Year'] == worst].reset_index()['avg']), color="maroon",linewidth=1)
cf_ax4.plot(day['day'] , -(cooling_yavg - cooling[cooling['Year'] == ywwd].reset_index()['avg']), color='goldenrod',linewidth=1)
cf_ax4.plot(day['day'] , (wind_yavg-wind_yavg), color='black', linewidth=0.5)
# cf_ax4.legend(['Best','Worst','YWWD'], loc=1)
cf_ax4.set_xticks([0,90,181,273])
cf_ax4.set_xticklabels(['Jan', 'April', 'July', 'Oct'],fontsize=11,**csfont)
cf_ax4.set_ylabel('\n'.join(wrap("Anomaly (C)", 12)),labelpad=15,fontsize=12,**csfont)



cf_ax5.plot(day['day'] , irradiance_yavg/1000, color='black', lw=0.7)
cf_ax5.set_ylabel('\n'.join(wrap("Irradiance (kW/m²)", 12)),labelpad=2,fontsize=12,**csfont)
cf_ax5.set_xticks([0,90,181,273])
cf_ax5.set_xticklabels(['Jan', 'April', 'July', 'Oct'],fontsize=11,**csfont)
cf_ax6.plot(day['day'] , (-(irradiance_yavg - irradiance[irradiance['Year'] == best].reset_index()['avg']))/1000, color='royalblue',linewidth=1)
cf_ax6.plot(day['day'] , (-(irradiance_yavg - irradiance[irradiance['Year'] == worst].reset_index()['avg']))/1000, color='maroon',linewidth=1)
cf_ax6.plot(day['day'] , (-(irradiance_yavg - irradiance[irradiance['Year'] == ywwd].reset_index()['avg']))/1000, color='goldenrod',linewidth=1)
cf_ax6.plot(day['day'] , (wind_yavg-wind_yavg), color='black', linewidth=0.5)
# cf_ax6.legend(['Best','Worst','YWWD'], loc=1)
cf_ax6.set_xticks([0,90,181,273])
cf_ax6.set_yticks([-2,0,1])
cf_ax6.set_xticklabels(['Jan', 'April', 'July', 'Oct'],fontsize=11,**csfont)
cf_ax6.set_ylabel('\n'.join(wrap("Anomaly (kW/m²)", 12)),labelpad=2,fontsize=12,**csfont)


cf_ax7.plot(day['day'] , stream_yavg/1000, color='black', lw=0.7)
cf_ax7.set_ylabel('\n'.join(wrap("Streamflow (10³ m³/s)", 12)),fontsize=12,**csfont)
cf_ax7.set_xticks([0,90,181,273])
cf_ax7.set_xticklabels(['Jan', 'April', 'July', 'Oct'],fontsize=11,**csfont)
cf_ax8.plot(day['day'] , (-(stream_yavg - stream[stream['Year'] == best].reset_index()['avg']))/1000, color='royalblue',linewidth=1)
cf_ax8.plot(day['day'] , (-(stream_yavg - stream[stream['Year'] == worst].reset_index()['avg']))/1000, color='maroon',linewidth=1)
cf_ax8.plot(day['day'] , (-(stream_yavg - stream[stream['Year'] == ywwd].reset_index()['avg']))/1000, color='goldenrod',linewidth=1)
cf_ax8.plot(day['day'] , (wind_yavg-wind_yavg), color='black', linewidth=0.5)
cf_ax8.set_xticks([0,90,181,273])
cf_ax8.set_yticks([-10,0,20])
cf_ax8.set_xticklabels(['Jan', 'April', 'July', 'Oct'],fontsize=11,**csfont)
cf_ax8.set_ylabel('\n'.join(wrap("Anomaly (10³m³/s)", 12)),labelpad=-2,fontsize=12,**csfont)


fig.legend(['Average', 'Lowest Damage year','Highest Damage year','Year With The Highest Damage Day'], loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.01),fontsize=12, prop= {'family':'Arial' , 'size':'12'})
#, fancybox=True, shadow=None
plt.savefig('Plots/Fig_4_Anomaly.pdf',bbox_inches='tight', dpi=250)
plt.show()
