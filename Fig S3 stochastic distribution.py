# -*- coding: utf-8 -*-
"""
Created on Thu May 27 12:33:33 2021

@author: mzeigha
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from textwrap import wrap
import matplotlib as mpl



def process(data):

    means = np.median(data, axis= 0)
    
    ######    # colormap = cm.RdBu_r # change this for the colormap of choice
    colormap = plt.cm.RdBu_r # change this for the colormap of choice
    percentiles = [0,1,5,10,25,50,75,90,95,99,100]
    
    L = int(len(percentiles))
    
    SDist=np.zeros((365,L))
    for i in range(L):
        for t in range(365):
          SDist[t,i]=np.percentile(data[:,t],percentiles[i])
    SDist[:,5]=means
    
    half = int((len(percentiles)-1)/2)
    
    return half,percentiles,colormap,SDist

# ###############################################################################
csfont = {'fontname':'Arial'}

stochastic_df = pd.read_csv('Results/Stochastic_df.csv')
scenarios = ['Solar_Power', 'CA_Wind_Power', 'Path66_flow']
scenarios_label = ['Solar Power' , 'Wind Power', 'Imports']
avg = np.zeros(365)

stochastic_df["Day"] = pd.RangeIndex(182500)
stochastic_df["Year"] = stochastic_df["Day"]//365
stochastic_df["DOY"] = stochastic_df['Day']-stochastic_df['Year']*365

annual = stochastic_df.groupby(['Year'], as_index=False)['Solar_Power', 'CA_Wind_Power', 'Path66_flow'].sum()

#%%

fig = plt.figure(constrained_layout=True, figsize = (23,10))


gs = fig.add_gridspec(2,3, width_ratios=[1,1,1], height_ratios = [1,1],hspace=0.1,wspace=0.05)

###################### LOAD 
cf_ax1 = fig.add_subplot(gs[0,0])
cf_ax2 = fig.add_subplot(gs[1,0])

###################### HYDRO
cf_ax3 = fig.add_subplot(gs[0,1])
cf_ax4 = fig.add_subplot(gs[1,1])

cf_ax5 = fig.add_subplot(gs[0,2])
cf_ax6 = fig.add_subplot(gs[1,2])

#%%
########################## CA Solar Plots 
stochastic_df['Solar_Power'][stochastic_df['Solar_Power']<0] = 0

hist = np.array(stochastic_df['Solar_Power']/1000).reshape((500,365))    
half,percentiles,colormap,SDist = process(hist)

for v in (0,364):
    avg[v] = hist[:,v].mean()


cf_ax1.plot(np.arange(0,365,1), SDist[:,half],color='black',label='Median')
   
for i in range(len(percentiles)-1):
    
    if i == 0:
        index = "Minimum"
        
    elif i == 9:
        index = "Maximum"
        
    elif i == 1:
        index= str(percentiles[i]) + 'st percentile'

    elif i <5:
        index= str(percentiles[i]) + 'th percentile'
    else:
        index= str(percentiles[i+1]) + 'th percentile'
        
    cf_ax1.fill_between(np.arange(0,365,1), SDist[:,i],SDist[:,i+1],color=colormap(i/len(percentiles)),label= index)

cf_ax1.set_ylabel('GWh',fontsize=28,**csfont)
cf_ax1.set_visible(True)
cf_ax1.tick_params(labelsize=26)
cf_ax1.set_xticks([0,90,181,273,335])
cf_ax1.set_xticklabels(['Jan', 'April', 'July', 'Oct', 'Dec'],**csfont)
cf_ax1.set_title('CAISO Solar Generation',pad= 25,fontsize=30,**csfont)




cf_ax2.hist(annual['Solar_Power']/1000000,25,color='bisque',edgecolor='black')
cf_ax2.set_xlabel('CAISO Average Solar Generation (TWh)',fontsize=28,**csfont)
cf_ax2.set_xticks([23.5,24,24.4])
cf_ax2.set_ylabel('Number of Years',fontsize=28,**csfont)
cf_ax2.tick_params(labelsize=26)


#%%
# ########################## CA Wind Plots 
stochastic_df['CA_Wind_Power'][stochastic_df['CA_Wind_Power']<0] = 0

hist = np.array(stochastic_df['CA_Wind_Power']/1000).reshape((500,365))    
half,percentiles,colormap,SDist = process(hist)

for v in (0,364):
    avg[v] = hist[:,v].mean()



cf_ax3.plot(np.arange(0,365,1), SDist[:,half],color='black',label='Median')
   
for i in range(len(percentiles)-1):

    if i <5:
        index= str(percentiles[i]) + 'th percentile'
    else:
        index= str(percentiles[i+1]) + 'th percentile'
    cf_ax3.fill_between(np.arange(0,365,1), SDist[:,i],SDist[:,i+1],color=colormap(i/len(percentiles)),label= index)

# cf_ax3.set_xlabel('Day of the year', fontsize=14)
cf_ax3.set_ylabel('GWh',fontsize=28,**csfont)
cf_ax3.set_visible(True)
cf_ax3.tick_params(labelsize=26)
cf_ax3.set_xticks([0,90,181,273,335])
cf_ax3.set_xticklabels(['Jan', 'April', 'July', 'Oct', 'Dec'],**csfont)
cf_ax3.set_title('CAISO Wind Power Generation',pad= 25,fontsize=30,**csfont)




cf_ax4.hist(annual['CA_Wind_Power']/1000000,25,color='bisque',edgecolor='black')
cf_ax4.set_xlabel('\n'.join(wrap("CAISO Average Wind Power Generation (TWh)", 21)),fontsize=28,**csfont)
cf_ax4.set_ylabel('Number of Years',fontsize=28,**csfont)
cf_ax4.set_xticks([9.5,10,10.5])
cf_ax4.tick_params(labelsize=26)


########################## Import Plots 
stochastic_df['Path66_flow'][stochastic_df['Path66_flow']<0] = 0

hist = np.array(stochastic_df['Path66_flow']/1000).reshape((500,365))    
half,percentiles,colormap,SDist = process(hist)

for v in (0,364):
    avg[v] = hist[:,v].mean()


cf_ax5.plot(np.arange(0,365,1), SDist[:,half],color='black',label='Median')
   
for i in range(len(percentiles)-1):
    
    if i == 0:
        index = "Minimum"
        
    elif i == 9:
        index = "Maximum"
        
    elif i == 1:
        index= str(percentiles[i]) + 'st percentile'

    elif i <5:
        index= str(percentiles[i]) + 'th percentile'
    else:
        index= str(percentiles[i+1]) + 'th percentile'
        
    cf_ax5.fill_between(np.arange(0,365,1), SDist[:,i],SDist[:,i+1],color=colormap(i/len(percentiles)),label= index)

cf_ax5.set_ylabel('GWh',fontsize=28,**csfont)
cf_ax5.set_visible(True)
cf_ax5.tick_params(labelsize=26)
cf_ax5.set_xticks([0,90,181,273,335])
cf_ax5.set_xticklabels(['Jan', 'April', 'July', 'Oct', 'Dec'],**csfont)
cf_ax5.set_title('CAISO Imported Power',pad= 25,fontsize=30,**csfont)




cf_ax6.hist(annual['Path66_flow']/1000,25,color='bisque',edgecolor='black')
cf_ax6.set_xlabel('CAISO Impoerted Power (GWh)',fontsize=28,**csfont)
cf_ax6.set_ylabel('Number of Years',fontsize=28,**csfont)
cf_ax6.tick_params(labelsize=26)

######################### Legend
# handles, labels = cf_ax1.get_legend_handles_labels()
# fig.legend(handles, labels, loc='upper center', ncol=6, bbox_to_anchor=(0.52, 1.18), fancybox=False, shadow=False, prop= {'family':'Arial' , 'size':'26'})# fontsize=26,**csfont)


cmap = plt.cm.RdBu_r
bounds = [0,1,5,10,25,50,75,90,95,99,100]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

cax = plt.axes([0.30, -0.15, 0.45, 0.08])

cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
             cax=cax, orientation='horizontal',
             label="Median")
# cbar.ax.tick_params(labelsize=13,**csfont)
cax.set_xlabel("Median", fontsize=35,labelpad = 10 , **csfont)
cax.xaxis.set_ticks_position('top')
cax.set_xticks( [0.5,3, 7.5, 15, 37.5, 50 ,62.5, 82.5, 92.5,97, 99.5],['Min', 1,5,10,25,"",75,90,95, 99, 'Max'], fontsize=35, **csfont)


plt.savefig('Plots/Fig_SI_Distribution.png' , bbox_inches='tight',dpi=250)

#%%
# plt.plot(range(365),stochastic_df[stochastic_df['Year']==100]['Path66_flow'])
# daily = stochastic_df.groupby(['DOY'], as_index=False)['Path66_flow'].mean()
# plt.plot(range(365),daily)
# plt.savefig('Plots/Fig_SI_ywwd_year.png' , bbox_inches='tight',dpi=250)
