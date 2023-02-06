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

load = pd.read_csv('CAISO/Load.csv')
hydro = pd.read_csv('CAISO/Hydro.csv')
price = pd.read_csv('CAISO/LMP.csv')


hydro['UTC time'] = pd.to_datetime(hydro['Local time'])

hydro.index = hydro['UTC time']

daily = hydro.Adjusted_WAT_Gen.resample('D',level=0).sum()

hydro_2019 = daily.loc["2019-01-01" :"2019-12-31" ]

price['datetime'] = pd.to_datetime(price['datetime'])

price.index = price['datetime']


daily_price = price.resample('D',level=0)['PGAE_DAM_LMP', 'SCE_DAM_LMP', 'SDGE_DAM_LMP'].mean().mean(axis= 1)
#%%
load['day'] = load['Unnamed: 0']//24

load_2019 = load.groupby(['day'])['CISO'].sum()



#%%
csfont = {'fontname':'Arial'}

stochastic_df = pd.read_csv('Results/Stochastic_df.csv')
scenarios = ['CA_load' , 'CA_Hydropower']
scenarios_label = ['CA Load' , 'CA Hydropower generaton']
avg = np.zeros(365)

stochastic_df["Day"] = pd.RangeIndex(182500)
stochastic_df["Year"] = stochastic_df["Day"]//365
stochastic_df["DOY"] = stochastic_df['Day']-stochastic_df['Year']*365

annual = stochastic_df.groupby(['Year'], as_index=False)['CA_load', 'CA_Hydropower'].sum()

#%%

fig = plt.figure(constrained_layout=True, figsize = (30,10))


gs = fig.add_gridspec(2,3, width_ratios=[1,1,1], height_ratios = [1,1],hspace=0.1,wspace=0.05)

###################### LOAD 
cf_ax1 = fig.add_subplot(gs[0,0])
cf_ax2 = fig.add_subplot(gs[1,0])

###################### HYDRO
cf_ax3 = fig.add_subplot(gs[0,1])
cf_ax4 = fig.add_subplot(gs[1,1])

cf_ax5 = fig.add_subplot(gs[0,2])
cf_ax6 = fig.add_subplot(gs[1,2])

# cf_ax7 = fig.add_subplot(gs[0,3])
# cf_ax8 = fig.add_subplot(gs[1,3])
########################## CA load Plots 
stochastic_df['CA_load'][stochastic_df['CA_load']<0] = 0

hist = np.array(stochastic_df['CA_load']/1000).reshape((500,365))    
half,percentiles,colormap,SDist = process(hist)

for v in (0,364):
    avg[v] = hist[:,v].mean()


cf_ax1.plot(np.arange(0,365,1), load_2019//1000 ,color='black',label='2019')
   
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

cf_ax1.set_ylabel('GWh',fontsize=30,**csfont)
cf_ax1.set_visible(True)
cf_ax1.tick_params(labelsize=28)
cf_ax1.set_xticks([0,90,181,273,335])
cf_ax1.set_xticklabels(['Jan', 'April', 'July', 'Oct', 'Dec'],**csfont)
cf_ax1.set_title('CAISO Electricity Demand',pad= 25,fontsize=32,**csfont)




cf_ax2.hist(annual['CA_load']/1000000,25,color='bisque',edgecolor='black')
cf_ax2.set_xlabel('CAISO Annual Demand (TWh)',fontsize=30,**csfont)
cf_ax2.set_ylabel('Number of Years',labelpad=30,fontsize=32,**csfont)
cf_ax2.tick_params(labelsize=28)



# ########################## CA Hydro Plots 
stochastic_df['CA_Hydropower'][stochastic_df['CA_Hydropower']<0] = 0

hist = np.array(stochastic_df['CA_Hydropower']/1000).reshape((500,365))    
half,percentiles,colormap,SDist = process(hist)

for v in (0,364):
    avg[v] = hist[:,v].mean()



cf_ax3.plot(np.arange(0,365,1), hydro_2019/1000 ,color='black',label='Median')
   
for i in range(len(percentiles)-1):

    if i <5:
        index= str(percentiles[i]) + 'th percentile'
    else:
        index= str(percentiles[i+1]) + 'th percentile'
    cf_ax3.fill_between(np.arange(0,365,1), SDist[:,i],SDist[:,i+1],color=colormap(i/len(percentiles)),label= index)

# cf_ax3.set_xlabel('Day of the year', fontsize=14)
cf_ax3.set_ylabel('GWh',fontsize=30,**csfont)
cf_ax3.set_visible(True)
cf_ax3.tick_params(labelsize=28)
cf_ax3.set_xticks([0,90,181,273,335])
cf_ax3.set_xticklabels(['Jan', 'April', 'July', 'Oct', 'Dec'],**csfont)
cf_ax3.set_title('CAISO Hydropower Generation',pad= 25,fontsize=32,**csfont)




cf_ax4.hist(annual['CA_Hydropower']/1000000,25,color='bisque',edgecolor='black')
cf_ax4.set_xlabel('\n'.join(wrap("CAISO Annual Hydropower Generation (TWh)", 21)),fontsize=30,**csfont)
cf_ax4.set_ylabel('Number of Years',labelpad=20,fontsize=30,**csfont)
# cf_ax4.set_xticks([50,70,90])
cf_ax4.tick_params(labelsize=28)


######################### Legend

Damages = pd.read_csv('Results/SNP_Damage.csv')
shadow_price = pd.read_csv('Results/Shadow_price.csv')

# scenarios = ['CA_load' , 'CA_Hydropower']
# scenarios_label = ['CA Load' , 'CA Hydropower generaton']
# avg = np.zeros(365)

Day = pd.DataFrame([])
Day['Day'] = pd.RangeIndex(182500)
Damages["Year"] = Day["Day"]//365
shadow_price["Year"] = Day["Day"]//365

annual_Damages = Damages.groupby(['Year'], as_index=True)['no_tax'].sum()
annual_shadow_price = shadow_price.groupby(['Year'], as_index=True)['no_tax'].mean()

avg = np.zeros(365)

# fig = plt.figure(constrained_layout=True, figsize = (15,10))

# gs = fig.add_gridspec(2,2, width_ratios=[1,1], height_ratios = [1,1], wspace=.05, hspace=0.05)

# ###################### Market Price 
# cf_ax1 = fig.add_subplot(gs[0,0])
# cf_ax2 = fig.add_subplot(gs[1,0])

# ###################### Damages 
# cf_ax3 = fig.add_subplot(gs[0,1])
# cf_ax4 = fig.add_subplot(gs[1,1])


##########################  Market Price 
hist = np.array(shadow_price['no_tax']).reshape((500,365))    
half,percentiles,colormap,SDist = process(hist)

for v in (0,364):
    avg[v] = hist[:,v].mean()



cf_ax5.plot(np.arange(0,365,1), daily_price ,color='black',label='Median')
   
for i in range(len(percentiles)-2):
    
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

# cf_ax5.set_xlabel('Day of the year', fontsize=14,fontweight='bold')
cf_ax5.set_ylabel('$/MWh',fontsize=30,**csfont)
cf_ax5.set_visible(True)
cf_ax5.tick_params(labelsize=28)
cf_ax5.set_xticks([0,90,181,273,335])
cf_ax5.set_xticklabels(['Jan', 'April', 'July', 'Oct', 'Dec'],**csfont)
# cf_ax5.set_yscale( 'log')
cf_ax5.set_title('CAISO Electricity Market Price',pad= 25,fontsize=32,**csfont)

# plt.savefig('Plots/Local Damages Distribution of {} scenario.png'.format(sen) , bbox_inches='tight',dpi=250)




cf_ax6.hist(annual_shadow_price,25,color='bisque',edgecolor='black')
cf_ax6.set_xlabel('Annual Average Price($/MWh)',fontsize=30,**csfont)
cf_ax6.set_ylabel('Number of Years',labelpad=20,fontsize=30,**csfont)
cf_ax6.tick_params(labelsize=28)


# ########################## Damages
# Damages['no_tax'][Damages['no_tax']<0] = 0

# hist = np.array(Damages['no_tax']/1000000).reshape((500,365))    
# half,percentiles,colormap,SDist = process(hist)

# for v in (0,364):
#     avg[v] = hist[:,v].mean()


# cf_ax7.plot(np.arange(0,365,1), SDist[:,half],color='black',label='Median')
   
# for i in range(len(percentiles)-1):

#     if i <5:
#         index= str(percentiles[i]) + 'th percentile'
#     else:
#         index= str(percentiles[i+1]) + 'th percentile'
#     cf_ax7.fill_between(np.arange(0,365,1), SDist[:,i],SDist[:,i+1],color=colormap(i/len(percentiles)),label= index)

# # cf_ax7.set_xlabel('Day of the year', fontsize=14,fontweight='bold')
# cf_ax7.set_ylabel('$M',fontsize=30,**csfont)
# cf_ax7.tick_params(labelsize=28)
# cf_ax7.set_xticks([0,90,181,273,335])
# cf_ax7.set_xticklabels(['Jan', 'April', 'July', 'Oct', 'Dec'],**csfont)
# cf_ax7.set_title('Local Air Damages',pad= 25,fontsize=32,**csfont)




# cf_ax8.hist(annual_Damages/1000000,25,color='bisque',edgecolor='black')
# cf_ax8.set_xlabel('Annual Damages ($M)',fontsize=30,**csfont)
# cf_ax8.set_ylabel('Number of Years',labelpad=10,fontsize=30,**csfont)
# cf_ax8.tick_params(labelsize=28)

# ######################### Legend
# handles, labels = cf_ax1.get_legend_handles_labels()
# fig.legend(handles, labels, loc='upper center', ncol=6, bbox_to_anchor=(0.52, 1.18), fancybox=False, shadow=False, prop= {'family':'Arial' , 'size':'32'})# fontsize=26,**csfont)

cmap = plt.cm.RdBu_r
bounds = [0,1,5,10,25,50,75,90,95,99,100]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

cax = plt.axes([0.30, -0.15, 0.45, 0.08])

cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
             cax=cax, orientation='horizontal',
             label="2019")
# cbar.ax.tick_params(labelsize=13,**csfont)
cax.set_xlabel("2019", fontsize=35,labelpad = 10 , **csfont)
cax.xaxis.set_ticks_position('top')
cax.set_xticks( [0.5,3, 7.5, 15, 37.5, 50 ,62.5, 82.5, 92.5,97, 99.5],['Min', 1,5,10,25,"",75,90,95, 99, 'Max'], fontsize=35, **csfont)

plt.savefig('Plots/Review_Distribution.png' , bbox_inches='tight',dpi=250)

