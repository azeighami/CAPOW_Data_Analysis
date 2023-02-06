# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 18:20:11 2021

@author: mzeigha
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from textwrap import wrap
import matplotlib as mpl

csfont = {'fontname':'Arial'}

damage1 = pd.read_csv('Results/SNP_damage.csv')

damage1['Day'] = pd.RangeIndex(182500)
damage1["Year"] = damage1["Day"]//365
damage1['DOY'] = damage1['Day']-damage1['Year']*365

shadow1 = pd.read_csv('Results/Shadow_price.csv')

shadow1['Day'] = pd.RangeIndex(182500)
shadow1["Year"] = shadow1["Day"]//365
shadow1['DOY'] = shadow1['Day']-shadow1['Year']*365


shadow1['no_tax'].mean()-shadow1['SNP'].mean()/shadow1['no_tax'].mean()

damage = damage1.groupby(['DOY'])['all_tax', 'no_tax', 'CO2', 'SNP'].mean()/1000000

shadow = shadow1.groupby(['DOY'])['all_tax', 'no_tax', 'CO2', 'SNP'].mean()

day = pd.DataFrame([])
day['day'] = pd.RangeIndex(365)


fig = plt.figure(constrained_layout=True, figsize = (20,7))


gs = fig.add_gridspec(1,2, width_ratios=[1,1], height_ratios = [1],hspace=0,wspace=0.02)

###################### LOAD 
cf_ax1 = fig.add_subplot(gs[0,0])
cf_ax2 = fig.add_subplot(gs[0,1])

cf_ax1.plot(day['day'] , damage['no_tax'],color='maroon', lw=2, label = "No Tax")
cf_ax1.plot(day['day'] , damage['SNP'],color='royalblue', lw=2, label = "SNP Tax")
cf_ax1.plot(day['day'] , damage['CO2'],color='lightseagreen', lw=2, label = "CO2 Tax")
cf_ax1.plot(day['day'] , damage['all_tax'], color='goldenrod',lw=2, label = "All Tax")


cf_ax1.set_ylabel('($M)',labelpad= 15, fontsize=32,**csfont)
cf_ax1.set_visible(True)
cf_ax1.tick_params(labelsize=24)
cf_ax1.set_xlim(0,364)
cf_ax1.set_ylim(0,3.5)
cf_ax1.set_xticks([0,90,181,273])
cf_ax1.set_xticklabels(['Jan', 'April', 'July', 'Oct'],fontsize=28, **csfont)
cf_ax1.set_yticks([1,2,3])
cf_ax1.set_title('Daily Damages',pad= 25,fontsize=30,**csfont)



cf_ax2.plot(day['day'] , shadow['no_tax'],color='maroon', lw=2, label = "No Tax")
cf_ax2.plot(day['day'] , shadow['SNP'],color='royalblue', lw=2, label = "Local emissions Tax")
cf_ax2.plot(day['day'] , shadow['CO2'],color='lightseagreen', lw=2, label = "CO2 Tax")
cf_ax2.plot(day['day'] , shadow['all_tax'], color='goldenrod',lw=2, label = "CO2 and Local Emissions Tax")


cf_ax2.set_ylabel('($)',labelpad= 15, fontsize=32,**csfont)
cf_ax2.set_visible(True)
cf_ax2.tick_params(labelsize=24)
cf_ax2.set_xlim(0,364)
cf_ax2.set_xticks([0,90,181,273])
cf_ax2.set_xticklabels(['Jan', 'April', 'July', 'Oct'],fontsize=28,**csfont)
cf_ax2.set_yticks([30,50,70,90])
cf_ax2.set_title('Daily Market Price',pad= 25,fontsize=30,**csfont)

handles, labels = cf_ax2.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=4, bbox_to_anchor=(0.52, 1.14), fancybox=False, shadow=False, prop= {'family':'Arial' , 'size':'26'})# fontsize=26,**csfont)


# plt.show()
plt.savefig('Plots/Fig_SI_Average_Distribution.png' , bbox_inches='tight',dpi=250)





#%%

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

avg = np.zeros(365)
annual = damage1.groupby(['Year'], as_index=False)['no_tax' , 'CO2', 'SNP', 'all_tax'].sum()

#%%

fig = plt.figure(constrained_layout=True, figsize = (30,10))


gs = fig.add_gridspec(2,4, width_ratios=[1,1,1,1], height_ratios = [1,1],hspace=0.1,wspace=0.05)

###################### LOAD 
cf_ax1 = fig.add_subplot(gs[0,0])
cf_ax2 = fig.add_subplot(gs[1,0])

###################### HYDRO
cf_ax3 = fig.add_subplot(gs[0,1])
cf_ax4 = fig.add_subplot(gs[1,1])

cf_ax5 = fig.add_subplot(gs[0,2])
cf_ax6 = fig.add_subplot(gs[1,2])

cf_ax7 = fig.add_subplot(gs[0,3])
cf_ax8 = fig.add_subplot(gs[1,3])


########################## No Tax 

hist = np.array(damage1['no_tax']/1000000).reshape((500,365))    
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

cf_ax1.set_ylabel('($M)',labelpad=30,fontsize=28,**csfont)
cf_ax1.set_visible(True)
cf_ax1.tick_params(labelsize=26)
cf_ax1.set_xticks([0,90,181,273,335])
cf_ax1.set_xticklabels(['Jan', 'April', 'July', 'Oct', 'Dec'],**csfont)
cf_ax1.set_title('No Tax',pad= 25,fontsize=30,**csfont)




cf_ax2.hist(annual['no_tax']/1000000,25,color='bisque',edgecolor='black')
cf_ax2.set_xlabel('Annual Damages ($M)',fontsize=28,**csfont)
cf_ax2.set_ylabel('Number of Years',labelpad=10,fontsize=28,**csfont)
cf_ax2.tick_params(labelsize=26)



# ########################## CO2 Tax  

hist = np.array(damage1['CO2']/1000000).reshape((500,365))    
half,percentiles,colormap,SDist = process(hist)

for v in (0,364):
    avg[v] = hist[:,v].mean()


cf_ax3.plot(np.arange(0,365,1), SDist[:,half],color='black',label='Median')
   
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
        
    cf_ax3.fill_between(np.arange(0,365,1), SDist[:,i],SDist[:,i+1],color=colormap(i/len(percentiles)),label= index)

# cf_ax3.set_ylabel('GWh',fontsize=28,**csfont)
cf_ax3.set_visible(True)
cf_ax3.tick_params(labelsize=26)
cf_ax3.set_xticks([0,90,181,273,335])
cf_ax3.set_xticklabels(['Jan', 'April', 'July', 'Oct', 'Dec'],**csfont)
cf_ax3.set_title('CO2 Tax',pad= 25,fontsize=30,**csfont)




cf_ax4.hist(annual['CO2']/1000000,25,color='bisque',edgecolor='black')
cf_ax4.set_xlabel('Annual Damages ($M)',fontsize=28,**csfont)
cf_ax4.set_xticks([900,1000])
# cf_ax4.set_ylabel('Number of Years',labelpad=30,fontsize=28,**csfont)
cf_ax4.tick_params(labelsize=26)


# ########################## All Tax  
hist = np.array(damage1['SNP']/1000000).reshape((500,365))    
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

# cf_ax5.set_ylabel('GWh',fontsize=28,**csfont)
cf_ax5.set_visible(True)
cf_ax5.tick_params(labelsize=26)
cf_ax5.set_xticks([0,90,181,273,335])
cf_ax5.set_xticklabels(['Jan', 'April', 'July', 'Oct', 'Dec'],**csfont)
cf_ax5.set_title('Local Emissions Tax',pad= 25,fontsize=30,**csfont)




cf_ax6.hist(annual['SNP']/1000000,25,color='bisque',edgecolor='black')
cf_ax6.set_xlabel('Annual Damages ($M)',fontsize=28,**csfont)
# cf_ax6.set_ylabel('Number of Years',labelpad=30,fontsize=28,**csfont)
cf_ax6.tick_params(labelsize=26)

# ########################## SNP Tax  
hist = np.array(damage1['all_tax']/1000000).reshape((500,365))    
half,percentiles,colormap,SDist = process(hist)

for v in (0,364):
    avg[v] = hist[:,v].mean()


cf_ax7.plot(np.arange(0,365,1), SDist[:,half],color='black',label='Median')
   
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
        
    cf_ax7.fill_between(np.arange(0,365,1), SDist[:,i],SDist[:,i+1],color=colormap(i/len(percentiles)),label= index)

# cf_ax7.set_ylabel('GWh',fontsize=28,**csfont)
cf_ax7.set_visible(True)
cf_ax7.tick_params(labelsize=26)
cf_ax7.set_xticks([0,90,181,273,335])
cf_ax7.set_xticklabels(['Jan', 'April', 'July', 'Oct', 'Dec'],**csfont)
cf_ax7.set_title('CO2 and Local Emissions Tax',pad= 25,fontsize=30,**csfont)




cf_ax8.hist(annual['all_tax']/1000000,25,color='bisque',edgecolor='black')
cf_ax8.set_xlabel('Annual Damages ($M)',fontsize=28,**csfont)
# cf_ax8.set_ylabel('Number of Years',labelpad=30,fontsize=28,**csfont)
cf_ax8.tick_params(labelsize=26)


######################### Legend
# handles, labels = cf_ax1.get_legend_handles_labels()
# fig.legend(handles, labels, loc='upper center', ncol=6, bbox_to_anchor=(0.52, 1.18), fancybox=False, shadow=False, prop= {'family':'Arial' , 'size':'30'})# fontsize=26,**csfont)


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


plt.savefig('Plots/Fig_SI_Damages_Distribution.png' , bbox_inches='tight',dpi=250)

######################################################## Market Price 
#%%

avg = np.zeros(365)
annual = shadow1.groupby(['Year'], as_index=False)['no_tax' , 'CO2', 'SNP', 'all_tax'].mean()

fig = plt.figure(constrained_layout=True, figsize = (30,10))


gs = fig.add_gridspec(2,4, width_ratios=[1,1,1,1], height_ratios = [1,1],hspace=0.1,wspace=0.05)

###################### LOAD 
cf_ax1 = fig.add_subplot(gs[0,0])
cf_ax2 = fig.add_subplot(gs[1,0])

###################### HYDRO
cf_ax3 = fig.add_subplot(gs[0,1])
cf_ax4 = fig.add_subplot(gs[1,1])

cf_ax5 = fig.add_subplot(gs[0,2])
cf_ax6 = fig.add_subplot(gs[1,2])

cf_ax7 = fig.add_subplot(gs[0,3])
cf_ax8 = fig.add_subplot(gs[1,3])


########################## No Tax 
shadow1['no_tax'][shadow1['no_tax']<0] = 0

hist = np.array(shadow1['no_tax']).reshape((500,365))    
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

cf_ax1.set_ylabel('Market Price ($)',labelpad=0,fontsize=28,**csfont)
cf_ax1.set_visible(True)
cf_ax1.tick_params(labelsize=26)
cf_ax1.set_xticks([0,90,181,273,335])
cf_ax1.set_xticklabels(['Jan', 'April', 'July', 'Oct', 'Dec'],**csfont)
cf_ax1.set_title('No Tax',pad= 10,fontsize=30,**csfont)
cf_ax1.set_ylim(0,1000)



cf_ax2.hist(annual['no_tax'],25,color='bisque',edgecolor='black')
cf_ax2.set_xlabel('Annual Average Market price ($)',fontsize=28,**csfont)
cf_ax2.set_ylabel('Number of Years',labelpad=30,fontsize=28,**csfont)
cf_ax2.tick_params(labelsize=26)



# ########################## CO2 Tax  
shadow1['CO2'][shadow1['CO2']<0] = 0
hist = np.array(shadow1['CO2']).reshape((500,365))    
half,percentiles,colormap,SDist = process(hist)

for v in (0,364):
    avg[v] = hist[:,v].mean()


cf_ax3.plot(np.arange(0,365,1), SDist[:,half],color='black',label='Median')
   
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
        
    cf_ax3.fill_between(np.arange(0,365,1), SDist[:,i],SDist[:,i+1],color=colormap(i/len(percentiles)),label= index)

# cf_ax3.set_ylabel('GWh',fontsize=28,**csfont)
cf_ax3.set_visible(True)
cf_ax3.tick_params(labelsize=26)
cf_ax3.set_xticks([0,90,181,273,335])
cf_ax3.set_xticklabels(['Jan', 'April', 'July', 'Oct', 'Dec'],**csfont)
cf_ax3.set_title('CO2 Tax',pad= 25,fontsize=30,**csfont)
cf_ax3.set_ylim(0,1000)



cf_ax4.hist(annual['CO2'],25,color='bisque',edgecolor='black')
cf_ax4.set_xlabel('Annual Average Market price ($)',fontsize=28,**csfont)
# cf_ax4.set_xticks([900,1000])
# cf_ax4.set_ylabel('Number of Years',labelpad=30,fontsize=28,**csfont)
cf_ax4.tick_params(labelsize=26)


# ########################## All Tax 
shadow1['SNP'][shadow1['SNP']<0] = 0 
hist = np.array(shadow1['SNP']).reshape((500,365))    
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

# cf_ax5.set_ylabel('GWh',fontsize=28,**csfont)
cf_ax5.set_visible(True)
cf_ax5.tick_params(labelsize=26)
cf_ax5.set_xticks([0,90,181,273,335])
cf_ax5.set_xticklabels(['Jan', 'April', 'July', 'Oct', 'Dec'],**csfont)
cf_ax5.set_title('Local Emissions Tax',pad= 25,fontsize=30,**csfont)
cf_ax5.set_ylim(0,1000)



cf_ax6.hist(annual['SNP'],25,color='bisque',edgecolor='black')
cf_ax6.set_xlabel('Annual Average Market price ($)',fontsize=28,**csfont)
# cf_ax6.set_ylabel('Number of Years',labelpad=30,fontsize=28,**csfont)
cf_ax6.tick_params(labelsize=26)

# ########################## SNP Tax  
shadow1['all_tax'][shadow1['all_tax']<0] = 0
hist = np.array(shadow1['all_tax']).reshape((500,365))    
half,percentiles,colormap,SDist = process(hist)

for v in (0,364):
    avg[v] = hist[:,v].mean()


cf_ax7.plot(np.arange(0,365,1), SDist[:,half],color='black',label='Median')
   
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
        
    cf_ax7.fill_between(np.arange(0,365,1), SDist[:,i],SDist[:,i+1],color=colormap(i/len(percentiles)),label= index)

# cf_ax7.set_ylabel('GWh',fontsize=28,**csfont)
cf_ax7.set_visible(True)
cf_ax7.tick_params(labelsize=26)
cf_ax7.set_xticks([0,90,181,273,335])
cf_ax7.set_xticklabels(['Jan', 'April', 'July', 'Oct', 'Dec'],**csfont)
cf_ax7.set_title('CO2 and Local Emissions Tax',pad= 25,fontsize=30,**csfont)
cf_ax7.set_ylim(0,1000)



cf_ax8.hist(annual['all_tax'],25,color='bisque',edgecolor='black')
cf_ax8.set_xlabel('Annual Average Market price ($)',fontsize=28,**csfont)
# cf_ax8.set_ylabel('Number of Years',labelpad=30,fontsize=28,**csfont)
cf_ax8.tick_params(labelsize=26)





######################### Legend
# handles, labels = cf_ax1.get_legend_handles_labels()
# fig.legend(handles, labels, loc='upper center', ncol=6, bbox_to_anchor=(0.52, 1.18), fancybox=False, shadow=False, prop= {'family':'Arial' , 'size':'30'})# fontsize=26,**csfont)

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

plt.savefig('Plots/Fig_SI_Price_Distribution.png' , bbox_inches='tight',dpi=250)


