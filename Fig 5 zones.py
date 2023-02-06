# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 23:14:58 2021

@author: mzeigha
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from scipy import stats


damage = pd.read_csv('Results/SNP_damage.csv')


damage['Day'] = pd.RangeIndex(182500)
damage["Year"] = damage["Day"]//365
damage['DOY'] = damage['Day']-damage['Year']*365
damage['diff'] = damage['no_tax'] - damage['SNP']

annual_damage = pd.DataFrame([])
annual_damage['no_tax'] = damage.groupby(['Year'], as_index = True)['no_tax'].sum()



for i in range(len(annual_damage)):
    annual_damage.loc[i,'percentile'] = stats.percentileofscore(annual_damage['no_tax'], annual_damage.loc[i,'no_tax'])
    


for i in range(len(damage)):
    damage.loc[i,'annual_damage'] = annual_damage.loc[damage.loc[i,'Year'], 'no_tax']
    damage.loc[i,'annual_perc'] = annual_damage.loc[damage.loc[i,'Year'], 'percentile']
    # print(i)
    

#%%
stochastic = pd.read_csv('Results/Stochastic_df.csv')
temp = pd.read_csv('Results/Average values/temp_annual.csv')

column_name = ['Damage','Prevention','Annual Damage', 'Annual Damage Percentile','Day of Year', 'Year' , 'Daily Temperature', 'Daily CA Load' ,'Daily CA Hydropower','Daily PNW Hydropower','Daily Wind Power', 'Daily Solar Power']

plot = pd.concat([damage[['no_tax','diff', 'annual_damage' ,'annual_perc','DOY', 'Year']],temp['avg'], stochastic[['CA_load','CA_Hydropower','PNW_Hydropower','CA_Wind_Power','Solar_Power']]], axis = 1)
plot.columns = column_name

plot.loc[plot['Daily PNW Hydropower']>650000 ,'Daily PNW Hydropower' ] = 650000

for i in ['Daily CA Load' ,'Daily CA Hydropower','Daily PNW Hydropower','Daily Wind Power', 'Daily Solar Power']:
    plot[i]=plot[i]/1000


plot.columns= ['Damage','Prevention','Annual Damage','Annual Damage Percentile', 'Day of Year' ,'Year' , 'Temperature', 'CA Load' ,'CA Hydropower','PNW Hydropower','Wind Power', 'Solar Power']


# plot1 = plot[plot['Damage']>4350000]

# plot2 = plot[plot['Damage']<=4350000]

plot1 = plot[plot['Damage']>4300000].reset_index()

plot2 = plot[plot['Damage']<=4300000]
plot2 = plot2[plot2['Damage']>2850000].reset_index()

plot3 = plot[plot['Damage']<=2850000]
plot3 = plot3[plot3['Damage']>1800000].reset_index()

plot4 = plot[plot['Damage']<=1800000].reset_index()



# for i in range(len(plot1)):
#     plot1.loc[i,'annual_per'] = annual_damage.loc[plot1.loc[i,'Year'], 'percentile']

# count = plot1[plot1['annual_per']>50]
#%%
count = pd.DataFrame()
count.loc[0,'count'] = plot1.shape[0]
count.loc[1,'count'] = plot2.shape[0]
count.loc[2,'count'] = plot3.shape[0]
count.loc[3,'count'] = plot4.shape[0]



#%%
from matplotlib.ticker import PercentFormatter
csfont = {'fontname':'Arial','size':'16'}
csfont_title = {'fontname':'Arial','size':'18'}



fig = plt.figure( figsize = (12,9), constrained_layout=True)
#constrained_layout=True,


gs = fig.add_gridspec(2,2, width_ratios=[1, 1, ], height_ratios = [1,1])#, wspace=0.25 )


ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[1,0])
ax4 = fig.add_subplot(gs[1,1])


# [0,10,20,30,40,50,60,70,80,90,100]
ax1.hist(plot4['Annual Damage Percentile'], bins = [0,10,20,30,40,50,60,70,80,90,100] ,weights=100*np.ones_like(plot4['Annual Damage Percentile']) / len(plot4['Annual Damage Percentile']), align ='mid', color='bisque',edgecolor='black' ,density=False)
# ax1.gca().yaxis.set_major_formatter(PercentFormatter(.1))
ax1.set_ylabel('% days', **csfont)
# ax1.set_xlabel('Annual damage percentile',**csfont)
# ax1.set_xticks([0,10,20,30,40,50,60,70,80,90,100])
ax1.set_title('Zone "a"', **csfont_title)




ax2.hist(plot3['Annual Damage Percentile'], bins = [0,10,20,30,40,50,60,70,80,90,100] ,weights=100*np.ones_like(plot3['Annual Damage Percentile']) / len(plot3['Annual Damage Percentile']),align ='mid', color='bisque',edgecolor='black' ,density=False)
# ax1.gca().yaxis.set_major_formatter(PercentFormatter(.1))
# ax2.set_ylabel('% of days', **csfont)
# ax2.set_xlabel('Annual damage percentile',**csfont)
ax2.set_xticks([0,10,20,30,40,50,60,70,80,90,100])
ax2.set_title('Zone "b"', **csfont_title)


ax3.hist(plot2['Annual Damage Percentile'], bins = [0,10,20,30,40,50,60,70,80,90,100] ,weights=100*np.ones_like(plot2['Annual Damage Percentile']) / len(plot2['Annual Damage Percentile']),align ='mid', color='bisque',edgecolor='black' ,density=False)
# ax1.gca().yaxis.set_major_formatter(PercentFormatter(.1))
ax3.set_ylabel('% days', **csfont)
ax3.set_xlabel('Annual damage percentile',**csfont)
ax3.set_xticks([0,10,20,30,40,50,60,70,80,90,100])
ax3.set_title('Zone "c"', **csfont_title)


ax4.hist(plot1['Annual Damage Percentile'], bins = [0,10,20,30,40,50,60,70,80,90,100] ,weights=100*np.ones_like(plot1['Annual Damage Percentile']) / len(plot1['Annual Damage Percentile']),align ='mid', color='bisque',edgecolor='black' ,density=False)
# ax1.gca().yaxis.set_major_formatter(PercentFormatter(.1))
# ax4.set_ylabel('% of days', **csfont)
ax4.set_xlabel('Annual damage percentile',**csfont)
ax4.set_xticks([0,10,20,30,40,50,60,70,80,90,100])
ax4.set_title('Zone "d"', **csfont_title)



plt.savefig('Plots/Fig_SI_zone D.png',bbox_inches='tight' ,dpi=250)
#%%

fig = plt.figure( figsize = (12,9), constrained_layout=True)
#constrained_layout=True,

csfont = {'fontname':'Arial'}

gs = fig.add_gridspec(4,4, width_ratios=[1, 1, 1, 1], height_ratios = [1,1,1, 1],hspace=0.12)#, wspace=0.25 )

ax1 = fig.add_subplot(gs[0:3,0:3])


ax2 = fig.add_subplot(gs[3,0])
ax3 = fig.add_subplot(gs[3,1])
ax4 = fig.add_subplot(gs[3,2])
ax5 = fig.add_subplot(gs[0,3])
ax6 = fig.add_subplot(gs[1,3])
ax7 = fig.add_subplot(gs[2,3])
ax8 = fig.add_subplot(gs[3,3])


##############################################################################################################


ax1.axvline(x=1.800000,color='black', linestyle='dotted')
ax1.axvline(x=2.850000,color='black', linestyle='dotted')
ax1.axvline(x=4.300000,color='black', linestyle='dotted')

damage2 = damage.sort_values(by='annual_damage')
vmin = np.round(damage2['annual_damage'].min()/1000000,decimals=1)
vmax = np.round(damage2['annual_damage'].max()/1000000,decimals=1)
vcenter = np.round(damage2['annual_damage'].mean()/1000000,decimals=1)
norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

cbar = ax1.scatter(damage2['no_tax']/1000000,damage2['diff']/1000000,edgecolor='black',linewidths=0.3,s=30 , c = damage2['annual_damage']/1000000, cmap = plt.cm.Reds ,norm = norm)# vmin = damage2['annual_damage'].min() , vmax =damage2['annual_damage'].max() )
# ax1.scatter(damage2['no_tax'],damage2['diff'],s=100, c='bisque',edgecolor='coral' )
# cbar = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.cool_r)

cax = plt.axes([0.47, 0.45, 0.02, 0.23])

cbar = fig.colorbar(cbar, ax = ax1, cax= cax , orientation="vertical" , ticks = [vmin,vcenter,vmax] )#.set_ticklabels([3.13,8.58])
cbar.set_ticklabels([vmin,vcenter,vmax])
cbar.ax.tick_params(labelsize=15)
cbar.set_label("Annual Damage ($M)", fontsize=16, labelpad = -95,**csfont)


ax1.set_ylabel('Prevented Damages under SNP Tax Scenario ($M)', fontsize=18,**csfont)
ax1.set_xlabel('Damages under No Tax Scenario ($M)',fontsize=18,**csfont)

ax1.tick_params(labelsize=16)

ax1.annotate('a',(1.1,0.2),annotation_clip=False, fontsize=30,**csfont,weight='bold')
ax1.annotate('b',(2.2,0.2),annotation_clip=False, fontsize=30,**csfont,weight='bold')
ax1.annotate('c',(3.6,0.2),annotation_clip=False, fontsize=30,**csfont,weight='bold')
ax1.annotate('d',(6.3,0.2),annotation_clip=False, fontsize=30,**csfont,weight='bold')
ax1.annotate('e',(6.4,2.15),annotation_clip=False, fontsize=30,**csfont,weight='bold')
ax1.annotate('f',(8.5,0.55),annotation_clip=False, fontsize=30,**csfont,weight='bold')


ax1.scatter(5.95, 2.07 , s = 600, facecolor = 'none' ,edgecolor='black',linewidths=1.5 )
ax1.scatter(8.57, 0.331 , s = 600, facecolor = 'none' ,edgecolor='black',linewidths=1.5 )


#                 0             1                2       3                 4            5               6             7
###############################################################################################################
columns = ['Prevention','Day of Year' , 'Temperature', 'CA Load' ,'CA Hydropower','PNW Hydropower','Wind Power', 'Solar Power']

sns.kdeplot(data = plot4 ,x = columns[1], color='royalblue', ax=ax2, label = 'a ({} % Days)'.format(str(np.round((len(plot4)/len(plot))*100 ,decimals = 2))))
sns.kdeplot(data = plot3 ,x = columns[1], color='lightseagreen', ax=ax2, label = 'b ({} % Days)'.format(str(np.round((len(plot3)/len(plot))*100 ,decimals = 2))))
sns.kdeplot(data = plot2 ,x = columns[1], color='goldenrod', ax=ax2, label = 'c ({} % Days)'.format(str(np.round((len(plot2)/len(plot))*100 ,decimals = 2))))
sns.kdeplot(data = plot1 ,x = columns[1], color='maroon', ax=ax2, label = 'd ({} % Days)'.format(str(np.round((len(plot1)/len(plot))*100 ,decimals = 2))))
ax2.set_xlabel(columns[1],fontsize=16 )
ax2.set_xticks([0,90,181,273])
ax2.set_xticklabels(['Jan','Apr', 'Jul', 'Oct'],fontsize=10,**csfont)
ax2.tick_params(labelsize=16)
# ax2.set_yticks([])
ax2.set_ylabel('Density', fontsize = 16,**csfont)


sns.kdeplot(data = plot1 ,x = columns[2], color='maroon', ax=ax3)
sns.kdeplot(data = plot2 ,x = columns[2], color='goldenrod', ax=ax3)
sns.kdeplot(data = plot3 ,x = columns[2], color='lightseagreen', ax=ax3)
sns.kdeplot(data = plot4 ,x = columns[2], color='royalblue', ax=ax3)
ax3.set_xlabel(columns[2]+' (C)',fontsize=16,**csfont)
ax3.set_ylabel('')
ax3.tick_params(labelsize=16)
# ax3.set_yticks([])

sns.kdeplot(data = plot1 ,x = columns[3], color='maroon', ax=ax4)
sns.kdeplot(data = plot2 ,x = columns[3], color='goldenrod', ax=ax4)
sns.kdeplot(data = plot3 ,x = columns[3], color='lightseagreen', ax=ax4)
sns.kdeplot(data = plot4 ,x = columns[3], color='royalblue', ax=ax4)
ax4.set_xlabel('CAISO Deamnd (GWh)',fontsize=16,**csfont)
ax4.set_ylabel('')
ax4.tick_params(labelsize=16)
# ax4.set_yticks([])


sns.kdeplot(data = plot1 ,x = columns[4], color='maroon', ax=ax5)
sns.kdeplot(data = plot2 ,x = columns[4], color='goldenrod', ax=ax5)
sns.kdeplot(data = plot3 ,x = columns[4], color='lightseagreen', ax=ax5)
sns.kdeplot(data = plot4 ,x = columns[4], color='royalblue', ax=ax5)
ax5.set_xlabel('CAISO Hydropower (GWh)',fontsize=16,**csfont)
ax5.set_ylabel('')
ax5.tick_params(labelsize=16)
# ax5.set_yticks([])


sns.kdeplot(data = plot1 ,x = columns[5], color='maroon', ax=ax6)
sns.kdeplot(data = plot2 ,x = columns[5], color='goldenrod', ax=ax6)
sns.kdeplot(data = plot3 ,x = columns[5], color='lightseagreen', ax=ax6)
sns.kdeplot(data = plot4 ,x = columns[5], color='royalblue', ax=ax6)
ax6.set_xlabel(columns[5]+' (GWh)',fontsize=16,**csfont)
ax6.set_ylabel('')
ax6.tick_params(labelsize=16)
# ax6.set_yticks([])


sns.kdeplot(data = plot1 ,x = columns[6], color='maroon', ax=ax7)
sns.kdeplot(data = plot2 ,x = columns[6], color='goldenrod', ax=ax7)
sns.kdeplot(data = plot3 ,x = columns[6], color='lightseagreen', ax=ax7)
sns.kdeplot(data = plot4 ,x = columns[6], color='royalblue', ax=ax7)
ax7.set_xlabel(columns[6]+' (GWh)',fontsize=16,**csfont)
ax7.set_ylabel('')
ax7.tick_params(labelsize=16)
# ax7.set_yticks([])


sns.kdeplot(data = plot1 ,x = columns[7], color='maroon', ax=ax8)
sns.kdeplot(data = plot2 ,x = columns[7], color='goldenrod', ax=ax8)
sns.kdeplot(data = plot3 ,x = columns[7], color='lightseagreen', ax=ax8)
sns.kdeplot(data = plot4 ,x = columns[7], color='royalblue', ax=ax8)
ax8.set_xlabel(columns[7]+' (GWh)',fontsize=16,**csfont)
ax8.set_ylabel('')
ax8.tick_params(labelsize=16)
# ax8.set_yticks([])


fig.legend( loc='lower center', ncol=4, fancybox=False, shadow=False,fontsize=18, bbox_to_anchor=(0.5, -0.09), prop= {'family':'Arial' , 'size':'17'})


plt.savefig('Plots/Fig_5_zones_3.png',bbox_inches='tight' ,dpi=250)
