# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 20:03:05 2021

@author: mzeigha
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.colors import TwoSlopeNorm
import matplotlib
import statistics

csfont = {'fontname':'Arial'}

damage = pd.read_csv('Results/SNP_damage.csv')
stochastic = pd.read_csv('Results/Stochastic_df.csv')



damage['Day'] = pd.RangeIndex(182500)
damage["Year"] = damage["Day"]//365
damage['DOY'] = damage['Day']-damage['Year']*365

damage['load'] = stochastic['CA_load']


damage = damage.drop(index = damage[damage['DOY']==364].index).reset_index()

production_no_tax = pd.read_csv('Results/production_df_no_tax.csv')
production_SNP = pd.read_csv('Results/production_df_SNP.csv')

generators = pd.read_csv('Results/generators.csv',index_col=0 )



plot4 = damage[damage['no_tax']>4300000]

plot3 = damage[damage['no_tax']<=4300000]
plot3 = damage[damage['no_tax']>2850000]

plot2 = damage[damage['no_tax']<=2850000]
plot2 = damage[damage['no_tax']>1800000]

plot1 = damage[damage['no_tax']<=1800000]

plot5 = pd.DataFrame(damage.iloc[146155,:]).T

plot6 = damage[damage['no_tax']==damage['no_tax'].max()]



production_no_tax_1 = production_no_tax[production_no_tax.index.isin(plot1.index)]
production_no_tax_2 = production_no_tax[production_no_tax.index.isin(plot2.index)]
production_no_tax_3 = production_no_tax[production_no_tax.index.isin(plot3.index)]
production_no_tax_4 = production_no_tax[production_no_tax.index.isin(plot4.index)]
production_no_tax_5 = production_no_tax[production_no_tax.index.isin(plot5.index)]
production_no_tax_6 = production_no_tax[production_no_tax.index.isin(plot6.index)]

production_no_tax_1_mean = production_no_tax_1.mean(axis =0)
production_no_tax_2_mean = production_no_tax_2.mean(axis =0)
production_no_tax_3_mean = production_no_tax_3.mean(axis =0)
production_no_tax_4_mean = production_no_tax_4.mean(axis =0)
production_no_tax_5_mean = production_no_tax_5.mean(axis =0)
production_no_tax_6_mean = production_no_tax_6.mean(axis =0)


production_SNP_1 = production_SNP[production_SNP.index.isin(plot1.index)]
production_SNP_2 = production_SNP[production_SNP.index.isin(plot2.index)]
production_SNP_3 = production_SNP[production_SNP.index.isin(plot3.index)]
production_SNP_4 = production_SNP[production_SNP.index.isin(plot4.index)]
production_SNP_5 = production_SNP[production_SNP.index.isin(plot5.index)]
production_SNP_6 = production_SNP[production_SNP.index.isin(plot6.index)]


production_SNP_1_mean = production_SNP_1.mean(axis =0)
production_SNP_2_mean = production_SNP_2.mean(axis =0)
production_SNP_3_mean = production_SNP_3.mean(axis =0)
production_SNP_4_mean = production_SNP_4.mean(axis =0)
production_SNP_5_mean = production_SNP_5.mean(axis =0)
production_SNP_6_mean = production_SNP_6.mean(axis =0)

#%%

df = generators[['netcap','var_om','NOXrate(lbs/MWh)','seg1','seg2','seg3']]
df['tax'] = generators['NOXTax($/MWh)'] + generators['SO2Tax($/MWh)'] + generators['PMTax($/MWh)']

for i in generators.index:
    if generators.loc[i,'zone']== 'PGE_valley':
       df.loc[i,'gas'] = 4.66
       
    elif generators.loc[i,'zone']=='PGE_bay':
       df.loc[i,'gas'] = 4.66
       
    else:
       df.loc[i,'gas'] = 4.47
       

for i in generators.index:
    df.loc[i, 'no_tax_cost'] = df.loc[i,'gas']*df.loc[i,'seg1']+df.loc[i,'gas']*df.loc[i,'seg2']+df.loc[i,'gas']*df.loc[i,'seg3']+df.loc[i,'gas']*df.loc[i,'var_om']
    df.loc[i, 'SNP_cost'] = df.loc[i, 'no_tax_cost'] + df.loc[i,'tax']
    df.loc[i, 'no_tax_A_CF'] = (production_no_tax_1_mean[i]/generators.loc[i,'netcap'])/24
    df.loc[i, 'SNP_A_CF'] = (production_SNP_1_mean[i]/generators.loc[i,'netcap'])/24
    df.loc[i,'diff_A'] = (df.loc[i, 'SNP_A_CF'] - df.loc[i, 'no_tax_A_CF'])*df.loc[i, 'netcap']*24
    
    df.loc[i, 'no_tax_B_CF'] = (production_no_tax_2_mean[i]/generators.loc[i,'netcap'])/24
    df.loc[i, 'SNP_B_CF'] = (production_SNP_2_mean[i]/generators.loc[i,'netcap'])/24
    df.loc[i,'diff_B'] = (df.loc[i, 'SNP_B_CF'] - df.loc[i, 'no_tax_B_CF'])*df.loc[i, 'netcap']*24
    
    df.loc[i, 'no_tax_C_CF'] = (production_no_tax_3_mean[i]/generators.loc[i,'netcap'])/24
    df.loc[i, 'SNP_C_CF'] = (production_SNP_3_mean[i]/generators.loc[i,'netcap'])/24
    df.loc[i,'diff_C'] = (df.loc[i, 'SNP_C_CF'] - df.loc[i, 'no_tax_C_CF'])*df.loc[i, 'netcap']*24

    df.loc[i, 'no_tax_D_CF'] = (production_no_tax_4_mean[i]/generators.loc[i,'netcap'])/24
    df.loc[i, 'SNP_D_CF'] = (production_SNP_4_mean[i]/generators.loc[i,'netcap'])/24
    df.loc[i,'diff_D'] = (df.loc[i, 'SNP_D_CF'] - df.loc[i, 'no_tax_D_CF'])*df.loc[i, 'netcap']*24  

    df.loc[i, 'no_tax_E_CF'] = (production_no_tax_5_mean[i]/generators.loc[i,'netcap'])/24
    df.loc[i, 'SNP_E_CF'] = (production_SNP_5_mean[i]/generators.loc[i,'netcap'])/24
    df.loc[i,'diff_E'] = (df.loc[i, 'SNP_E_CF'] - df.loc[i, 'no_tax_E_CF'])*df.loc[i, 'netcap']*24

    df.loc[i, 'no_tax_F_CF'] = (production_no_tax_6_mean[i]/generators.loc[i,'netcap'])/24
    df.loc[i, 'SNP_F_CF'] = (production_SNP_6_mean[i]/generators.loc[i,'netcap'])/24
    df.loc[i,'diff_F'] = (df.loc[i, 'SNP_F_CF'] - df.loc[i, 'no_tax_F_CF'])*df.loc[i, 'netcap']*24
    


#%%

bin1 = df[df['tax']<=4]

bin2 = df[df['tax']>4]
bin2 = bin2[bin2['tax']<=6]

bin3 = df[df['tax']>6]
bin3 = bin3[bin3['tax']<=8]

bin4 = df[df['tax']>8]
bin4 = bin4[bin4['tax']<=10]

bin5 = df[df['tax']>10]
bin5 = bin5[bin5['tax']<=12]

bin6 = df[df['tax']>12]
bin6 = bin6[bin6['tax']<=14]

bin7 = df[df['tax']>14]
bin7 = bin7[bin7['tax']<=16]

bin8 = df[df['tax']>16]
bin8 = bin8[bin8['tax']<=18]


bin9 = df[df['tax']>18]
bin9 = bin9[bin9['tax']<=100]

bin10 = df[df['tax']>100]

bins = pd.DataFrame([], columns = ['diff_A','diff_B','diff_C','diff_D','diff_E','diff_F', 'cost'])
marginal_cost = pd.DataFrame([])

for i in ['diff_A','diff_B','diff_C','diff_D','diff_E','diff_F']:
    
    bins.loc[0,i] = bin1[i].sum()/1000
    bins.loc[1,i] = bin2[i].sum()/1000
    bins.loc[2,i] = bin3[i].sum()/1000
    bins.loc[3,i] = bin4[i].sum()/1000
    bins.loc[4,i] = bin5[i].sum()/1000
    bins.loc[5,i] = bin6[i].sum()/1000
    bins.loc[6,i] = bin7[i].sum()/1000
    bins.loc[7,i] = bin8[i].sum()/1000
    bins.loc[8,i] = bin9[i].sum()/1000
    bins.loc[9,i] = bin10[i].sum()/1000  
    # bins.loc[10,i] = bin11[i].sum()/1000    
    # bins.loc[11,i] = bin12[i].sum()/1000
    
bins.loc[0,'cost'] = bin1['no_tax_cost'].mean()
bins.loc[1,'cost'] = bin2['no_tax_cost'].mean()
bins.loc[2,'cost'] = bin3['no_tax_cost'].mean()
bins.loc[3,'cost'] = bin4['no_tax_cost'].mean()
bins.loc[4,'cost'] = bin5['no_tax_cost'].mean()
bins.loc[5,'cost'] = bin6['no_tax_cost'].mean()
bins.loc[6,'cost'] = bin7['no_tax_cost'].mean()
bins.loc[7,'cost'] = bin8['no_tax_cost'].mean()
bins.loc[8,'cost'] = bin9['no_tax_cost'].mean()
bins.loc[9,'cost'] = bin10['no_tax_cost'].mean()  

bins.loc[0,'SNP_cost'] = bin1['SNP_cost'].mean()
bins.loc[1,'SNP_cost'] = bin2['SNP_cost'].mean()
bins.loc[2,'SNP_cost'] = bin3['SNP_cost'].mean()
bins.loc[3,'SNP_cost'] = bin4['SNP_cost'].mean()
bins.loc[4,'SNP_cost'] = bin5['SNP_cost'].mean()
bins.loc[5,'SNP_cost'] = bin6['SNP_cost'].mean()
bins.loc[6,'SNP_cost'] = bin7['SNP_cost'].mean()
bins.loc[7,'SNP_cost'] = bin8['SNP_cost'].mean()
bins.loc[8,'SNP_cost'] = bin9['SNP_cost'].mean()
bins.loc[9,'SNP_cost'] = bin10['SNP_cost'].mean()  


fig = plt.figure( figsize = (9,12), constrained_layout=False)

gs = fig.add_gridspec(3,2, width_ratios=[1, 1], height_ratios = [1, 1,1] ,hspace=0.1, wspace=0.1 )

ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[1,0])
ax4 = fig.add_subplot(gs[1,1])
ax5 = fig.add_subplot(gs[2,0])
ax6 = fig.add_subplot(gs[2,1])


ax1.bar(bins.index,bins['diff_A'],color='bisque',edgecolor='black')
ax1.set_xticks([0,1,2,3,4,5,6,7,8,9])
ax1.set_xticklabels([])
ax1.set_yticks([-45,-35,-25,-15,-5,5,15,25,35,45])
# ax4.set_yticklabels([])
# ax1.annotate('Total Generation = {} GWh'.format(np.round(((df['SNP_A_CF'])*df['netcap']*24/1000).sum(),decimals=1)),(-0.5,-28.000),annotation_clip=False, fontsize=13,**csfont)
# ax1.annotate('Load = {} GWh'.format(np.round((plot1['load'].mean()/1000),decimals=1)),(-0.5,-33.000),annotation_clip=False, fontsize=13,**csfont)
ax1.annotate('Demand = {} GWh'.format(np.round(plot1['load'].mean()/1000 ,decimals=1)),(-0.5,-30.000),annotation_clip=False, fontsize=11,**csfont)
ax1.annotate('Fossil Fuel Capacity Usage = {}%'.format(np.round((100*df['SNP_A_CF']*df['netcap']).sum()/df['netcap'].sum() ,decimals=1)),(-0.5,-37.000),annotation_clip=False, fontsize=11,**csfont)
ax1.annotate('a',(-0.5,47.000),annotation_clip=False, fontsize=19,**csfont,weight='bold')
ax1.set_ylim(-42, 55)

ax2.bar(bins.index,bins['diff_B'],color='bisque',edgecolor='black')
ax2.set_xticks([0,1,2,3,4,5,6,7,8,9])
ax2.set_xticklabels([])
ax2.set_yticks([-45,-35,-25,-15,-5,5,15,25,35,45])
ax2.set_yticklabels([])
# ax2.annotate('Total Generation = {} GWh'.format(np.round(((df['SNP_B_CF'])*df['netcap']*24/1000).sum(),decimals=1)),(-0.5,-28.000),annotation_clip=False, fontsize=13,**csfont)
# ax2.annotate('Load = {} GWh'.format(np.round((plot2['load'].mean()/1000),decimals=1)),(-0.5,-33.000),annotation_clip=False, fontsize=13,**csfont)
ax2.annotate('Demand = {} GWh'.format(np.round(plot2['load'].mean()/1000 ,decimals=1)),(-0.5,-30.000),annotation_clip=False, fontsize=11,**csfont)
ax2.annotate('Fossil Fuel Capacity Usage =  {}%'.format( np.round((100*df['SNP_B_CF']*df['netcap']).sum()/df['netcap'].sum() ,decimals=1)),(-0.5,-37.000),annotation_clip=False, fontsize=11,**csfont)
ax2.annotate('b',(-0.5,47.000),annotation_clip=False, fontsize=19,**csfont,weight='bold')
ax2.set_ylim(-42, 55)

ax3.bar(bins.index,bins['diff_C'],color='bisque',edgecolor='black')
ax3.set_xticks([0,1,2,3,4,5,6,7,8,9])
ax3.set_xticklabels([])
ax3.set_yticks([-45,-35,-25,-15,-5,5,15,25,35,45])
# ax4.set_yticklabels([])
# ax3.annotate('Total Generation = {} GWh'.format(np.round(((df['SNP_C_CF'])*df['netcap']*24/1000).sum(),decimals=1)),(-0.5,-28.000),annotation_clip=False, fontsize=13,**csfont)
# ax3.annotate('Load = {} GWh'.format(np.round((plot3['load'].mean()/1000),decimals=1)),(-0.5,-33.000),annotation_clip=False, fontsize=13,**csfont)
ax3.annotate('Demand = {} GWh'.format(np.round(plot3['load'].mean()/1000 ,decimals=1)),(-0.5,-30.000),annotation_clip=False, fontsize=11,**csfont)
ax3.annotate('Fossil Fuel Capacity Usage = {}%'.format(np.round((100*df['SNP_C_CF']*df['netcap']).sum()/df['netcap'].sum() ,decimals=1)),(-0.5,-37.000),annotation_clip=False, fontsize=11,**csfont)
ax3.annotate('c',(-0.5,47.000),annotation_clip=False, fontsize=19,**csfont,weight='bold')
ax3.set_ylim(-42, 55)

ax4.bar(bins.index,bins['diff_D'],color='bisque',edgecolor='black')
ax4.set_xticks([0,1,2,3,4,5,6,7,8,9])
ax4.set_xticklabels([])
ax4.set_yticks([-45,-35,-25,-15,-5,5,15,25,35,45])
ax4.set_yticklabels([])
# ax4.annotate('Total Generation = {} GWh'.format(np.round(((df['SNP_D_CF'])*df['netcap']*24/1000).sum(),decimals=1)),(-0.5,-28.000),annotation_clip=False, fontsize=13,**csfont)
# ax4.annotate('Load = {} GWh'.format(np.round((plot4['load'].mean()/1000),decimals=1)),(-0.5,-33.000),annotation_clip=False, fontsize=13,**csfont)
ax4.annotate('Demand = {} GWh'.format(np.round(plot4['load'].mean()/1000 ,decimals=1)),(-0.5,-30.000),annotation_clip=False, fontsize=11,**csfont)
ax4.annotate('Fossil Fuel Capacity Usage = {}%'.format(np.round((100*df['SNP_D_CF']*df['netcap']).sum()/df['netcap'].sum() ,decimals=1)),(-0.5,-37.000),annotation_clip=False, fontsize=11,**csfont)
ax4.annotate('d',(-0.5,47.000),annotation_clip=False, fontsize=19,**csfont,weight='bold')
ax4.set_ylim(-42, 55)

ax5.bar(bins.index,bins['diff_E'],color='bisque',edgecolor='black')
ax5.set_xticks([0,1,2,3,4,5,6,7,8,9])
ax5.set_xticklabels(['0-4','4-6','6-8','8-10','10-12','12-14','14-16','16-18','18-100','100-286'], rotation=270)
ax5.set_yticks([-45,-35,-25,-15,-5,5,15,25,35,45])
# ax5.annotate('Total Generation = {} GWh'.format(np.round(((df['SNP_E_CF'])*df['netcap']*24/1000).sum(),decimals=1)),(-0.5,-28.000),annotation_clip=False, fontsize=13,**csfont)
# ax5.annotate('Load = {} GWh'.format(np.round((plot5['load'].mean()/1000),decimals=1)),(-0.5,-33.000),annotation_clip=False, fontsize=13,**csfont)
ax5.annotate('Demand = {} GWh'.format(np.round(plot5['load'].mean()/1000 ,decimals=1)),(-0.5,-30.000),annotation_clip=False, fontsize=11,**csfont)
ax5.annotate('Fossil Fuel Capacity Usage = {}%'.format(np.round((100*df['SNP_E_CF']*df['netcap']).sum()/df['netcap'].sum() ,decimals=1)),(-0.5,-37.000),annotation_clip=False, fontsize=11,**csfont)
ax5.annotate('e',(-0.5,47.000),annotation_clip=False, fontsize=19,**csfont,weight='bold')
ax5.set_ylim(-42, 55)

ax6.bar(bins.index,bins['diff_F'],color='bisque',edgecolor='black')
ax6.set_xticks([0,1,2,3,4,5,6,7,8,9])
ax6.set_xticklabels(['0-4','4-6','6-8','8-10','10-12','12-14','14-16','16-18','18-100','100-286'], rotation=270)
ax6.set_yticks([-45,-35,-25,-15,-5,5,15,25,35,45])
ax6.set_yticklabels([])
# ax6.annotate('Total Generation = {} GWh'.format(np.round(((df['SNP_F_CF'])*df['netcap']*24/1000).sum(),decimals=1)),(-0.5,-23.000),annotation_clip=False, fontsize=13,**csfont)
ax6.annotate('Demand = {} GWh'.format(np.round(plot6['load'].mean()/1000 ,decimals=1)),(-0.5,-30.000),annotation_clip=False, fontsize=11,**csfont)
ax6.annotate('Fossil Fuel Capacity Usage = {}%'.format(np.round((100*df['SNP_F_CF']*df['netcap']).sum()/df['netcap'].sum() ,decimals=1)),(-0.5,-37.000),annotation_clip=False, fontsize=11,**csfont)
ax6.annotate('f',(-0.5,47.000),annotation_clip=False, fontsize=19,weight='bold', **csfont)
ax6.set_ylim(-42, 55)


plt.annotate('Change in Power Generation (GWh)',(-14.8,63),annotation_clip=False,fontsize= 15,rotation=90,**csfont)
plt.annotate('Emission Penalty ($/MWh)',(-5,-73),annotation_clip=False,fontsize= 15,**csfont)

plt.savefig('Plots/Fig_6_bins.pdf', bbox_inches='tight',dpi=250)

