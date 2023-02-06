# -*- coding: utf-8 -*-
"""
Created on Thu May 27 15:20:10 2021

@author: mzeigha
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import pearsonr




#####Initializing
scenarios = ['all_tax' , 'CO2', 'no_tax', 'SNP']
scenarios_label = ['All' , 'CO2' , 'No', 'SNP']
stochastic_column = ['CA Load', 'PNW Load','CA Hydropower','PNW Hydropower','Path66 Flow','CA Wind Power','PNW Wind Power', 'CA Solar Power']
column_name = ['Local Damages', 'Market Price', 'Racial Inequality','Pollution Burden','CAISO Demand', 'PNW Load','CAISO Hydropower','PNW Hydropower','Path66 Flow','CAISO Wind Power','PNW Wind Power', 'CAISO Solar Power']
cmap = plt.cm.RdBu_r
# cmap = 'coolwarm'


# stochastic_percentile_df = pd.DataFrame([])
# SNP_damage_percentile_df = pd.DataFrame([])
# Shadow_price_percentile_df = pd.DataFrame([])
Day = pd.DataFrame([])


#### Reading the "Daily results" of simulation 
stochastic_df = pd.read_csv('Results/Stochastic_df.csv')
SNP_damage_df = pd.read_csv('Results/SNP_damage.csv')
Shadow_price_df = pd.read_csv('Results/Shadow_price.csv')
race = pd.read_csv('Equity/pollution_race_correlation.csv')
pollution_burden = pd.read_csv('Equity/pollution_score_damages_corr.csv')

#%%
########### Converting the Daily data to yearly

Day["Day"] = pd.RangeIndex(182500)

stochastic_df["Year"] = Day["Day"]//365
SNP_damage_df["Year"] = Day["Day"]//365
Shadow_price_df["Year"] = Day["Day"]//365


stochastic_df = stochastic_df.groupby(['Year']).sum()
SNP_damage_df = SNP_damage_df.groupby(['Year']).sum()
Shadow_price_df = Shadow_price_df.groupby(['Year']).mean()


# #####Building the percentile dataframe for each of results dataframe 

# for j in range(8):
#     print (j)
#     for i in range(len(stochastic_df)):
#         stochastic_percentile_df.loc[i,j] = stats.percentileofscore(stochastic_df.values[:,j], stochastic_df.values[i,j])
# print (stochastic_percentile_df)


      
# for j in range(4):
#     print (j)
#     for i in range(len(SNP_damage_df)):
#         SNP_damage_percentile_df.loc[i,j] = stats.percentileofscore(SNP_damage_df.values[:,j], SNP_damage_df.values[i,j])
# print (SNP_damage_percentile_df)

# for j in range(4):
#     print (j)
#     for i in range(len(Shadow_price_df)):
#         Shadow_price_percentile_df.loc[i,j] = stats.percentileofscore(Shadow_price_df.values[:,j], Shadow_price_df.values[i,j])
# print (Shadow_price_percentile_df)




# stochastic_percentile_df.columns = stochastic_column
# SNP_damage_percentile_df.columns = scenarios
# Shadow_price_percentile_df.columns = scenarios

########## Saving the DataFrames
# ##stochastic_percentile_df.to_csv (r'stochastic_percentile_df.csv', index = False, header=True)    
# ##CO2_damage_percentile_df.to_csv (r'CO2_damage_percentile_df.csv', index = False, header=True)    
# ##SNP_damage_percentile_df.to_csv (r'SNP_damage_percentile_df.csv', index = False, header=True)
# ##Shadow_price_percentile_df.to_csv (r'Shadow_price_percentile_df.csv', index = False, header=True)



# ########## Reading the DataFrames
# ##stochastic_percentile_df = pd.read_csv('stochastic_percentile_df.csv')
# ##CO2_damage_percentile_df = pd.read_csv('CO2_damage_percentile_df.csv')
# ##SNP_damage_percentile_df = pd.read_csv('SNP_damage_percentile_df.csv')
# ##Shadow_price_percentile_df = pd.read_csv('Shadow_price_percentile_df.csv')

#%% 
scenarios=['no_tax']

for k in scenarios:
    
        
    plot_df = plot_df2 = pd.concat([SNP_damage_df.loc[:,k] , Shadow_price_df.loc [:,k], race[k],pollution_burden[k] , stochastic_df ] , axis = 1)
    plot_df.columns = column_name
    plot_df2.columns = column_name
    
    plot_df = plot_df.drop(columns = ['PNW Load','Path66 Flow' , 'PNW Wind Power'])
    plot_df2 = plot_df2.drop(columns = ['PNW Load','Path66 Flow' , 'PNW Wind Power'])



    temp_df = pd.DataFrame(index=plot_df.index,columns=plot_df.columns)
    for i in range(len(plot_df)):
        for j in range(len(plot_df.columns)):
            temp_df.iloc[i,j] = (plot_df.iloc[i,j]-plot_df[plot_df.columns[j]].min())/(plot_df[plot_df.columns[j]].max()-plot_df[plot_df.columns[j]].min())*100
        print (i)
    plot_df = temp_df    
#%%    
    
         
# #     ##### Categorize to high and low
    for i in range(len(plot_df)):
        if plot_df.loc[i,'Local Damages'] <= 10:
            plot_df.loc[i,'Dataset'] = 'Low'
        elif plot_df.loc[i,'Local Damages'] >= 90:
            plot_df.loc[i,'Dataset'] = 'High'
        else:
            plot_df.loc[i,'Dataset'] = 'None'
################################################################

            
    plot_df_high = plot_df.loc[plot_df['Dataset'] == 'High']
    plot_df_low = plot_df.loc[plot_df['Dataset'] == 'Low']
    plot_df_none = plot_df.loc[plot_df['Dataset'] == 'None']
    
    
    plot_df = plot_df.drop(columns = ['Dataset'])        
    plot_df['Rank'] = plot_df['Local Damages']
    # plot_df['Rank'] = SNP_damage_df['no_tax']
    # plot_df['Rank'] = SNP_damage_percentile_df['no_tax']
    

    csfont = {'fontname':'arial'}
    
    
    plt.figure(figsize = (15,6))
    plt.style.use('seaborn-white')
    fig = parallel_coordinates(plot_df.sort_values(by='Local Damages'),'Rank', colormap=cmap, alpha = 0.7, lw = 1.5)
    plt.legend('')

    plt.annotate('${}B'.format(str(np.round(np.max(SNP_damage_df.loc[:,k])/1000000000,decimals=2))),(0.02,100),annotation_clip=False, fontsize=14)
    plt.annotate('${}M'.format(str(np.round(np.min(SNP_damage_df.loc[:,k])/1000000,decimals=0))),(0.02,-3.8),annotation_clip=False, fontsize=14)

    plt.annotate('${}'.format(str(np.round(np.max(Shadow_price_df.loc[:,k]),2))),(1.02,100),annotation_clip=False, fontsize=14)
    plt.annotate('${}'.format(str(np.round(np.min(Shadow_price_df.loc[:,k]),2))),(1.02,-3.8),annotation_clip=False, fontsize=14)
    
    plt.annotate(str(np.round(np.max(race.loc[:,k]),4)),(2.02,100),annotation_clip=False, fontsize=14)
    plt.annotate(str(np.round(np.min(race.loc[:,k]),4)),(2.02,-3.8),annotation_clip=False, fontsize=14)
    
    plt.annotate(str(np.round(np.max(pollution_burden.loc[:,k]),4)),(3.02,100),annotation_clip=False, fontsize=14)
    plt.annotate(str(np.round(np.min(pollution_burden.loc[:,k]),4)),(3.02,-3.8),annotation_clip=False, fontsize=14)
        
    plt.annotate('{} TWh'.format(str(np.round(np.max(stochastic_df.iloc[:,0]/1000000),decimals=0))),(4.02,100),annotation_clip=False, fontsize=14)
    plt.annotate('{} TWh'.format(str(np.round(np.min(stochastic_df.iloc[:,0]/1000000),decimals=0))),(4.02,-3.8),annotation_clip=False, fontsize=14)    
    
    plt.annotate('{} TWh'.format(str(np.round(np.max(stochastic_df.iloc[:,2]/1000000),decimals=0))),(5.02,100),annotation_clip=False, fontsize=14)
    plt.annotate('{} TWh'.format(str(np.round(np.min(stochastic_df.iloc[:,2]/1000000),decimals=0))),(5.02,-3.8),annotation_clip=False, fontsize=14) 
    
    plt.annotate('{} TWh'.format(str(np.round(np.max(stochastic_df.iloc[:,3]/1000000),decimals=0))),(6.02,100),annotation_clip=False, fontsize=14)
    plt.annotate('{} TWh'.format(str(np.round(np.min(stochastic_df.iloc[:,3]/1000000),decimals=0))),(6.02,-3.8),annotation_clip=False, fontsize=14) 
 
    plt.annotate('{} TWh'.format(str(np.round(np.max(stochastic_df.iloc[:,5]/1000000),decimals=0))),(7.02,100),annotation_clip=False, fontsize=14)
    plt.annotate('{} TWh'.format(str(np.round(np.min(stochastic_df.iloc[:,5]/1000000),decimals=0))),(7.02,-3.8),annotation_clip=False, fontsize=14)

    plt.annotate('{} TWh'.format(str(np.round(np.max(stochastic_df.iloc[:,7]/1000000),decimals=0))),(8.02,100),annotation_clip=False, fontsize=14)
    plt.annotate('{} TWh'.format(str(np.round(np.min(stochastic_df.iloc[:,7]/1000000),decimals=0))),(8.02,-3.8),annotation_clip=False, fontsize=14)
  
    ticks = ['${}M'.format(str(np.round(np.min(SNP_damage_df.loc[:,k])/1000000,decimals=0))), '${}B'.format(str(np.round(np.max(SNP_damage_df.loc[:,k])/1000000000,decimals=2)))]
    
    plt.xticks(rotation=90, fontsize=17)
    plt.yticks([])

    
    # plt.title('Parallel Plot of Emission Damage vs Power System Variables', fontsize=14)
    plt.savefig('Plots/Fig_6_Parallel', bbox_inches='tight',dpi=250)
    # plt.clf()
    # a = ['${}B'.format(str(np.round(np.max(SNP_damage_df.loc[:,k])/1000000000,decimals=2))), '${}M'.format(str(np.round(np.min(SNP_damage_df.loc[:,k])/1000000,decimals=0)))]
#%%

scenarios=['no_tax','SNP']

def pearsonr_pval(x,y):
    return pearsonr(x,y)[1]

for k in scenarios:
    
        
    plot_df = plot_df2 = pd.concat([SNP_damage_df.loc[:,k] , Shadow_price_df.loc [:,k], race[k],pollution_burden[k] , stochastic_df ] , axis = 1)
    plot_df.columns = column_name
    plot_df2.columns = column_name
    
    # plot_df = plot_df.drop(columns = ['PNW Load','Path66 Flow' , 'PNW Wind Power','Racial Inequality','Pollution Burden' ])
    # plot_df2 = plot_df2.drop(columns = ['PNW Load','Path66 Flow' , 'PNW Wind Power','Racial Inequality','Pollution Burden'])
    

    plot_df = plot_df.drop(columns = ['PNW Load','Path66 Flow' , 'PNW Wind Power' ])
    plot_df2 = plot_df2.drop(columns = ['PNW Load','Path66 Flow' , 'PNW Wind Power'])
    
    
    csfont = {'fontname':'arial'}
    heatmap = plot_df2.corr()
    Pval = plot_df2.corr(method=pearsonr_pval)

    mask = np.triu(np.ones_like(heatmap, dtype=bool))

    mask = np.delete(mask,8, axis = 1)

    mask = np.delete(mask,0, axis = 0)
    
    heatmap = heatmap.drop('Local Damages')  
    heatmap = heatmap.drop('CAISO Solar Power', axis = 1)  
    
    plt.figure(figsize = (8,6))
    plt.style.use('seaborn-white')
    ax = sns.heatmap(heatmap,  mask = mask , cmap = 'RdBu_r' , square = True, center=0, vmin = -1, vmax = 1 ,annot_kws={'size': '13', "fontfamily":"Arial"}, annot = np.round(heatmap,decimals=2), cbar=False)# cbar_kws = {'ticks' :[-1, -0.5, 0 ,0.5, 1]})
    plt.xticks( fontsize=13,**csfont)
    plt.yticks( fontsize=13,**csfont)
    
    
    

    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    
    cbar = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.RdBu_r)

    cax = plt.axes([0.7, 0.5, 0.02, 0.3])
    
    cbar = plt.colorbar(cbar, cax= cax , orientation="vertical" , ticks = [-1,0,1] )#.set_ticklabels([3.13,8.58])
    cbar.set_ticklabels([-1,0,1])
    cbar.ax.tick_params(labelsize=15)
    # cbar.set_label("Annual Damage ($B)", fontsize=16, labelpad = -95,**csfont)
    
    
    # cbar = ax.collections[0].colorbar
    # cbar.ax.tick_params(labelsize=14)
    plt.savefig('Plots/Fig_3_heatmap_annual_{}.pdf'.format(k), bbox_inches='tight',dpi=250)
    # sns.pairplot(plot_df2 , corner=True , diag_kind= 'hist')
    # plt.savefig('Plots/Fig_6_Scatterplot_Matrix', bbox_inches='tight',dpi=250)
