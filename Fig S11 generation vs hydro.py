# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 20:18:07 2022

@author: mzeigha
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import seaborn as sns
from textwrap import wrap



production_no_tax = pd.read_csv('Results/production_df_no_tax.csv')
production_SNP = pd.read_csv('Results/production_df_SNP.csv')
stochastic_df = pd.read_csv('Results/Stochastic_df.csv')
generators = pd.read_csv('Results/generators.csv',index_col=0 )
county_damage_per_generator = pd.read_csv('Crosswalk/county_damage_per_generator.csv', index_col=0).T


stochastic_df['Day'] = pd.RangeIndex(182500)
stochastic_df["Year"] = stochastic_df["Day"]//365
stochastic_df['DOY'] = stochastic_df['Day']-stochastic_df['Year']*365

stochastic_df = stochastic_df[stochastic_df['DOY'] != 364 ]

population = pd.read_csv('Demographics/white_population.csv')
population["non_white"] = (population['P1_001N']-population['P1_003N'])/population['P1_001N']

for i in range(len(population)):
    population.loc[i,'county'] = str(population.loc[i,'GEO_ID'])[-5:]    
population.index = population['county']

#%%
annual_gen_no_tax = production_no_tax.groupby('Year')[generators.index].sum()
annual_gen_SNP = production_SNP.groupby('Year')[generators.index].sum()
annual_stochastic = stochastic_df.groupby('Year')['CA_load' ,'CA_Hydropower' ,'PNW_Hydropower', 'Path66_flow'].sum()

corr_no_tax = pd.concat([annual_gen_no_tax,annual_stochastic], axis= 1).corr().drop(columns = generators.index , index = ['CA_load' ,'CA_Hydropower' ,'PNW_Hydropower', 'Path66_flow'])
corr_SNP = pd.concat([annual_gen_SNP,annual_stochastic], axis= 1).corr().drop(columns = generators.index , index = ['CA_load' ,'CA_Hydropower' ,'PNW_Hydropower', 'Path66_flow'])
corr_diff = corr_no_tax - corr_SNP

corr_no_tax.columns = ['CA_load_no_tax', 'CA_Hydropower_no_tax', 'PNW_Hydropower_no_tax', 'Path66_flow_no_tax']
corr_SNP.columns = ['CA_load_SNP', 'CA_Hydropower_SNP', 'PNW_Hydropower_SNP', 'Path66_flow_SNP']
corr_diff.columns = ['CA_load_diff', 'CA_Hydropower_diff', 'PNW_Hydropower_diff', 'Path66_flow_diff']



#%%
csfont = {'fontname':'arial'}

generators2 = pd.read_csv('Results/generators_no_tax.csv',index_col=0 )

    
df = pd.concat([generators[['netcap','var_om','NOXrate(lbs/MWh)']],generators2[['seg1','seg2','seg3','zone']]],axis = 1)
df['tax'] = generators['NOXTax($/MWh)'] + generators['SO2Tax($/MWh)'] + generators['PMTax($/MWh)']

for i in generators.index:
    df.loc[i,'CF_no_tax'] = (production_no_tax.sum()[i]/500)/(generators.loc[i,'netcap']*24*364)
    df.loc[i,'CF_SNP'] = (production_SNP.sum()[i]/500)/(generators.loc[i,'netcap']*24*364)
    if df.loc[i,'zone'] == 'PGE_valley':
        df.loc[i,'gas'] = 4.66
       
    elif df.loc[i,'zone']=='PGE_bay':
        df.loc[i,'gas'] = 4.66
       
    else:
        df.loc[i,'gas'] = 4.47
        
df['CF_diff'] = df['CF_SNP'] - df['CF_no_tax']

df['no_tax_cost'] =df['gas']*df['seg2']+df['var_om']
df['SNP_cost'] = df['no_tax_cost'] + df['tax']



no_tax_damage = county_damage_per_generator*production_no_tax[generators.index].sum()/500
SNP_damage = county_damage_per_generator*production_SNP[generators.index].sum()/500

no_tax_damage["non_white"] = population["non_white"]
SNP_damage["non_white"] = population["non_white"]

# df['non_white_SNP'] = SNP_damage["non_white"]*SNP_damage[SNP_damage["non_white"] > 0.5 ].drop(columns = "non_white").sum(axis =1)/23242101
# df['non_white_no_tax'] = no_tax_damage["non_white"]*no_tax_damage[no_tax_damage["non_white"] > 0.5 ].drop(columns = "non_white").sum(axis =1)/23242101

df['non_white_SNP'] = SNP_damage[SNP_damage["non_white"] > 0.5 ].drop(columns = "non_white").sum()/SNP_damage.drop(columns = "non_white").sum()
df['non_white_no_tax'] = no_tax_damage[no_tax_damage["non_white"] > 0.5 ].drop(columns = "non_white").sum()/no_tax_damage.drop(columns = "non_white").sum()


#%%
black_no_tax_2 = pd.DataFrame()
black_SNP_2 = pd.DataFrame()
for i in no_tax_damage.index:
    for j in generators.index:
        
        black_no_tax_2.loc[i,j] = (no_tax_damage.loc[i,j]*no_tax_damage.loc[i,'non_white'])/(population['P1_001N'].sum()-population['P1_003N'].sum())
        # white_no_tax_2.loc[i,j] = (no_tax_damage.loc[i,j]*(1-no_tax_damage.loc[i,'non_white']))/(population['P1_003N'].sum())

        black_SNP_2.loc[i,j] = (SNP_damage.loc[i,j]*SNP_damage.loc[i,'non_white'])/(population['P1_001N'].sum()-population['P1_003N'].sum())
        # white_SNP_2.loc[i,j] = (SNP_damage.loc[i,j]*(1-SNP_damage.loc[i,'non_white']))/(population['P1_003N'].sum())



df['non_white_SNP2'] = black_SNP_2.sum()
df['non_white_no_tax2'] = black_no_tax_2.sum()

df = df.drop(columns = ['var_om','NOXrate(lbs/MWh)','seg1','seg2','seg3','zone','gas'])
df = pd.concat([df,corr_no_tax,corr_SNP ,corr_diff], axis = 1)

#%%
from textwrap import wrap


fig = plt.figure( figsize = (20,10),constrained_layout=True)

gs = fig.add_gridspec(2,3, width_ratios=[1,1,1], height_ratios = [1,1] ,hspace=0.1,wspace=0.05)

#############  

cf_ax1 = fig.add_subplot(gs[0,0])
cf_ax2 = fig.add_subplot(gs[0,1])
cf_ax3 = fig.add_subplot(gs[0,2])

cf_ax4 = fig.add_subplot(gs[1,0])
cf_ax5 = fig.add_subplot(gs[1,1])
cf_ax6 = fig.add_subplot(gs[1,2])




class nlcmap(object):
    def __init__(self, cmap, levels):
        self.cmap = cmap
        self.N = cmap.N
        self.monochrome = self.cmap.monochrome
        self.levels = np.asarray(levels, dtype='float64')
        self._x = self.levels
        self.levmax = self.levels.max()
        self.transformed_levels = np.linspace(0.0, self.levmax,
              len(self.levels))

    def __call__(self, xi, alpha=1.0, **kw):
        yi = np.interp(xi, self._x, self.transformed_levels)
        return self.cmap(yi / self.levmax, alpha)


levels = [0.72, 4, 6, 8, 10, 12, 14, 16, 18, 100, 168]

cmap_nonlin = nlcmap(plt.cm.Reds, levels)


csfont_tick = {'labelsize':22}
csfont_title = {'fontname':'Arial','size':'26'}


df =df.loc[generators.index]
df2 = df#[abs(df['CF_diff'])>=0.01]
df2 = df2.sort_values('no_tax_cost', ascending = True )

cf_ax1.scatter(df2.index , df2['CF_no_tax'], c = cmap_nonlin(df2['tax']), s=df2['netcap'], alpha = 1,edgecolor='black',linewidth=0.25)
cf_ax1.set_xticks([])
cf_ax1.tick_params(**csfont_tick)
cf_ax1.set_ylabel('No Tax',**csfont_title)
cf_ax1.set_title('Capacity Factor',pad = 30, **csfont_title)
# cf_ax1.set_ylim([-1,1])



cf_ax2.scatter( df2.index , df2['CA_Hydropower_no_tax'], c = cmap_nonlin(df2['tax']), s=df2['netcap'], alpha = 1,edgecolor='black',linewidth=0.25)
cf_ax2.set_xticks([])
cf_ax2.set_title('\n'.join(wrap('Pearson R Corrleation between Capacity Factor and CA Hydropower',35)),pad = 20,**csfont_title)
# cf_ax2.set_ylabel('No Tax generation VS CA Hydropower',fontsize=16)
cf_ax2.set_ylim([-1,0.6])
cf_ax2.tick_params(**csfont_tick)


cf_ax3.scatter( df2.index , df2['non_white_no_tax'], c = cmap_nonlin(df2['tax']), s=df2['netcap'], alpha = 1,edgecolor='black',linewidth=0.25)
cf_ax3.set_xticks([])
# cf_ax3.set_ylabel('No Tax',fontsize=16)
cf_ax3.set_title('\n'.join(wrap('Portion of Damages to Majority Non-white Counties', 30)),pad = 20,**csfont_title)
# cf_ax3.set_ylim([-1,0.5])
cf_ax3.tick_params(**csfont_tick)

df2 = df2.sort_values('SNP_cost', ascending = True )

cf_ax4.scatter(df2.index , df2['CF_SNP'], c = cmap_nonlin(df2['tax']), s=df2['netcap'], alpha = 1,edgecolor='black',linewidth=0.25)
cf_ax4.set_xticks([])
cf_ax4.set_ylabel('With Local Tax',**csfont_title)
# cf_ax1.set_title('Capacity Factor',fontsize=16)
# cf_ax1.set_ylim([-1,1])
cf_ax4.tick_params(**csfont_tick)


cf_ax5.scatter( df2.index , df2['CA_Hydropower_SNP'], c = cmap_nonlin(df2['tax']), s=df2['netcap'], alpha = 1,edgecolor='black',linewidth=0.25)
cf_ax5.set_xticks([])
# cf_ax5.set_title('Power Generation vs. CA Hydropower',fontsize=16)
# cf_ax5.set_ylabel('SNP generation VS CA Hydropower',fontsize=16)
cf_ax5.set_ylim([-1,0.6])
cf_ax5.tick_params(**csfont_tick)


cf_ax6.scatter( df2.index , df2['non_white_SNP'], c = cmap_nonlin(df2['tax']), s=df2['netcap'], alpha = 1,edgecolor='black',linewidth=0.25)
cf_ax6.set_xticks([])
# cf_ax6.set_ylabel('SNP tax',fontsize=16)
# cf_ax3.set_title('Portion of Damages to none white counties',fontsize=16)
# cf_ax3.set_ylim([-1,1])
cf_ax6.tick_params(**csfont_tick)

cf_ax1.annotate('a',(200,0.2),annotation_clip=False ,  fontsize=40,**csfont,weight='bold')
cf_ax2.annotate('b',(200,-0.70),annotation_clip=False ,  fontsize=40,**csfont,weight='bold')
cf_ax3.annotate('c',(200,.65),annotation_clip=False,  fontsize=40,**csfont,weight='bold')
cf_ax4.annotate('d',(200,0.2),annotation_clip=False ,  fontsize=40,**csfont,weight='bold')
cf_ax5.annotate('e',(200,-0.70),annotation_clip=False ,  fontsize=40,**csfont,weight='bold')
cf_ax6.annotate('f',(200,.65),annotation_clip=False ,  fontsize=40,**csfont,weight='bold')



sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, 
                norm=plt.Normalize(vmin=0, vmax=df['tax'].max()))
sm._A = []
cbar = plt.colorbar(sm ,cax = plt.axes([1.01, 0.2, 0.02, 0.5]), pad = 0.1 , shrink= 0.9 , aspect=11 )#, ticks=[vmin,vcenter,vmax])
cbar.ax.tick_params(labelsize=16)
cbar.set_ticks(np.linspace(0.0, 268, 5))
cbar.set_ticklabels([.72, 6 ,12,16,268], fontsize = 22)
cbar.set_label('Local Emissions Tax ($/MWh)',labelpad = 10, **csfont_title)


# cbar = plt.colorbar(mappable=plt.cm.ScalarMappable(norm=norm, cmap= cmap), cax = plt.axes([1.01, 0.3, 0.015, 0.4]) ,orientation='vertical', pad = 0.05 , shrink= 1 , aspect=11 )
# cbar.set_label('Change in Capacity Factor',fontsize = 14)

plt.figtext(0.26,-0.04,'Lower Marginal cost ----------------------> Higher Marginal cost', fontdict = {"fontname":"Arial",'size':28})


plt.savefig('Plots/Fig_S11.png' , bbox_inches='tight',dpi=250)

