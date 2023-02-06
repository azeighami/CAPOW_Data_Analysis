# -*- coding: utf-8 -*-
"""
Created on Sat Oct  8 13:45:38 2022

@author: mzeigha
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import seaborn as sns
from textwrap import wrap
import matplotlib.colors as colors


production_no_tax = pd.read_csv('Results/production_df_no_tax.csv')
# production_SNP = pd.read_csv('Results/production_df_SNP.csv')

generators = pd.read_csv('Results/generators.csv',index_col=0 )
county_damage_per_generator = pd.read_csv('Crosswalk/county_damage_per_generator.csv', index_col=0).T


population = pd.read_csv('Demographics/white_population.csv')
population["non_white"] = (population['P1_001N']-population['P1_003N'])/population['P1_001N']

for i in range(len(population)):
    population.loc[i,'county'] = str(population.loc[i,'GEO_ID'])[-5:]    
population.index = population['county']



egrid = pd.read_csv('C:/Users/mzeigha/Documents/GitHub/CAPOW_Emission/Generator/eGRID Plants.csv', dtype = str)
gen = pd.read_csv('C:/Users/mzeigha/Documents/GitHub/CAPOW_Emission/Generator/generator_final_DOE.csv', index_col=('name'))

for i in gen.index:
    a = egrid[egrid['ORISPL']==str(gen.loc[i,'DOE'])]
        
    try:
        gen.loc[i,'county2'] = a.iloc[0,16] + a.iloc[0,17]
    except Exception:
        pass
        
generators = pd.concat([generators,gen['county2']], axis = 1)


for i in generators.index:

    if generators.loc[i,'zone'] == 'PGE_valley':
        generators.loc[i,'gas'] = 4.66
       
    elif generators.loc[i,'zone']=='PGE_bay':
        generators.loc[i,'gas'] = 4.66
       
    else:
        generators.loc[i,'gas'] = 4.47
        

generators['no_tax_cost'] =generators['gas']*generators['seg2']+generators['var_om']



generators = generators.sort_values(by = 'no_tax_cost')

M = pd.DataFrame([])
for i in generators.index:
    for j in population.index:
        if generators.loc[i, 'county2'] == str(j):
            M.loc[i,j] = generators.loc[i,'NOXTax($/MWh)'] + generators.loc[i,'SO2Tax($/MWh)'] + generators.loc[i,'PMTax($/MWh)']
        else:
            M.loc[i,j] = 0

no_tax= production_no_tax.groupby(['Year'])[generators.index].sum()
# SNP= production_SNP.groupby(['Year'])[generators.index].sum()


portion = ((county_damage_per_generator.T).loc[generators.index]).T

no_tax_damage = {}
# SNP_damage = {}
for i in range(500):
    no_tax_damage[i] = (portion*no_tax.loc[i])
    # SNP_damage[i] = (county_damage_per_generator*SNP.loc[i]).T
        

no_tax_damage_avg = (portion*no_tax.sum()/500).T
# SNP_damage_avg = county_damage_per_generator*SNP.sum()/500

           
# M2 = pd.DataFrame([])
# for i in generators.index:
#     for j in population.index:
#         if generators.loc[i, 'county2'] == str(j):
#             M2.loc[i,j] = no_tax_damage_avg.loc[i,j]
#         # else:
#         #     M2.loc[i,j] = 0


#%%
amir = pd.DataFrame([])
for i in range (500):
    amir.loc[i, 'sum'] = no_tax_damage[i].sum().sum()
            
#%%            

# M3 = pd.DataFrame([])
# for i in generators.index:
#     for j in population.index:
#         if generators.loc[i, 'county2'] == str(j):
#             M3.loc[i,j] = no_tax_damage[151].loc[i,j]
#         # else:
#         #     M3.loc[i,j] = 0
            
M2 = no_tax_damage_avg
M3 = no_tax_damage[151].T
M4 = M3-M2



xcsfont = {'fontname':'arial' , 'fontweight':'bold'}
csfont = {'fontname':'arial'}

fig = plt.figure( figsize = (15,10))


gs = fig.add_gridspec(4,1, width_ratios=[1], height_ratios = [1, 1, 1, 1] , wspace=0, hspace=0.3 )

############# Wind 
cf_ax1 = fig.add_subplot(gs[0,0])
cf_ax2 = fig.add_subplot(gs[1,0])
cf_ax3 = fig.add_subplot(gs[2,0])
cf_ax4 = fig.add_subplot(gs[3,0])

cmap1 = 'Reds'


A = pd.concat([M.T , population['non_white']], axis = 1)
A = A.sort_values(by = 'non_white')

ylabel = [np.round(A.iloc[0,-1],decimals=2), np.round(A.iloc[10,-1],decimals=2), np.round(A.iloc[20,-1],decimals=2), np.round(A.iloc[30,-1],decimals=2), np.round(A.iloc[40,-1],decimals=2),np.round(A.iloc[50,-1],decimals=2)]
yticks = [0,10,20,30,40,50]

xlabel = [np.round(generators.iloc[0,-1],decimals=2), np.round(generators.iloc[50,-1],decimals=2), np.round(generators.iloc[100,-1],decimals=2), np.round(generators.iloc[150,-1],decimals=2), np.round(generators.iloc[200,-1],decimals=2)]
xticks = [0,50,100,150,200]

A.index = A['non_white']
A = A.drop(columns = ['non_white'])

norm=colors.LogNorm(vmin = 1 , vmax = 100 )
cbar = plt.cm.ScalarMappable(norm=norm, cmap=cmap1)


a = cf_ax1.matshow(A, norm=norm, cmap=cmap1)
cf_ax1.set_yticks( yticks, ylabel ,fontsize=14, **csfont)
cf_ax1.set_xticks( xticks, xlabel, fontsize=14, **csfont)
cf_ax1.xaxis.set_label_position('top') 
cf_ax1.set_xlabel('a)',loc='left',labelpad = 6, fontsize=15, **xcsfont)
# cf_ax1.set_ylabel('SNP Tax',fontsize=16, **csfont)
cf_ax1.set_title('Power Plant (increasing Marginal Cost in $/MWh)',y = 1.2, fontsize=16, **csfont)
cf_ax1.axhline(32,color='black', linestyle='dotted')

cf_ax1.annotate('32 Majority white counties', (150,5),annotation_clip=False , fontsize=11,**csfont)
cf_ax1.annotate('26 Majority non-white counties', (140,37),annotation_clip=False , fontsize=11,**csfont)
cf_ax1.annotate('Non-White portion of County Population', (-32,210), rotation=90 , annotation_clip=False , fontsize=16,**csfont)

cax = plt.axes([0.72, 0.73, 0.02, 0.15])
cbar = plt.colorbar(a, cax= cax , orientation="vertical", ticks = [1,10,100] )#.set_ticklabels([3.13,8.58])
cbar.set_ticklabels([0, 10, 100])
cbar.ax.tick_params(labelsize=12)
cbar.set_label('\n'.join(wrap("Damage Rate ($/MWh)", 12)),labelpad=2,fontsize=14,**csfont)
# '\n'.join(wrap("Anomaly (C)", 12)),labelpad=15,fontsize=12,**csfont)                    




B = pd.concat([M2.T , population['non_white']], axis = 1)
B = B.sort_values(by = 'non_white')
B.index = B['non_white']
B = B.drop(columns = ['non_white'])
norm=colors.LogNorm(vmin=1, vmax=M3.max().max())
cbar = plt.cm.ScalarMappable(norm=norm, cmap=cmap1)


cf_ax2.matshow(B, norm = colors.LogNorm(), cmap=cmap1)
cf_ax2.set_yticks( yticks, ylabel ,fontsize=14, **csfont)
cf_ax2.set_xticks([])
# cf_ax2.set_ylabel('Average Year',fontsize=16, **csfont)
# cf_ax2.set_title('Net Cap',fontsize=16, **csfont)
cf_ax2.xaxis.set_label_position('top') 
cf_ax2.set_xlabel('b)',loc='left',labelpad = 6, fontsize=15, **xcsfont)




C = pd.concat([M3.T , population['non_white']], axis = 1)
C = C.sort_values(by = 'non_white')
C.index = C['non_white']
C = C.drop(columns = ['non_white'])

cf_ax3.matshow(C, norm = colors.LogNorm(), cmap=cmap1)
cf_ax3.set_yticks( yticks, ylabel ,fontsize=14, **csfont)
cf_ax3.set_xticks([])
# cf_ax3.set_ylabel('Worst Year',fontsize=16, **csfont)
# cf_ax3.title('Net Cap')
cf_ax3.xaxis.set_label_position('top') 
cf_ax3.set_xlabel('c)',loc='left',labelpad = 6, fontsize=15, **xcsfont)

cax2 = plt.axes([0.72, 0.33, 0.02, 0.34])
cbar = plt.colorbar(cbar, cax= cax2 , orientation="vertical")#, ticks = [1,100,200] )#.set_ticklabels([3.13,8.58])
# cbar.set_ticklabels([0, 10, 200])
cbar.ax.tick_params(labelsize=12)
cbar.set_label("Damage ($)",labelpad=2,fontsize=14,**csfont)







# norm= TwoSlopeNorm(vmin=M4.min().min(), vcenter=0, vmax=M4.max().max())


D = pd.concat([M4.T , population['non_white']], axis = 1)
D = D.sort_values(by = 'non_white')
D.index = D['non_white']
D = D.drop(columns = ['non_white'])

 # norm=colors.SymLogNorm
a = cf_ax4.matshow(D, norm = colors.LogNorm(), cmap=cmap1)
cf_ax4.set_yticks( yticks, ylabel ,fontsize=14, **csfont)
cf_ax4.set_xticks([])
# cf_ax4.set_ylabel('Disparity',fontsize=16, **csfont)
cf_ax4.xaxis.set_label_position('top') 
cf_ax4.set_xlabel('d)',loc='left',labelpad = 6, fontsize=15, **xcsfont)

# ylabel = [-10000, -100, 0 ,100,10000,1000000,100000000]
# yticks = [-10000, -100, 0 ,100,10000,1000000,100000000]

cax3 = plt.axes([0.72, 0.127, 0.02, 0.15])
cbar = plt.colorbar(a, cax= cax3 , orientation="vertical")#, ticks = [ 0, 100] )#.set_ticklabels([3.13,8.58])
# cbar.set_ticklabels(['-10000','','','','', '0','','','','','' ,'100000000'])
cbar.ax.tick_params(labelsize=13)
cbar.set_label("Damage ($)",labelpad=13,fontsize=14,**csfont)

plt.savefig('Plots/REVIEW_MP2_1.png' , bbox_inches='tight',dpi=250)


# #%%
# N2 = SNP_damage_avg.T
# N3 = SNP_damage[151]
# N4 = N3-N2


# csfont = {'fontname':'arial'}

# fig = plt.figure( figsize = (15,10))


# gs = fig.add_gridspec(4,1, width_ratios=[1], height_ratios = [1, 1, 1, 1] , wspace=0, hspace=0.1 )

# ############# Wind 
# cf_ax1 = fig.add_subplot(gs[0,0])
# cf_ax2 = fig.add_subplot(gs[1,0])
# cf_ax3 = fig.add_subplot(gs[2,0])
# cf_ax4 = fig.add_subplot(gs[3,0])




# A = pd.concat([M.T , population['non_white']], axis = 1)
# A = A.sort_values(by = 'non_white')

# ylabel = [np.round(A.iloc[0,-1],decimals=2), np.round(A.iloc[10,-1],decimals=2), np.round(A.iloc[20,-1],decimals=2), np.round(A.iloc[30,-1],decimals=2), np.round(A.iloc[40,-1],decimals=2),np.round(A.iloc[50,-1],decimals=2)]
# yticks = [0,10,20,30,40,50]

# xlabel = [np.round(generators.iloc[0,2],decimals=2), np.round(generators.iloc[50,2],decimals=2), np.round(generators.iloc[100,2],decimals=2), np.round(generators.iloc[150,2],decimals=2), np.round(generators.iloc[200,2],decimals=2)]
# xticks = [0,50,100,150,200]

# A.index = A['non_white']
# A = A.drop(columns = ['non_white'])

# norm=colors.LogNorm(vmin = 1 , vmax = A.max().max() )
# cbar = plt.cm.ScalarMappable(norm=norm, cmap='RdBu_r')


# cf_ax1.matshow(A, norm=colors.LogNorm(), cmap='RdBu_r')
# cf_ax1.set_yticks( yticks, ylabel ,fontsize=14, **csfont)
# cf_ax1.set_xticks( xticks, xlabel ,fontsize=14, **csfont)
# cf_ax1.set_ylabel('SNP Tax',fontsize=16, **csfont)
# cf_ax1.set_title('Net Cap',fontsize=16, **csfont)
# cf_ax1.axhline(32,color='black', linestyle='dotted')

# cf_ax1.annotate('32 Majority white counties', (160,5),annotation_clip=False , fontsize=11,**csfont)
# cf_ax1.annotate('26 Majority non-white counties', (150,37),annotation_clip=False , fontsize=11,**csfont)


# cax = plt.axes([0.75, 0.705, 0.02, 0.17])
# cbar = plt.colorbar(cbar, cax= cax , orientation="vertical", ticks = [1,10,200] )#.set_ticklabels([3.13,8.58])
# cbar.set_ticklabels([0, 10, 200])
# cbar.ax.tick_params(labelsize=15)
                    
                    

# B = pd.concat([N2.T , population['non_white']], axis = 1)
# B = B.sort_values(by = 'non_white')
# B.index = B['non_white']
# B = B.drop(columns = ['non_white'])
# norm=colors.LogNorm(vmin=1, vmax=M3.max().max())
# cbar = plt.cm.ScalarMappable(norm=norm, cmap='RdBu_r')


# cf_ax2.matshow(B, norm = colors.LogNorm(), cmap='RdBu_r')
# cf_ax2.set_yticks( yticks, ylabel ,fontsize=14, **csfont)
# cf_ax2.set_xticks([])
# cf_ax2.set_ylabel('Average Year',fontsize=16, **csfont)
# # cf_ax2.title('Net Cap')



# C = pd.concat([N3.T , population['non_white']], axis = 1)
# C = C.sort_values(by = 'non_white')
# C.index = C['non_white']
# C = C.drop(columns = ['non_white'])

# cf_ax3.matshow(C, norm = colors.LogNorm(), cmap='RdBu_r')
# cf_ax3.set_yticks( yticks, ylabel ,fontsize=14, **csfont)
# cf_ax3.set_xticks([])
# cf_ax3.set_ylabel('Worst Year',fontsize=16, **csfont)
# # cf_ax3.title('Net Cap')

# cax2 = plt.axes([0.75, 0.33, 0.02, 0.35])
# cbar = plt.colorbar(cbar, cax= cax2 , orientation="vertical")#, ticks = [1,100,200] )#.set_ticklabels([3.13,8.58])
# # cbar.set_ticklabels([0, 10, 200])
# cbar.ax.tick_params(labelsize=15)


# norm= TwoSlopeNorm(vmin=N4.min().min(), vcenter=0, vmax=N4.max().max())
# cbar = plt.cm.ScalarMappable(norm=norm, cmap='RdBu_r')
# D = pd.concat([N4.T , population['non_white']], axis = 1)
# D = D.sort_values(by = 'non_white')
# D.index = D['non_white']
# D = D.drop(columns = ['non_white'])


# cf_ax4.matshow(D, norm = colors.LogNorm(), cmap='RdBu_r')
# cf_ax4.set_yticks( yticks, ylabel ,fontsize=14, **csfont)
# cf_ax4.set_xticks([])
# cf_ax4.set_ylabel('Worst Year',fontsize=16, **csfont)


# cax3 = plt.axes([0.75, 0.125, 0.02, 0.17])
# cbar = plt.colorbar(cbar, cax= cax3 , orientation="vertical")#, ticks = [1,10,200] )#.set_ticklabels([3.13,8.58])
# # cbar.set_ticklabels([0, 10, 200])
# cbar.ax.tick_params(labelsize=15)

# plt.savefig('Plots/REVIEW2_SNP.png' , bbox_inches='tight',dpi=250)
