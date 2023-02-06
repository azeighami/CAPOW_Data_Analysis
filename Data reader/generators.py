# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 12:37:57 2021

@author: mzeigha
"""


import pandas as pd
import numpy as np

scenarios = ['SNP']
#['no_tax','SNP','CO2', 'all_tax']
directory = [ 1]#, 2, 3, 4, 5]
sim_years = 1


generators = pd.read_csv('generators.csv', index_col=0)


Day = pd.RangeIndex(8736)


for s in scenarios:
    print ("scenario: " + str(s))
    production = pd.DataFrame([])
    production_df = pd.DataFrame([])
    for i in directory:
        print ("directory: " + str(i))        
        for j in range (sim_years):
            print ("sim year: " + str(j))
            
            mwh1 = pd.read_csv('{}/{}/CA{}/mwh_1.csv'.format(str(i), s, str(j)))
            mwh2 = pd.read_csv('{}/{}/CA{}/mwh_2.csv'.format(str(i), s, str(j)))
            mwh3 = pd.read_csv('{}/{}/CA{}/mwh_3.csv'.format(str(i), s, str(j)))
            
            
            mwh1_simp = mwh1.groupby(['Generator', 'Time'],as_index=True)['Value'].mean()
            mwh2_simp = mwh2.groupby(['Generator', 'Time'],as_index=True)['Value'].mean()
            mwh3_simp = mwh3.groupby(['Generator', 'Time'],as_index=True)['Value'].mean()
            
            mwh_simp = mwh1_simp + mwh2_simp + mwh3_simp
            
            for gen in ['AGRICO_6_PL3N5']:#generators.index:
                mwh_temp = pd.DataFrame(mwh_simp[gen])
                mwh_temp['Day'] = Day//24
                mwh_avg = mwh_temp.groupby(["Day"])['Value'].sum()
                production.loc[:, gen] = mwh_avg
                production['Year'] = (i-1)*100 + j
        
            production_df = pd.concat([production_df,production],axis = 0)
    
#     if s == 'no_tax':
#         production_df_no_tax = production_df
#     # elif s== 'CO2':
#     #     production_df_CO2 = production_df
#     # elif s== 'no_tax':
#     #     production_df_no_tax = production_df
#     else :
#         production_df_SNP = production_df
        
# # production_df_all_tax.to_csv (r'5years/production_df_all_tax.csv', index = False, header=True) 
# # production_df_CO2.to_csv (r'5years/production_df_CO2.csv', index = False, header=True) 
# production_df_no_tax.to_csv (r'5years/production_df_no_tax2.csv', index = False, header=True) 
# production_df_SNP.to_csv (r'5years/production_df_SNP2.csv', index = False, header=True) 