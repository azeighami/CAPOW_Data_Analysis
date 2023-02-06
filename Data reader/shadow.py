# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 17:58:36 2021

@author: mzeigha
"""


import pandas as pd
import numpy as np



scenarios = ['all_tax' , 'CO2', 'no_tax', 'SNP']
directory = [1, 2, 3, 4, 5]
# directory = [3]
sim_years = 100
# slack = 0

Shadow_price = pd.DataFrame([], columns= scenarios)#,index=None)
Shadow_price_df = pd.DataFrame([])#,index=None)
a = 0
for i in directory:
    for j in range (sim_years):
        for s in scenarios:
                
            shadow_temp = pd.read_csv('{}/{}/CA{}/shadow_price.csv'.format(str(i), s, str(j)), index_col=0)
            shadow_temp.loc[shadow_temp['Value'] >1000 , 'Value' ] = 10000
            shadow_hourly = shadow_temp.groupby(['Time'] , as_index=False).mean()
            shadow_hourly = shadow_hourly.drop(columns = ['Time'])
            shadow_hourly.reset_index()#(drop=True, inplace=True)

            Shadow_price [s] = shadow_hourly.loc[:,'Value']

        Shadow_price_df = Shadow_price_df.append(Shadow_price)            
            