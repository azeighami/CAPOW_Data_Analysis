# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 12:00:39 2021

@author: mzeigha
"""

import pandas as pd
import numpy as np


scenarios = ['all_tax' , 'CO2', 'no_tax', 'SNP']
generators = pd.read_csv('generator_emission.csv')
plant = pd.DataFrame([])
plant_annual = pd.DataFrame([])
emission = pd.DataFrame([])
Day = pd.RangeIndex(182000)
sim_years = 500


for s in scenarios:
    plant_annual = pd.DataFrame([])
    production = pd.read_csv('production_df_{}.csv'.format(s)).drop(columns = ["Year"])
    
#     for i in generators.index:
#         plant['generation'] = production.iloc[:,i]        
#         plant['year'] = Day//364
#         plant_temp = pd.DataFrame(np.array(plant.groupby(["year"])['generation'].sum()).reshape(1,500))
#         plant_annual = plant_annual.append(plant_temp, ignore_index= True)
    
#     # plants = pd.concat([generators,plant_annual] , axis = 1)

    
#     for i in range(sim_years):
#         for g in generators.index:
#             for p in ['SO2','NOX','PM']:
#                 if p == 'SO2':
#                     emission.loc[g,'{}lb{}'.format(p, str(i))] = generators.loc[g, 'SO2rate(lbs/MWh)'] * plant_annual.iloc[g,i]
#                 elif p == 'NOX':
#                     emission.loc[g,'{}lb{}'.format(p, str(i))] = generators.loc[g, 'NOXrate(lbs/MWh)'] * plant_annual.iloc[g,i]                    
#                 elif p == 'PM':
#                     emission.loc[g,'{}dol{}'.format(p, str(i))] = generators.loc[g, 'PMTax($/MWh)'] * plant_annual.iloc[g,i]                    

#     plants = pd.concat([generators,emission] , axis = 1)
    
#     if s == 'all_tax':
#         emission_all_tax = plants
#     elif s== 'CO2':
#         emission_CO2 = plants
#     elif s== 'no_tax':
#         emission_no_tax = plants
#     elif s== 'SNP':
#         emission_SNP = plants
        
# emission_all_tax.to_csv ('emission_all_tax.csv', index = False, header=True) 
# emission_CO2.to_csv ('emission_CO2.csv', index = False, header=True) 
# emission_no_tax.to_csv ('emission_no_tax.csv', index = False, header=True) 
# emission_SNP.to_csv ('emission_SNP.csv', index = False, header=True) 