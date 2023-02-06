# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 13:40:45 2021

@author: mzeigha
"""

import pandas as pd
import numpy as np


#Initialize lists to store average annual values in each region for each needed year
scenarios = ['all_tax' , 'CO2', 'no_tax', 'SNP']
stochastic_column = ['CA_load', 'PNW_Load','CA_Hydropower','PNW_Hydropower','Path66_flow','CA_Wind_Power','PNW_Wind_Power', 'Solar_Power']
# column_name = ['CO2_Damage', 'SNP_Damage', 'Shadow_Price','CA_load', 'PNW_Load','CA_Hydropower','PNW_Hydropower','Path66_flow','CA_Wind_Power','PNW_Wind_Power', 'Solar_Power']

PNW_wind , CA_wind , solar_daily, daily_load, CA_hydro, PNW_hydro, PNW_daily_load, CA_daily_load, Path66_flow = pd.DataFrame([]), pd.DataFrame([]), pd.DataFrame([]), pd.DataFrame([]), pd.DataFrame([]), pd.DataFrame([]), pd.DataFrame([]), pd.DataFrame([]), pd.DataFrame([]) 
CO2_damage, SNP_damage, Shadow_price = pd.DataFrame([], columns= scenarios), pd.DataFrame([], columns= scenarios), pd.DataFrame(np.zeros((365,4)), columns= scenarios)
CO2_damage_temp = SNP_damage_temp = pd.DataFrame(np.zeros((365,4)), columns= scenarios)
Shadow_price_temp, Shadow_price_df = pd.DataFrame([],index=None), pd.DataFrame([],index=None), 


#Loop through 100 sim years 
directory = [1]#,2,3, 4, 5]
# directory = [3]
# sim_years = 100


hour_100 = pd.RangeIndex(876000)
day = pd.RangeIndex(36500)
hour = pd.RangeIndex(8736)
# index = pd.date_range('1/1/2000', periods=876000, freq='H')
# index_shadow = pd.date_range('1/1/2000', periods=8736, freq='H')
# index_daily = pd.date_range('1/1/2000', periods=36500, freq='D')




# Reading the synthetic weather data ##
for i in directory:  
        
    try:
        # wind_temp = pd.read_csv('{}/Stochastic_engine/Synthetic_wind_power/wind_power_sim.csv'.format(str(i)), index_col=0)#.set_index (index)
        # wind_temp['hour'] = hour_100
        # # wind_temp['day'] = wind_temp['hour']//24
        # solar_temp = pd.read_csv('{}/Stochastic_engine/Synthetic_solar_power/solar_power_sim.csv'.format(str(i)), index_col=0)#.set_index (index)
        # solar_temp['hour'] = hour_100
        # solar_temp['day'] = solar_temp['hour']//24
        # CA_hydro_temp = pd.read_excel('{}/Stochastic_engine/CA_hydropower/CA_hydro_daily.xlsx'.format(str(i)), index_col=0)#.set_index (index_daily)
        # CA_hydro_temp['day'] = day
        # PNW_hydro_temp = pd.read_excel('{}/Stochastic_engine/PNW_hydro/PNW_hydro_daily.xlsx'.format(str(i)), index_col=0)#.set_index (index_daily)
        # PNW_hydro_temp['day'] = day
        load_path = pd.read_csv('{}/Stochastic_engine/Synthetic_demand_pathflows/Load_Path_Sim.csv'.format(str(i)), index_col=0)#.set_index (index_daily)
        load_path['day'] = day
        # hourly_load = pd.read_csv('{}/Stochastic_engine/Synthetic_demand_pathflows/Sim_hourly_load.csv'.format(str(i)), index_col=0)#.set_index (index)
        # hourly_load['hour'] = hour_100
        # hourly_load['day'] = hourly_load['hour']//24
            
    except:
        print('Year {} weather data not found'.format(str(100*i )))
    
  
    # PNW_wind_temp = pd.DataFrame(wind_temp.groupby(["hour"])['PNW'].sum())
    # CA_wind_temp = pd.DataFrame(wind_temp.groupby(["hour"])['CAISO'].sum())
    # solar_daily_temp = pd.DataFrame(solar_temp.groupby(["hour"])['CAISO'].sum())
    # daily_load_temp = hourly_load.groupby(["hour"]).sum()
    # PNW_daily_load_temp = pd.DataFrame(daily_load_temp.iloc[:,1])
    # CA_daily_load_temp = pd.DataFrame(daily_load_temp.iloc[:,2] + daily_load_temp.iloc[:,3] + daily_load_temp.iloc[:,4] + daily_load_temp.iloc[:,5])
    Path66_flow_temp = pd.DataFrame(load_path.groupby(["day"])['Path66_sim'].sum())
    # CA_hydro_temp_daily = pd.DataFrame(CA_hydro_temp.iloc[:,0] + CA_hydro_temp.iloc[:,1])
    

    # PNW_wind = PNW_wind.append (PNW_wind_temp, ignore_index=True)
    # CA_wind = CA_wind.append (CA_wind_temp, ignore_index=True)
    # solar_daily = solar_daily.append(solar_daily_temp, ignore_index=True)
    # PNW_daily_load = PNW_daily_load.append(PNW_daily_load_temp, ignore_index=True)
    # CA_daily_load = CA_daily_load.append(CA_daily_load_temp, ignore_index=True)
    Path66_flow = Path66_flow.append(Path66_flow_temp, ignore_index=True)
    # CA_hydro = CA_hydro.append(CA_hydro_temp_daily, ignore_index=True)
    # PNW_hydro = PNW_hydro.append(PNW_hydro_temp, ignore_index=True)
    


stochastic_df = pd.DataFrame([] , columns = stochastic_column)
stochastic_df.iloc[:,0] = CA_daily_load.iloc[:,0] 
stochastic_df.iloc[:,1] = PNW_daily_load.iloc[:,0] 
stochastic_df.iloc[:,2] = CA_hydro.iloc[:,0] 
stochastic_df.iloc[:,3] = PNW_hydro.iloc[:,0] 
stochastic_df.iloc[:,4] = Path66_flow.iloc[:,0] 
stochastic_df.iloc[:,5] = CA_wind.iloc[:,0]
stochastic_df.iloc[:,6] = PNW_wind.iloc[:,0]
stochastic_df.iloc[:,7] = solar_daily.iloc[:,0]
  
stochastic_df.to_csv (r'Stochastic_hourly.csv', index = False, header=True) 
    
#%%   
    
# # # ## Reading the CO2 and Local air damages data ##
# for i in directory:
#     for j in range (sim_years):
#         for s in scenarios:
#             try:
                
#                 CO2_damage_temp.loc[:,s] = pd.read_csv('{}/{}/CA{}/co2_damages.csv'.format(str(i), s, str(j)), index_col=0)
#                 SNP_damage_temp.loc[:,s] = pd.read_csv('{}/{}/CA{}/snp_damages.csv'.format(str(i), s, str(j)), index_col=0)
                
#                 shadow_temp = pd.read_csv('{}/{}/CA{}/shadow_price.csv'.format(str(i), s, str(j)), index_col=0)
#                 shadow_temp.loc[shadow_temp['Value'] >1000 , 'Value' ] = 1000
#                 shadow_hourly = shadow_temp.groupby(['Time'] , as_index=True).mean()
#                 shadow_hourly.index = index_shadow
#                 Shadow_price_temp = shadow_hourly.resample("D").mean()
#                 Shadow_price_temp.reset_index(drop=True, inplace=True)
#                 # Shadow_price_temp.loc [364,:] = Shadow_price_temp.loc [363,:]
#                 Shadow_price [s] = Shadow_price_temp
                
#             except:
#                 print('Damages for {} tax scenario of year {} was not found'.format( s, str(100*(i-1) +j )))  
            
#         CO2_damage_temp.loc [364,:] = CO2_damage_temp.loc [363,:]
#         SNP_damage_temp.loc [364,:] = SNP_damage_temp.loc [363,:]
#         Shadow_price.loc [364,:] = Shadow_price.loc [363,:]
        
#         CO2_damage = CO2_damage.append(CO2_damage_temp)
#         SNP_damage = SNP_damage.append(SNP_damage_temp)
#         Shadow_price_df = Shadow_price_df.append(Shadow_price)
    

        
# # CO2_damage.index = index_daily
# # SNP_damage.index = index_daily
# # Shadow_price_df.index = index_daily

# # CO2_damage = CO2_damage.resample('D').sum()
# # SNP_damage = SNP_damage.resample('D').sum()
# # Shadow_price_df = Shadow_price_df.resample('D').mean()
        
# CO2_damage.to_csv (r'CO2_damage.csv', index = False, header=True)    
# SNP_damage.to_csv (r'SNP_damage.csv', index = False, header=True)
# Shadow_price_df.to_csv (r'Shadow_price.csv', index = False, header=True)