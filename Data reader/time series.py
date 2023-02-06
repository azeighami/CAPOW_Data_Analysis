# -*- coding: utf-8 -*-
"""
Created on Mon May 17 18:10:10 2021

@author: mzeigha
"""

import pandas as pd
import numpy as np



directory = [1, 2, 3, 4, 5]
Day = pd.RangeIndex(37960)
drop = [0,101,102,103]

stream_annual ,wind_annual , temp_annual ,irradiance_annual= pd.DataFrame([]), pd.DataFrame([]),pd.DataFrame([]), pd.DataFrame([])
stream_temp ,wind_temp , temp_temp ,irradiance_temp= pd.DataFrame([]), pd.DataFrame([]),pd.DataFrame([]), pd.DataFrame([])

##### np.zeros((37960,3)),columns = ['avg','Year','DOY']

for i in directory:  

    stream_temp ,wind_temp , temp_temp ,irradiance_temp= pd.DataFrame([]), pd.DataFrame([]),pd.DataFrame([]), pd.DataFrame([])    

    FCRPS_stream = pd.read_csv('{}/Stochastic_engine/Synthetic_streamflows/synthetic_streamflows_FCRPS.csv'.format(str(i)),header=None)
    Willamette_stream = pd.read_csv('{}/Stochastic_engine/Synthetic_streamflows/synthetic_streamflows_Willamette.csv'.format(str(i)), index_col=0)
    CA_stream = pd.read_csv('{}/Stochastic_engine/Synthetic_streamflows/synthetic_streamflows_CA.csv'.format(str(i)), index_col=0)
    irradiance = pd.read_csv('{}/Stochastic_engine/Synthetic_weather/synthetic_irradiance_data.csv'.format(str(i)), index_col=0)
    weather = pd.read_csv('{}/Stochastic_engine/Synthetic_weather/synthetic_weather_data.csv'.format(str(i)), index_col=0)

    wind = weather.drop(['SALEM_T','EUGENE_T','SEATTLE_T','BOISE_T','PORTLAND_T','SPOKANE_T','PASCO_T','FRESNO_T','LOS ANGELES_T','SAN DIEGO_T','SACRAMENTO_T','SAN JOSE_T','SAN FRANCISCO_T','TUCSON_T','PHOENIX_T','LAS VEGAS_T','OAKLAND_T'], axis = 1)
    temp = weather.drop(['SALEM_W','EUGENE_W','SEATTLE_W','BOISE_W','PORTLAND_W','SPOKANE_W','PASCO_W','FRESNO_W','LOS ANGELES_W','SAN DIEGO_W','SACRAMENTO_W','SAN JOSE_W','SAN FRANCISCO_W','TUCSON_W','PHOENIX_W','LAS VEGAS_W','OAKLAND_W'], axis = 1)
    
    stream = pd.concat ([FCRPS_stream, Willamette_stream, CA_stream.iloc[:, 0:15]], axis = 1)
    stream_temp['avg'] = stream.mean(axis=1)
    stream_temp["Year"] = Day//365
    stream_temp["DOY"] = Day - (stream_temp["Year"])*365
    
    irradiance_temp['avg'] = irradiance.mean(axis=1)
    irradiance_temp['Year'] = Day//365
    irradiance_temp["DOY"] = Day - irradiance_temp["Year"]*365
    
    wind_temp['avg'] = wind.mean(axis=1)
    wind_temp['Year'] = Day//365  
    wind_temp["DOY"] = Day - wind_temp["Year"]*365
    
    temp_temp['avg'] = temp.mean(axis=1)
    temp_temp['Year'] = Day//365
    temp_temp["DOY"] = Day - temp_temp["Year"]*365 
    
    
    irradiance_temp = irradiance_temp[~irradiance_temp.Year.isin([0,101,102,103])]
    stream_temp = stream_temp[~stream_temp.Year.isin([0,101,102,103])]
    wind_temp = wind_temp[~wind_temp.Year.isin([0,101,102,103])]
    temp_temp = temp_temp[~temp_temp.Year.isin([0,101,102,103])]


    stream_temp["Year"] = stream_temp["Year"]+(i-1)*100   
    irradiance_temp['Year'] = irradiance_temp['Year']+(i-1)*100   
    wind_temp['Year'] =wind_temp['Year']+(i-1)*100   
    temp_temp['Year'] = temp_temp['Year'] +(i-1)*100  
    
    
    
    stream_annual = pd.concat ([stream_annual, stream_temp], axis = 0)
    wind_annual = pd.concat([wind_annual,wind_temp],axis = 0)
    temp_annual = pd.concat([temp_annual,temp_temp],axis=0)
    irradiance_annual = pd.concat([irradiance_annual,irradiance_temp], axis = 0)
    
    
stream_annual.to_csv (r'a/stream_annual.csv', index = False, header=True)   
wind_annual.to_csv (r'a/wind_annual.csv', index = False, header=True)   
temp_annual.to_csv (r'a/temp_annual.csv', index = False, header=True)   
irradiance_annual.to_csv (r'a/irradiance_annual.csv', index = False, header=True)   