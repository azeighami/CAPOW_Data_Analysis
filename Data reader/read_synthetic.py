 # -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 14:03:40 2021

@author: mzeigha
"""

import pandas as pd


directory = [1, 2, 3, 4, 5]
Day = pd.RangeIndex(37960)

stream_annual ,wind_annual , temp_annual ,irradiance_annual= pd.DataFrame([]), pd.DataFrame([]),pd.DataFrame([]), pd.DataFrame([])

for i in directory:  
        
    FCRPS_stream = pd.read_csv('{}/Stochastic_engine/Synthetic_streamflows/synthetic_streamflows_FCRPS.csv'.format(str(i)),header=None)
    Willamette_stream = pd.read_csv('{}/Stochastic_engine/Synthetic_streamflows/synthetic_streamflows_Willamette.csv'.format(str(i)), index_col=0)
    CA_stream = pd.read_csv('{}/Stochastic_engine/Synthetic_streamflows/synthetic_streamflows_CA.csv'.format(str(i)), index_col=0)
    irradiance = pd.read_csv('{}/Stochastic_engine/Synthetic_weather/synthetic_irradiance_data.csv'.format(str(i)), index_col=0)#.set_index (index_daily)
    weather = pd.read_csv('{}/Stochastic_engine/Synthetic_weather/synthetic_weather_data.csv'.format(str(i)), index_col=0)#.set_index (index_daily)

    wind = weather.drop(['SALEM_T','EUGENE_T','SEATTLE_T','BOISE_T','PORTLAND_T','SPOKANE_T','PASCO_T','FRESNO_T','LOS ANGELES_T','SAN DIEGO_T','SACRAMENTO_T','SAN JOSE_T','SAN FRANCISCO_T','TUCSON_T','PHOENIX_T','LAS VEGAS_T','OAKLAND_T'], axis = 1)
    temp = weather.drop(['SALEM_W','EUGENE_W','SEATTLE_W','BOISE_W','PORTLAND_W','SPOKANE_W','PASCO_W','FRESNO_W','LOS ANGELES_W','SAN DIEGO_W','SACRAMENTO_W','SAN JOSE_W','SAN FRANCISCO_W','TUCSON_W','PHOENIX_W','LAS VEGAS_W','OAKLAND_W'], axis = 1)
    
    stream = pd.concat ([FCRPS_stream, Willamette_stream, CA_stream.iloc[:, 0:15]], axis = 1)
    stream["Year"] = Day//365

    
    irradiance['Year'] = Day//365    
    
    wind.loc[:,'Year'] = Day//365    
    
    temp.loc[:,'Year'] = Day//365    


    # stream_temp = stream.groupby(['Year']).mean()
    # wind_temp = wind.groupby(['Year']).mean()
    # temp_temp = (temp.groupby(['Year']).mean())
    # irradiance_temp = irradiance.groupby(['Year']).mean()
    
    
    # stream_temp = stream_temp.drop([0,101,102,103])
    # wind_temp = wind_temp.drop([0,101,102,103])
    # temp_temp = temp_temp.drop([0,101,102,103])
    # irradiance_temp = irradiance_temp.drop([0,101,102,103])
   
    stream_temp = stream[~stream['Year'].isin([0,101,102,103])]
    wind_temp = wind[~wind['Year'].isin([0,101,102,103])]
    temp_temp = temp[~temp['Year'].isin([0,101,102,103])]
    irradiance_temp = irradiance[~irradiance['Year'].isin([0,101,102,103])]

    
    stream_annual = pd.concat ([stream_annual, stream_temp], axis = 0)
    wind_annual = pd.concat([wind_annual,wind_temp],axis = 0)
    temp_annual = pd.concat([temp_annual,temp_temp],axis=0)
    irradiance_annual = pd.concat([irradiance_annual,irradiance_temp], axis = 0)
    
    
stream_annual.to_csv (r'stream_annual.csv', index = False, header=True)   
wind_annual.to_csv (r'wind_annual.csv', index = False, header=True)   
temp_annual.to_csv (r'temp_annual.csv', index = False, header=True)   
irradiance_annual.to_csv (r'irradiance_annual.csv', index = False, header=True)   





