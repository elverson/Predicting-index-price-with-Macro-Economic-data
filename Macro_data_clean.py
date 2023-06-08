# -*- coding: utf-8 -*-
"""
Created on Sun May 28 10:05:34 2023

@author: XuebinLi
"""

import pandas as pd
import numpy as np

def groupby_month(df):
    # Assuming your dataframe is named 'df' and the date column is named 'Date'
    
    # Convert 'Date' column to datetime type
    df['Date_'+data] = pd.to_datetime(df['Date_'+data])
    
    # Group by month and get the last day value
    df_grouped = df.groupby(df['Date_'+data].dt.to_period('M')).agg({data: 'first'}) 
    df_grouped = df_grouped.reset_index()
    df_grouped['Date_'+data] = df_grouped['Date_'+data].dt.to_timestamp()
        
    return df_grouped

def remove_comma(df):
    df['S&P500'] = df[data].str.replace(',','').astype(float)
    return df

def fill_data(df,t_fill):
    
    #fill up quarterly csv to monthly
    df = df.set_index('Date_'+data)
    df = df.resample('M').ffill()
    
    #replace . with Nan and forward fill
    #forward fill empty cell and then backward fill empty cell
    mask = df[data] != t_fill
    df[data] = np.where(mask, df[data], np.nan)
    df[data] = df[data].fillna(method='ffill')
    df[data] = df[data].fillna(method='bfill')
    # Fill missing values in the first two rows using forward fill with limit=1
    df[data].fillna(method='ffill', limit=1, inplace=True)
        
    return df

#for loop each list
list_data = ['CONSUMERCONFIDENCEINDEX','2YEARYIELD','10YEARYIELD'
              ,'CONSUMERSENTIMENT','CORECPI','COREPCE','CPI','GDP'
              ,'INTERESTRATE_MM','LABORFORCE_FORECAST'
              ,'MEDIANHOUSESALES','MONTHLYHOUSESUPPLY','UNEMPLOYMENTFORECAST'
              ,'NON_FARM','REAL_GDP_FORECAST','SHORTINTERESTRATE_FORECAST'
              ,'UNEMPLOYMENTRATE','CPI_FORECAST','S&P500']






# list_data = ['CONSUMERCONFIDENCEINDEX_MM_1960'
#               ,'CONSUMERSENTIMENT_MM_1952','CORECPI_MM_1960','COREPCE_MM_1959'
#               ,'MONTHLYHOUSESUPPLY_MM_1963','MONTHLYSUPPLYOFHOUSES_MM_1963'
#               ,'NON_FARM_MM_1939'
#               ,'UNEMPLOYMENTRATE_MM_1977','S&P500']

             
final_dataframe = pd.DataFrame()
for data in  list_data:   

    #read csv
    #group by Month if more than 546
    df = pd.read_csv(data+'.csv' ,index_col = None)
    #standardize column 1 name to 'Date'
    df.columns = ['Date_'+data,data]
    if(df['Date_'+data].count()>546):
        df = groupby_month(df)
    
    #standardize date format
    df['Date_'+data]= df['Date_'+data].apply(pd.to_datetime)
    df['Date_'+data] = pd.to_datetime(df['Date_'+data],format='%YYYY%mm%dd')
    df['Date_'+data] = df['Date_'+data].dt.strftime('%Y%m%d')
    df['Date_'+data] = df['Date_'+data].astype(int)
    
    #slice databased on date 19770101 to 20230101
    df = df[(df['Date_'+data]>=19770401) & (df['Date_'+data]<= 20220701)] 
    print(data)
    print(df['Date_'+data].count())
    
    df = df.reset_index(drop=True)
    
    #fill by monthly   
    #convert to datetime
    df['Date_'+data] = pd.to_datetime(df['Date_'+data], format='%Y%m%d')
    #df = df.drop_duplicates(subset=['Date_'+data])
    df = fill_data(df,'.')
    #save and overwrite csv
    df.to_csv('C:/Users/XuebinLi/OneDrive - Linden Shore LLC/Desktop/Macro Data_edit/cleaned/'+data+'new.csv')
    
    #concat all csv into 1
    final_dataframe = pd.concat([final_dataframe, df[data]],axis=1)
final_dataframe = remove_comma(final_dataframe)
final_dataframe.to_csv('C:/Users/XuebinLi/OneDrive - Linden Shore LLC/Desktop/Macro Data_edit/cleaned/cleaned_final.csv')








