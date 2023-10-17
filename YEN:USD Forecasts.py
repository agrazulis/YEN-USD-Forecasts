#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 12 09:53:48 2021

@author: alexandergrazulis
"""

import os
os.getcwd()
os.chdir("/Users/alexandergrazulis/Documents/GitHub/YEN-USD-Forecasts/Supporting Data")

###############################################################################
########################## LOADING LIBRARIES ##################################
###############################################################################

import numpy as np # package for scientific computing
import pandas as pd # package for data manipulation
import os # package for communicating with operating system
import statsmodels.formula.api as smf # package for statistical models i.e. OLS
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

###############################################################################
############# CREATING INTEREST RATE DIFFERENTIALS FOR US/JAPAN ###############
###############################################################################

# Loading Japan data
df_japan = pd.read_csv("Japan_LIBOR.csv", header=0)

# Loading US data
df_us = pd.read_csv("US_LIBOR.csv", header=0)

# Loading exchange rate data
df_exch_jap = pd.read_csv("JAP_EXCH.csv", header=0)

# Dropping unnecessary columns
df_exch_jap = df_exch_jap[['dollar-yen-exchange-rate-historical-chart','Unnamed: 1','Unnamed: 2']]

# Only keeping rows from Jan. 4th 2000 to Feb. 8th 2021 
df_exch_jap = df_exch_jap.loc[7284:12804,:]

# Renaming columns in data frame
df_exch_jap.columns = ['Date', 'EX_JAP', 'EX_JAP_2']

# Turning exchange rate data into numerics
exch_cols = ['EX_JAP', 'EX_JAP_2']
df_exch_jap[exch_cols] = df_exch_jap[exch_cols].apply(pd.to_numeric, errors='coerce', axis=1)

# Merging two csv files 
df_japan = df_japan.merge(df_us)

#Renaming columns for japan/us data
df_japan.columns = ['Date', 'I_JAP', 'I_US']

# Creating empty column for interest rate differentials
df_japan['IDIFF_JAP'] = ""

# Turning interest rate columns into numerics
int_cols = ['I_JAP', 'I_US']
df_japan[int_cols] = df_japan[int_cols].apply(pd.to_numeric, errors='coerce', axis=1)

# Calculating interest rate differentials 
df_japan['IDIFF_JAP'] = df_japan['I_US'] - df_japan['I_JAP']

# Dropping Na rows
df_japan = df_japan.dropna()

# Merging exchange rates with data frame
df_japan = df_japan.merge(df_exch_jap)

# Making the date column a datetime object for future analysis 
df_japan['Date'] = pd.to_datetime(df_japan['Date'])

df_int_forecast_japan = (df_japan)

# Defining function to perform linear interpolation
def ffbf(x):
    return x.interpolate().ffill().bfill()

#################### CPI/IN_PROD/ UN_RATE(JAPAN) ##############################

# Loading in the new data
japan_data = pd.read_csv('JAPAN_LIBOR_CPI_UNRATE.CSV', header=0)
japan_data = japan_data.iloc[:,0:4] # Getting rid of the libor since we already input this earlier and keeping CPI, unemployment rate, and industrial production
new_header = japan_data.iloc[0] # Saving a new header variable with the names of our variables
japan_data.columns = new_header # Setting the new header with the variable names
japan_data.drop(japan_data.head(1).index, inplace = True) # Dropping the first row since it contains the headers that we set
# forward filling the unemployment rate and CPI
japan_data['CPI'] = pd.to_numeric(japan_data['CPI']) # Converting this to numeric so that I can use percent change function
japan_data['CPI'] = ffbf(japan_data['CPI'])
japan_data['CPI'] = japan_data['CPI'].pct_change(fill_method='ffill') # Forward filling to handle NaNs
japan_data.loc[1,'CPI'] = -0.0010070493 # Manually imputing the percent change for the first month 
#japan_data['CPI_change'] = japan_data['CPI'].pct_change
japan_data['CPI'] = japan_data['CPI'].replace(to_replace=0, method='ffill') # Replacing the 0 values with the previous non-zero value and forward filling
# Linearly interpolating the industrial production 
japan_data['IND_PROD'] = ffbf(japan_data['IND_PROD'])
# Setting the date as a time time object 
japan_data['Date'] = pd.to_datetime(japan_data['Date'])
# Merging the dataframes based on date
df_int_forecast_japan = pd.merge_asof(df_int_forecast_japan, japan_data, on='Date',direction = 'backward', allow_exact_matches=True)


############################### CPI (US) ######################################

cpi_growth_us = pd.read_csv('CPI_US.csv', header=0)
# Making the date column a datetime object for merging purposes
cpi_growth_us['Date'] = pd.to_datetime(cpi_growth_us['Date'])
# Merging the gdp data with the swiss data by date and filling all exact date matches and filling to the nearest date that comes after any missing dates (within a 1 day range)
df_int_forecast_japan = pd.merge_asof(df_int_forecast_japan, cpi_growth_us, on='Date',direction = 'backward', allow_exact_matches=True, tolerance=pd.Timedelta(days = 4))

# Instead of linear interpolation I will forward fill the days for each month with the value reported on the first day for each respective month 
df_int_forecast_japan['CPI_US'] = ffbf(df_int_forecast_japan['CPI_US'])
df_int_forecast_japan['CPI_US'] = df_int_forecast_japan['CPI_US'].pct_change(fill_method='ffill') # Calculating the percent change in the CPI for US and filling NaN values with zeros
df_int_forecast_japan.loc[0,'CPI_US'] = 0.0029620853 # Manually imputing the first percentage change value
df_int_forecast_japan['CPI_US'] = df_int_forecast_japan['CPI_US'].replace(to_replace=0, method='ffill') # Replacing the 0 values with the previous non-zero value and forward filling

###################### UN_RATE/IND_PROD (US) ##################################

us_data = pd.read_csv('US_IND_UNRATE.csv', header=0)
us_data['Date'] = pd.to_datetime(us_data['Date'])
df_int_forecast_japan = pd.merge_asof(df_int_forecast_japan, us_data, on='Date',direction = 'backward', allow_exact_matches=True, tolerance=pd.Timedelta(days = 4))
df_int_forecast_japan['IND_PROD_US'].fillna(method='ffill', inplace=True)
#df_int_forecast_japan['UN_RATE_US'].fillna(method='ffill', inplace=True)
#df_int_forecast_japan['UN_RATE'].fillna(method='ffill', inplace=True)

###############################################################################
############################# CLEANING THE DATA ###############################
###############################################################################

#Taking log of nominal exchange rates at time t
s_current = np.log(df_japan.loc[0:len(df_japan)-2,'EX_JAP_2']).reset_index(drop=True)
s_current = s_current.rename('s_current')

#Taking log of nominal exchange rates at time t+1
s_future = np.log(df_japan.loc[1:,'EX_JAP_2']).reset_index(drop=True)
s_future = s_future.rename('s_future')

#Finding the change in log of exchange rates at t+1 and t
s_change = s_future - s_current
s_change = s_change.rename('s_change')

## Independent Variables
# inf_us, inf_japan : us inflation at t, japan inflation at t
inf_japan = df_int_forecast_japan.loc[0:len(df_int_forecast_japan)-2,'CPI'].reset_index(drop=True)
inf_us = df_int_forecast_japan.loc[0:len(df_int_forecast_japan)-2,'CPI_US'].reset_index(drop=True)
inf_diff = pd.to_numeric(inf_us) - pd.to_numeric(inf_japan)
inf_diff = inf_diff.rename('inf_diff')

# output_us, output_japan : us output at t, japan output at t
output_japan = df_int_forecast_japan.loc[0:len(df_int_forecast_japan)-2,'IND_PROD'].reset_index(drop=True)
output_us = df_int_forecast_japan.loc[0:len(df_int_forecast_japan)-2,'IND_PROD_US'].reset_index(drop=True)

# Date: time at t+1
date = df_japan.loc[1:,'Date'].reset_index(drop=True)

#Interest rate differential between the US and japan (lagged)
interest_diff = df_japan.loc[0:len(df_japan)-2,'IDIFF_JAP'].reset_index(drop=True)
interest_diff = interest_diff.rename('int_diff')


# Saving new data set with all the variables of interest
japan_const = pd.concat([date, s_future, s_current, s_change, inf_diff, interest_diff, output_us, output_japan], axis=1)

japan_const['IND_PROD'] = pd.to_numeric(japan_const['IND_PROD'])

## Estimate of Output Gap
# Linear Trend
# Output Gap in Japan and US
japan_const['t'] = pd.DataFrame({'t' : range(1,len(japan_const)+1)})

for i in range(0,len(japan_const)): # Updating the potential output each period
    linear_trend_japan = smf.ols(formula = 'IND_PROD ~ t', data=japan_const.iloc[0+i]).fit()
    japan_const.loc[i,'japan_potential'] = linear_trend_japan.predict(japan_const['t'][0+i:1+i])[0+i]
    linear_trend_us = smf.ols(formula = 'IND_PROD_US ~ t', data=japan_const.iloc[0+i]).fit()
    japan_const.loc[i,'us_potential'] = linear_trend_us.predict(japan_const['t'][0+i:1+i])[0+i]
    
japan_const['japan_gap'] = (pd.to_numeric(japan_const['IND_PROD']) - japan_const['japan_potential'])/japan_const['japan_potential']
japan_const['us_gap'] = (pd.to_numeric(japan_const['IND_PROD_US']) - japan_const['us_potential'])/japan_const['us_potential']


# Output Gap differential between US and Japan
japan_const['gap_diff'] = japan_const['us_gap'] - japan_const['japan_gap']


# Keep variables only used for our model
japan_const=japan_const[['Date','s_future','s_current','s_change','inf_diff','int_diff', 'gap_diff']]

df_int_forecast_japan['UN_RATE'] = pd.to_numeric(df_int_forecast_japan['UN_RATE'])
df_int_forecast_japan['UN_RATE_US'] = ffbf(df_int_forecast_japan['UN_RATE_US'])
df_int_forecast_japan['UN_RATE'] = ffbf(df_int_forecast_japan['UN_RATE'])

japan_const['un_rate_us'] = df_int_forecast_japan['UN_RATE_US']
japan_const['un_rate_jap'] = df_int_forecast_japan['UN_RATE']
japan_const['un_diff'] = japan_const['un_rate_us'] - japan_const['un_rate_jap']

#Make sure all columns have float numbers
japan_const.loc[:,'s_future'] = japan_const.loc[:,'s_future'].apply(float)
japan_const.loc[:,'s_current'] = japan_const.loc[:,'s_current'].apply(float)
japan_const.loc[:,'s_change'] = japan_const.loc[:,'s_change'].apply(float)  
japan_const.loc[:,'int_diff'] = japan_const.loc[:,'int_diff'].apply(float) 
japan_const.loc[:,'inf_diff'] = japan_const.loc[:,'inf_diff'].apply(float) 
japan_const.loc[:,'gap_diff'] = japan_const.loc[:,'gap_diff'].apply(float)
japan_const['un_rate_jap'] = japan_const.loc[:,'un_rate_jap'].apply(float)
japan_const['un_rate_us'] = japan_const.loc[:,'un_rate_us'].apply(float)


###############################################################################
############# RUNNNING REGRESSIONS TO GENERATE 1-MONTH FORECASTS ##############
###############################################################################

# Creating empty columns for fitted values of changes in log exchange rates
japan_const['s_change_fitted_1-month'] = np.nan

# 1 month ahead out of sample forecast 
for i in range(22, len(japan_const)):
    tmp = smf.ols(formula = 's_change ~ inf_diff + gap_diff + int_diff + (un_rate_us - un_rate_jap)', data=japan_const[i-22:i]).fit()
    japan_const.loc[i,'s_change_fitted_1-month'] = tmp.predict(japan_const[['inf_diff','gap_diff','int_diff','un_rate_us','un_rate_jap']][i:i+22])[i]

# Saving forecasts
japan_const['s_forecast_1-month'] = japan_const['s_change_fitted_1-month'] + japan_const['s_current']

# Forecast error
japan_const['error_1-month'] = japan_const['s_future'] - japan_const['s_forecast_1-month']

# Plotting forecast
fig, ax = plt.subplots()
plt.plot(japan_const['Date'][3500:3750], japan_const['s_future'][3500:3750], label='Actual')
plt.plot(japan_const['Date'][3500:3750], japan_const['s_forecast_1-month'][3500:3750], label='Forecast')
plt.ylabel('Log of Nominal Exchange Rate')
plt.xlabel("Date")
plt.title('Actual vs Forecasted: YEN/USD 1-Month')
ax.xaxis.set_major_locator(mdates.MonthLocator(base=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%M'))
plt.legend()
plt.show()



fig, ax = plt.subplots()
plt.plot(japan_const['Date'][3500:3750], japan_const['un_diff'][3500:3750])
plt.ylabel('Unemployment Rate (%)')
plt.xlabel("Date")
plt.title('US Japan Unemployment Rate Differentials')
ax.xaxis.set_major_locator(mdates.MonthLocator(base=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%M'))
plt.show()

###############################################################################
############ RUNNNING REGRESSIONS TO GENERATE 12-MONTH FORECASTS ##############
###############################################################################

# Creating empty columns for fitted values of changes in log exchange rates
japan_const['s_change_fitted_12-month'] = np.nan

# 12 month ahead out of sample forecast 
for i in range(254, len(japan_const)):
    tmp = smf.ols(formula = 's_change ~ inf_diff + gap_diff + int_diff + (un_rate_us - un_rate_jap)', data=japan_const[i-254:i]).fit()
    japan_const.loc[i,'s_change_fitted_12-month'] = tmp.predict(japan_const[['inf_diff','gap_diff','int_diff','un_rate_us','un_rate_jap']][i:i+254])[i]

# Saving forecasts
japan_const['s_forecast_12-month'] = japan_const['s_change_fitted_12-month'] + japan_const['s_current']

# Forecast error
japan_const['error_12-month'] = japan_const['s_future'] - japan_const['s_forecast_12-month']

# Plotting forecast
plt.plot(japan_const.loc[3500:3750, 's_future'], color='black', label='Actual')
plt.plot(japan_const.loc[3500:3750, 's_forecast_12-month'], color='red', label='Forecast')
plt.ylabel('Log of 12-Month LIBOR')
plt.title('Actual vs Forecasted: US/JAP 12-Month')
plt.legend(loc='upper right')
plt.show()
