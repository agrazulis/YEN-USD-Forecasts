# YEN-USD-Forecasts

# General Overview
Script for a 1-month and 12-month ahead out-of-sample forecast for the YEN/USD. The forecasting model utilizes differentials between the two countries inflation rate, unemployment rate, interest rate and output. 

# Use Case
Research by Tanya Molodtsova and David Papell in their 2009 paper "Out-of-sample exchange rate predictability with Taylor rule fundamentals" shows that exchange rate predictability is much stronger with Taylor rule models than with conventional monetary, purchasing power parity or conventional interest rate models. I use this research as the basis for my YEN/USD forecasting models, which aim to beat the predictive accuracy of a random walk. More specifically, I use the gap (i.e., differentials) between Taylor Rule fundamentals in each country and write for loops to run our 1-month and 12-month ahead out of sample forecasts for the YEN/USD exchange rate. You can use this code as a launching off point for any exchange rate forecasting using the Taylor rule approach.

# Important Notes
1. The Taylor Rule fundamentals (i.e., variables) used are interest rates, inflation rates and output. As noted above, I take the difference of these variables between both countries in order to obtain our forecasts.
2. I plot the forecasted values over the actual values to provide a visulization of predictive accuracy. This also allows me to visualize the consistent differences between predicited and actual valuess, thus allowing for a quick understanding of where the model is failing.
3. I use the ordinary least squares (OLS) method for my forecasting. This comes from the StatsModels package which you can see imported at the top of my script.
