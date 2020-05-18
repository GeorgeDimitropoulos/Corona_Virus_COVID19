# %% Import libraries

import numpy as np
import pandas as pd
import csv
import os
import glob
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import statistics
import math

# %% read data

import pandas as pd
df = pd.read_csv('../data/time_series_covid_19_confirmed.csv')

# %% focus on data regarding Netherlands

df_nl=df[df["Province/State"]=="Netherlands"]

# %% transform data into a "regular" pd dataframe format

df2 = pd.melt(df_nl, id_vars=["Country/Region"], 
                  var_name="Date", value_name="Confirmed Cases")

# %% clean data

df2=df2[3:]

# %% reset index of dataframe 

df2.reset_index()

# %% add data of current day (do not exist on initial data)

df2.at[64,"Country/Region"]="Netherlands"
df2.at[64,"Date"]="3/23/20"
df2.at[64,"Confirmed Cases"]=4749

# %% make date a datetime object

df2["date"] = pd.to_datetime(df2["Date"])

# %% remove previous column of date

df2.drop(['Date'], axis=1)

# %% set date as an index

df2=df2.set_index('date')

#%% function to make lineplot of signal over time

def lineplot_signal_in_time(signal,dataframe):
    sns.set(rc={'figure.figsize':(11, 4)})
    fig=dataframe[signal].plot(linewidth=0.5)
    fig.set_xlabel("DateTime")
    fig.set_ylabel(signal)
    figure = fig.get_figure() 
    return figure

# %% plot confirmed cases signal over time

figure=lineplot_signal_in_time("Confirmed Cases",df2)

# %% find optimal values for p and q for the ARIMA model

import itertools
p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))


# %% obtain report of the model

y=df2["Confirmed Cases"]
import statsmodels.api as sm
mod = sm.tsa.statespace.SARIMAX(y.astype(float),
                                order=(1, 1, 1),
                                seasonal_order=(1, 1, 0, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = mod.fit()
print(results.summary().tables[1])


# %% obtain diagnostic plots

results.plot_diagnostics(figsize=(16, 8))
plt.show()

# %% check prediction capability of the model on known data

pred = results.get_prediction(start=pd.to_datetime('2020-03-05'), dynamic=False)
pred_ci = pred.conf_int()
ax = y['2020-01-22':'2020-04-12'].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)
ax.set_xlabel('Date')
ax.set_ylabel('Confirmed Cases')
plt.legend()
plt.show()


# %% calculate MSE on known data

y_forecasted = pred.predicted_mean
y_truth = y['2020-01-22':]
mse = ((y_forecasted - y_truth) ** 2).mean()
print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))

# %% calculate root MSE on known data

print('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse), 2)))


# %% visualize prediction of the model accompanied by the 
# intervals of confidence

import pylab
pred_uc = results.get_forecast(steps=10)
pred_ci = pred_uc.conf_int()
ax = y.plot(label='observed', figsize=(14, 7))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Date')
ax.set_ylabel('Confirmed Cases')

plt.legend()
fig1 = plt.gcf()
fig1.suptitle('Netherlands', fontsize=20)
plt.show()

fig1.savefig('NL.png',transparent=False, bbox_inches='tight', dpi=100)




# %%
