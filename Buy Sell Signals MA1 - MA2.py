#!/usr/bin/env python
# coding: utf-8

# In[25]:


#Libraries
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as web
import numpy as np
import plotly.graph_objects as go
plt.style.use('seaborn-pastel')


# In[26]:


stock = input("Enter Ticker: ")


# In[27]:


MovAv1 = int(input("MA: "))


# In[28]:


MovAv2 = int(input("MA: "))


# In[29]:


Start = input(" Year-Month-Day: ")


# In[30]:


#DIA OHLC Moving Average Data
stock = web.DataReader(stock, data_source='yahoo', start=Start)

stock['MA'] = stock['Close'].rolling(MovAv1).mean()
stock['MA1'] = stock['Close'].rolling(MovAv2).mean()

MA = stock['MA']
MA1 = stock['MA1']

stock = stock.dropna()
stock

#Close - MA - MA1 Graph
plt.figure(figsize=(16,8))
plt.title('Close - MA(1) - MA(2) Price History')
plt.plot(stock['Close'])
plt.plot(MA)
plt.plot(MA1)
plt.xlabel('Year',fontsize=18)
plt.ylabel('Price USD ($)',fontsize=18)
plt.legend(['Close', 'MA(1)', 'MA(2)'], loc='upper left')
plt.show()

#Create a function to signal when to buy and sell an asset
def buy_sell(signal):
  sigPriceBuy = []
  sigPriceSell = []
  flag = -1
  for i in range(0,len(signal)):
    #if MA > MA1  then buy else sell
      if signal['MA'][i] > signal['MA1'][i]:
        if flag != 1:
          sigPriceBuy.append(signal['Close'][i])
          sigPriceSell.append(np.nan)
          flag = 1
        else:
          sigPriceBuy.append(np.nan)
          sigPriceSell.append(np.nan)
        #print('Buy')
      elif signal['MA'][i] < signal['MA1'][i]:
        if flag != 0:
          sigPriceSell.append(signal['Close'][i])
          sigPriceBuy.append(np.nan)
          flag = 0
        else:
          sigPriceBuy.append(np.nan)
          sigPriceSell.append(np.nan)
        #print('sell')
      else: #Handling nan values
        sigPriceBuy.append(np.nan)
        sigPriceSell.append(np.nan)
  
  return (sigPriceBuy, sigPriceSell)


# In[31]:


#Create a new dataframe
signal = pd.DataFrame(index=stock['Close'].index)
signal['Close'] = stock['Close']
signal['MA'] = MA
signal['MA1'] = MA1


# In[32]:


signal


# In[33]:


x = buy_sell(signal)
signal['Buy_Signal_Price'] = x[0]
signal['Sell_Signal_Price'] = x[1]


# In[34]:


signal


# In[35]:


#Daily returns Data
stock_daily_returns = stock['Adj Close'].diff()

fig = plt.figure(figsize=(18,10))
ax1 = fig.add_axes([0.1,0.1,0.8,0.8])
ax1.plot(stock_daily_returns)
ax1.set_xlabel("Date")
ax1.set_ylabel("Returns")
ax1.set_title("Daily returns data")
plt.show()


# In[36]:


stock['cum'] = stock_daily_returns.cumsum()
stock.tail()


# In[37]:


fig = plt.figure(figsize=(18,10))
ax1 = fig.add_axes([0.1,0.1,0.8,0.8])
stock['cum'].plot()
ax1.set_xlabel("Date")
ax1.set_ylabel("Growth of $1 investment")
ax1.set_title("Stock Daily Cumulative Returns Data")
plt.show()


# In[38]:


# MA > MA1 Calculation
stock['Shares'] = [1 if stock.loc[ei, 'MA']>stock.loc[ei, 'MA1'] else 0 for ei in stock.index]


# In[39]:


#Strategy Profit Plot
stock['Close1'] = stock['Close'].shift(-1)
stock['Profit'] = [stock.loc[ei, 'Close1'] - stock.loc[ei, 'Close'] if stock.loc[ei, 'Shares']==1 else 0 for ei in stock.index]
stock['Profit'].plot(figsize=(18, 10))
plt.axhline(y=0, color='red')
plt.title('Profit')


# In[40]:


#Profit per Day, and Accumulative Wealth
stock['Wealth'] = stock['Profit'].cumsum()
stock.tail()


# In[41]:


stock['diff'] = stock['Wealth'] - stock['cum']
stock.tail()


# In[42]:


stock['pctdiff'] = (stock['diff'] / stock['cum'])*100
stock.tail()


# In[43]:


start = stock.iloc[0]
stock['start'] = start
start['Close']


# In[44]:


start1 = start['Close']
stock['start1'] = start1
stock


# In[56]:


#Plot the Data with Buy and Sell Signals
my_stocks = signal
ticker = 'Close'

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters ()

fig, ax1 = plt.subplots(figsize=(18, 10))

plt.rc('axes',edgecolor='navy')

plt.annotate('$\it{Data Source: Yahoo Finance}$', xy=(0.02, 0.05), xycoords='axes fraction', color='navy')
plt.annotate('Program by: Taylor Bommarito', xy=(0.02, 0.08), xycoords='axes fraction', color='navy')
plt.annotate('1 Share of Stock from Buy and Hold Strategy is : ${:.2f}'.format(stock.loc[stock.index[-2],'cum']), xy=(0.215, 1.13), xycoords='axes fraction', fontsize=18, weight='bold', color='navy')
plt.annotate('Strategy difference is : ${:.2f}'.format(stock.loc[stock.index[-2],'diff']), xy=(0.34, 1.075), xycoords='axes fraction', fontsize=18, weight='bold', color='navy')
plt.annotate('Strategy Percent Improvement : {:.3f}%'.format(stock.loc[stock.index[-2],'pctdiff']), xy=(0.28, 1.02), xycoords='axes fraction', fontsize=18, weight='bold', color='navy')


ax1.set_facecolor('white')

ax1.grid(False)

ax1.tick_params(axis='y', colors='red')


curve1 = ax1.plot(my_stocks[ticker],  label='Stock Close Price Starts at : ${:.2f}'.format(stock.loc[stock.index[-2],'start1']), color='olive', linewidth= 1.5, alpha = 0.50)
plt.legend(title='Left Y-Axis Legend (Red Side)', loc='upper left')
plt.xlabel('Date', color = 'navy', fontsize=18, weight='bold')
plt.ylabel('Close Price USD ($)', color = 'red',fontsize=18, weight='bold')


curve1 = ax1.plot(my_stocks['MA'], label='MA (1)', color='orange', linewidth= 1.5)
curve1 = ax1.plot(my_stocks['MA1'], label='MA (2)',color='blue', linewidth= 1.5, alpha = 0.35)
curve1 = ax1.plot(my_stocks.index, my_stocks['Buy_Signal_Price'], color = 'green', label='Buy Signal', marker = '^', alpha = 1)
curve1 = ax1.plot(my_stocks.index, my_stocks['Sell_Signal_Price'], color = 'red', label='Sell Signal', marker = 'v', alpha = 1)
plt.title('1 Share of Stock from MA(1) > MA (2) Strategy is : ${:.2f}'.format(stock.loc[stock.index[-2],'Wealth']), color = 'navy', fontsize=18, weight='bold', pad=100)
plt.legend(title='Right Y-Axis Legend (Blue Side)', loc='lower right')


ax1.tick_params(axis='x', colors='navy')
ax1.tick_params(axis='y', colors='red')


plt.plot()
plt.show()


# In[57]:


#Accumulative Wealth from Strategy Chart
fig, ax1 = plt.subplots(figsize=(18, 10))

plt.rc('axes',edgecolor='navy')

plt.annotate('$\it{Data Source: Yahoo Finance}$', xy=(0.02, 0.05), xycoords='axes fraction', color='navy')
plt.annotate('Program by: Taylor Bommarito', xy=(0.02, 0.08), xycoords='axes fraction', color='navy')
plt.annotate('1 Share of Stock from Buy and Hold Strategy is : ${:.2f}'.format(stock.loc[stock.index[-2],'cum']), xy=(0.215, 1.13), xycoords='axes fraction', fontsize=18, weight='bold', color='navy')
plt.annotate('Strategy difference is : ${:.2f}'.format(stock.loc[stock.index[-2],'diff']), xy=(0.34, 1.075), xycoords='axes fraction', fontsize=18, weight='bold', color='navy')
plt.annotate('Strategy Percent Improvement : {:.3f}%'.format(stock.loc[stock.index[-2],'pctdiff']), xy=(0.28, 1.02), xycoords='axes fraction', fontsize=18, weight='bold', color='navy')
ax1.set_facecolor('white')

ax1.grid(False)

ax1.tick_params(axis='x', colors='navy')
ax1.tick_params(axis='y', colors='red')

plt.plot(stock['Wealth'], label='MA Wealth Accumulation $$$$ (Starts at $0)', color='green')
plt.plot(stock['cum'], label='Buy Hold Wealth Accumulation $$$$ (Starts at $0)', color='blue', alpha=0.35)
plt.legend( loc='upper left')
plt.xlabel('Date', color = 'navy', fontsize=18, weight='bold')
plt.ylabel('Amount Earned USD ($)', color = 'red', fontsize=18, weight='bold')
plt.title('1 Share of Stock from MA(1) > MA (2) Strategy is : ${:.2f}'.format(stock.loc[stock.index[-2],'Wealth']), color = 'navy', fontsize=18, weight='bold', pad=100)


# In[ ]:





# In[ ]:




