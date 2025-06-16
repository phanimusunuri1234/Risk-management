#!/usr/bin/env python
# coding: utf-8

# # Value at Risk (VaR) 📉
# 
# Value at Risk (VaR) is a statistical measure used in risk management to quantify the potential loss in value of a portfolio or asset over a defined period, given a specified confidence level. It provides a probabilistic estimate of the worst-case loss that could occur within the specified timeframe under normal market conditions.
# 
# Definition VaR is defined as the maximum loss that is not expected to be exceeded with a given confidence level over a specified period. For instance, a 1-day VaR at a 95% confidence level means that there is a 5% chance that the loss will exceed the VaR estimate over one day.
# 
# Formula The formula for VaR depends on the method used to estimate it. Common methods include:
# 
# Historical Simulation Method : For a given confidence level  (α)  (e.g., 95% or 99%) and a historical dataset of returns, VaR can be estimated as:
# VaR α = Quantile α ( Return Distribution ) where ( Quantile α ) is the value below which ( α % ) of the observations fall.
# 
# Variance-Covariance Method : In this method, assuming returns are normally distributed, VaR can be calculated using:
# [ VaR α = μ − Z α ⋅ σ ]
# 
# where:
# 
# ( μ ) = Mean of the portfolio returns ( σ ) = Standard deviation of the portfolio returns ( Z α ) = Z-score corresponding to the confidence level ( α ) (e.g., 1.645 for 95% confidence)
# 
# Monte Carlo Simulation : VaR can also be estimated using Monte Carlo simulation by simulating a large number of portfolio returns and calculating the percentile value at the desired confidence level.
# Example Suppose you have a portfolio with the following characteristics:
# 
# Mean return ( ( μ )) = 0.1% Standard deviation ( ( σ )) = 1.5% Confidence level = 95%
# 
# To calculate the 1-day VaR using the Variance-Covariance method:
# 
# Find the Z-score for a 95% confidence level, which is approximately 1.645.
# 
# Apply the formula: [ VaR 95 % = μ − Z 95 % ⋅ σ ]
# 
# [ VaR 95 % = 0.001 − 1.645 ⋅ 0.015 ]
# 
# [ VaR 95 % = 0.001 − 0.024675 ]
# 
# [ VaR 95 % = − 0.023675 ]
# 
# The 1-day VaR at the 95% confidence level is approximately -2.37%. This means there is a 5% chance that the portfolio will lose more than 2.37% of its value in one day.
# 
# Conclusion : Value at Risk (VaR) is a crucial tool in risk management, helping investors and firms to understand potential losses and make informed decisions. By using different methods to estimate VaR, one can better assess and manage financial risks under various market conditions.

# In[ ]:


pip install yfinance
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt


# In[3]:


# Download 1 year of daily adjusted closing prices
stocks = ['INFY.NS', 'RELIANCE.NS', 'HDFCBANK.NS']
data = yf.download(stocks, start='2024-06-01', end='2025-06-01') [('Close')]


# In[4]:


data.head()


# In[6]:


data.tail()


# In[7]:


# Calculate daily returns (% change)
returns = data.pct_change().dropna()

# Preview returns
print(returns.tail())


# In[8]:


# Summary statistics
print(returns.describe())
# Correlation matrix
print(returns.corr())


# In[9]:


# Plot daily returns for each stock
data['INFY.NS'].plot(figsize=(12,6), title="Infosys Daily Returns")
plt.grid(True)
plt.show()


# In[10]:


# Plot daily returns for each stock
data['HDFCBANK.NS'].plot(figsize=(12,6), title="HDFC Daily Returns")
plt.grid(True)
plt.show()


# In[11]:


# Plot daily returns for each stock
data['RELIANCE.NS'].plot(figsize=(12,6), title="RELIANCE Daily Returns")
plt.grid(True)
plt.show()


# In[12]:


# Define Portfolio Weights & Returns

# Define portfolio weights
weights = np.array([0.4, 0.35, 0.25])  # INFY, RELIANCE, HDFCBANK

# Portfolio daily returns (dot product of weights and daily returns)
#Portfolio Return=0.4⋅R INFY+0.35⋅R RELIANCE+0.25⋅R HDFCBANK

portfolio_returns = returns.dot(weights)

# Plot portfolio returns
portfolio_returns.plot(figsize=(10,5), title="Portfolio Daily Returns")
plt.grid(True)
plt.show()


# In[13]:


portfolio_returns


# # Historical VaR

# In[14]:


# 95% confidence level → 5% left tail
confidence_level = 0.95
historical_var_95 = np.percentile(portfolio_returns, (1 - confidence_level) * 100)

print(f"Historical VaR at 95%: {historical_var_95:.4f}")


# # Parametric VaR

# In[15]:


# Assuming normal distribution
mean = portfolio_returns.mean()
std_dev = portfolio_returns.std()

z_score_95 = 1.65  # for 95%
parametric_var_95 = mean - z_score_95 * std_dev

print(f"Parametric VaR at 95%: {parametric_var_95:.4f}")


# # Monte Carlo Simulation 

# In[16]:


# Simulate 10,000 returns using normal distribution
simulated_returns = np.random.normal(mean, std_dev, 10000)

# Calculate portfolio loss
initial_investment = 100000  # ₹1 lakh portfolio

simulated_end_values = initial_investment * (1 + simulated_returns)
simulated_losses = initial_investment - simulated_end_values

# Monte Carlo VaR
monte_carlo_var_95 = np.percentile(simulated_losses, 5)
print(f"Monte Carlo VaR at 95%: ₹{monte_carlo_var_95:.2f}")


# In[17]:


# simulated_end_values = initial_investment * (1 + simulated_returns)
# Explanation: This line calculates the final value of the portfolio for each simulated return.
# We use `1 + simulated_returns` because a return is a percentage change.
# If `returns` is 5% (0.05), `1 + 0.05 = 1.05`. Multiplying `initial_investment` by `1.05`
# gives you the original investment plus the 5% gain.
# If `returns` is -2% (-0.02), `1 + (-0.02) = 0.98`. Multiplying by `0.98` gives you
# the original investment minus the 2% loss.
# This formula efficiently calculates the new value after a percentage increase or decrease.


# In[18]:


print("Value at Risk (VaR) – ₹1,00,000 Portfolio at 95% Confidence")
print("----------------------------------------------------------")
print(f"Historical VaR     : ₹{abs(historical_var_95)*100000:.2f}")
print(f"Parametric VaR     : ₹{abs(parametric_var_95)*100000:.2f}")
print(f"Monte Carlo VaR    : ₹{monte_carlo_var_95:.2f}")


# In[21]:


print(f"Based on {len(simulated_losses)} Monte Carlo simulations:")
print(f"With {int(confidence_level*100)}% confidence, the portfolio is not expected to lose more than ₹{monte_carlo_var_95:,.2f} in one day.")
print(f"There is a {int((1 - confidence_level)*100)}% chance that losses could exceed this amount under normal market conditions.")


# In[ ]:




