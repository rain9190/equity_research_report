import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import LinearRegression

start_date = '2020-08-31'
end_date = '2025-08-31'

# change the ticker here for whatever stock you want
stock = yf.download('LEMONTREE.NS', start=start_date, end=end_date, interval='1mo', auto_adjust=True)['Close']
nifty = yf.download('^NSEI', start=start_date, end=end_date, interval='1mo', auto_adjust=True)['Close']

stock_returns = stock.pct_change()
nifty_returns = nifty.pct_change()

table = pd.concat([stock, stock_returns, nifty, nifty_returns], axis=1)
table.columns = ['Stock_Price', 'Stock_Return', 'Nifty_Price', 'Nifty_Return']
table = table.dropna()
print("Monthly Prices and Returns Table:\n")
print(table)

# the regression code
X = table['Nifty_Return'].values.reshape(-1,1)  
y = table['Stock_Return'].values             
reg = LinearRegression().fit(X, y)
beta = reg.coef_[0]
alpha = reg.intercept_

print("\nRegression Equation:")
print(f"Stock_Return = {alpha:.6f} + {beta:.6f} * Nifty_Return")
print(f"\nBeta of the Stock: {beta:.4f}")