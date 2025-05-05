import yfinance as yf
import pandas as pd

# Download historical data for Apple
symbol = 'AAPL'
df = yf.download(symbol, start='2015-01-01', end='2024-12-31')

# Save or inspect the data
print(df.head())
df.to_csv(f'{symbol}_historical.csv')
