import yfinance as yf
import os

# Create data folder if it doesn't exist
os.makedirs('data', exist_ok=True)

# Define cryptocurrencies and their Yahoo Finance tickers
cryptos = {
    "Polkadot": "DOT-USD",
    "Cardano": "ADA-USD",
    "Cosmos": "ATOM-USD",
    "Dogecoin": "DOGE-USD",
    "Bitcoin": "BTC-USD"
}

# Define date range
start_date = "2023-01-01"
end_date = "2025-11-01"

# Download and save data for each crypto
for name, ticker in cryptos.items():
    print(f"Downloading data for {name} ({ticker})...")
    data = yf.download(ticker, start=start_date, end=end_date)
    file_path = f"data/{name.lower()}_data.csv"
    data.to_csv(file_path)
    print(f"Saved {name} data to {file_path}")

print("All cryptocurrency data downloaded and saved in 'data/' folder.")
