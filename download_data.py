import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def download_market_data():
    """Download VIX and S&P 500 data from Yahoo Finance"""
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Calculate date range (5 years back from today)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)
    
    print(f"Downloading data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # Check if we can connect to Yahoo Finance
    print("Testing yfinance connection...")
    connection_working = False
    try:
        # Try a simpler approach with shorter timeframe
        ticker = yf.Ticker('AAPL')
        hist = ticker.history(period='5d')
        if not hist.empty:
            connection_working = True
            print("Connection successful!")
        else:
            print("Connection failed - no data returned")
    except Exception as e:
        print(f"Connection failed: {e}")
    
    if not connection_working:
        print("Yahoo Finance connection failed. Please check your internet connection.")
        return None, None
    
    # Download VIX data using Ticker method
    print("Downloading VIX data...")
    try:
        vix_ticker = yf.Ticker('^VIX')
        vix = vix_ticker.history(start=start_date, end=end_date)
        
        if vix.empty:
            print("Trying alternative VIX approach...")
            vix = yf.download('^VIX', start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), progress=False, auto_adjust=True, prepost=True, threads=True)
        
        if not vix.empty:
            vix_file = 'data/vix_data.csv'
            vix.to_csv(vix_file)
            print(f"VIX data saved to {vix_file}")
            print(f"VIX data shape: {vix.shape}")
        else:
            print("Could not download VIX data")
            vix = pd.DataFrame()
    except Exception as e:
        print(f"Error downloading VIX: {e}")
        vix = pd.DataFrame()
    
    # Download S&P 500 data using Ticker method
    print("\nDownloading S&P 500 data...")
    try:
        sp500_ticker = yf.Ticker('^GSPC')
        sp500 = sp500_ticker.history(start=start_date, end=end_date)
        
        if sp500.empty:
            print("Trying SPY ETF as alternative...")
            spy_ticker = yf.Ticker('SPY')
            sp500 = spy_ticker.history(start=start_date, end=end_date)
        
        if not sp500.empty:
            sp500_file = 'data/sp500_data.csv'
            sp500.to_csv(sp500_file)
            print(f"S&P 500 data saved to {sp500_file}")
            print(f"S&P 500 data shape: {sp500.shape}")
        else:
            print("Could not download S&P 500 data")
            sp500 = pd.DataFrame()
    except Exception as e:
        print(f"Error downloading S&P 500: {e}")
        sp500 = pd.DataFrame()
    
    # Display sample data
    print("\n" + "="*50)
    print("VIX Data Sample (last 5 rows):")
    print(vix.tail())
    
    print("\n" + "="*50)
    print("S&P 500 Data Sample (last 5 rows):")
    print(sp500.tail())
    
    return vix, sp500

if __name__ == "__main__":
    try:
        vix_data, sp500_data = download_market_data()
        print("\n✅ Data download completed successfully!")
    except Exception as e:
        print(f"❌ Error downloading data: {str(e)}")