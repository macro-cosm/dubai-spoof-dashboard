import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def curate_sp500_data():
    """
    Remove the last 3 days from S&P 500 data to avoid artificially low volume issues.
    Updates both the original sp500_data.csv and regenerates spoof_sp500_data.csv
    """
    print("🔧 Starting S&P 500 data curation...")
    
    # Load original S&P 500 data
    print("Loading original S&P 500 data...")
    sp500_data = pd.read_csv("data/sp500_data.csv")
    sp500_data["Date"] = pd.to_datetime(sp500_data["Date"], utc=True)
    sp500_data = sp500_data.set_index("Date")
    sp500_data = sp500_data.sort_index()
    
    print(f"Original data shape: {sp500_data.shape}")
    print(f"Original date range: {sp500_data.index[0]} to {sp500_data.index[-1]}")
    
    # Remove last 3 days
    curated_data = sp500_data.iloc[:-3]
    
    print(f"Curated data shape: {curated_data.shape}")
    print(f"Curated date range: {curated_data.index[0]} to {curated_data.index[-1]}")
    
    # Save curated data back to original file
    curated_data.to_csv("data/sp500_data.csv")
    print("✅ Updated data/sp500_data.csv with curated data (last 3 days removed)")
    
    # Show what was removed
    removed_data = sp500_data.iloc[-3:]
    print(f"\n📊 Removed data (last 3 days):")
    print(removed_data[['Open', 'High', 'Low', 'Close', 'Volume']])
    
    return curated_data

def regenerate_spoof_data():
    """Regenerate spoof S&P 500 data using the curated dataset"""
    print("\n🔄 Regenerating spoof S&P 500 data with curated dataset...")
    
    # Import the generation function
    from gen_sp500_data import generate_sp500_simulation_data
    
    # Regenerate with same seed for consistency
    final_df, cutoff_date = generate_sp500_simulation_data(n_samples=100, seed=42)
    
    print("✅ Spoof data regenerated with curated S&P 500 data")
    return final_df, cutoff_date

def main():
    """Main curation function"""
    try:
        # Step 1: Curate the original data
        curated_data = curate_sp500_data()
        
        # Step 2: Regenerate spoof data
        spoof_data, cutoff_date = regenerate_spoof_data()
        
        print(f"\n🎯 Summary:")
        print(f"- Original S&P 500 data curated (last 3 days removed)")
        print(f"- New date range: {curated_data.index[0].strftime('%Y-%m-%d')} to {curated_data.index[-1].strftime('%Y-%m-%d')}")
        print(f"- Spoof data regenerated with {spoof_data.shape[0]} days")
        print(f"- Cutoff date: {cutoff_date.strftime('%Y-%m-%d')}")
        print("✅ S&P 500 data curation completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during S&P 500 data curation: {str(e)}")

if __name__ == "__main__":
    main()