"""
Generate synthetic GDP simulation data for dashboard illustration
"""

import pandas as pd
import numpy as np
from datetime import datetime


def parse_quarter_date(date_str):
    """Convert 'YYYY QN' format to datetime"""
    year, quarter = date_str.split(" Q")
    year = int(year)
    quarter = int(quarter)
    month = (quarter - 1) * 3 + 1  # Q1=Jan, Q2=Apr, Q3=Jul, Q4=Oct
    return pd.Timestamp(year, month, 1)


def generate_synthetic_gdp_data(sigma=0.025, num_samples=100, seed=42):
    """
    Generate synthetic GDP simulation data based on actual GDP

    Parameters:
    - sigma: volatility parameter (default 0.075)
    - num_samples: number of simulation samples to generate
    - seed: random seed for reproducibility
    """
    np.random.seed(seed)

    # Load actual GDP data
    print("Loading actual GDP data...")
    actual_gdp = pd.read_csv("data/actualGDP.csv", index_col=0)
    actual_gdp = actual_gdp.T  # Transpose to have dates as rows
    actual_gdp.index = [parse_quarter_date(date) for date in actual_gdp.index]
    actual_gdp.columns = ["GDP"]

    # Filter to start from 2014 Q1 and end at 2019 Q1 (to match original simulation data)
    start_date = pd.Timestamp(2014, 1, 1)  # 2014 Q1
    cutoff_date = pd.Timestamp(2019, 1, 1)  # 2019 Q1

    actual_gdp_filtered = actual_gdp[
        (actual_gdp.index >= start_date) & (actual_gdp.index <= cutoff_date)
    ]

    print(
        f"Filtered GDP data from {actual_gdp_filtered.index[0].strftime('%Y Q%q')} to {actual_gdp_filtered.index[-1].strftime('%Y Q%q')}"
    )
    print(f"Number of time periods: {len(actual_gdp_filtered)}")

    # Get the date index for simulations (same as filtered actual data)
    date_columns = []
    for date in actual_gdp_filtered.index:
        quarter = (date.month - 1) // 3 + 1
        date_columns.append(f"{date.year} Q{quarter}")

    # Generate average values: real_gdp * (1 + 0.02 * gaussian_noise)
    print("Generating average trajectory...")
    T = len(actual_gdp_filtered)
    noise_avg = np.random.normal(0, 1, T)  # Gaussian white noise with variance 1
    print(f"Average noise first 3 values: {noise_avg[:3]}")

    # Average trajectory
    avg_multiplier = 1 + 0.02 * noise_avg
    avg_gdp = actual_gdp_filtered["GDP"].values * avg_multiplier
    print(f"Actual GDP first 3 values: {actual_gdp_filtered['GDP'].values[:3]}")
    print(f"Average multiplier first 3 values: {avg_multiplier[:3]}")
    print(f"Average GDP first 3 values: {avg_gdp[:3]}")
    print(f"Average GDP range: {avg_gdp.min():.2f} to {avg_gdp.max():.2f}")

    # Generate simulation samples
    print(f"Generating {num_samples} simulation samples with sigma={sigma}...")
    simulation_data = {}

    for i in range(num_samples):
        sample_name = f"Sample{i+1}"

        # Generate noise for this sample
        noise_i = np.random.normal(0, 1, T)
        print(f"Sample {i+1}: noise_i first 3 values: {noise_i[:3]}")

        # Time-varying volatility factor starting from 0 at t=0
        sample_values = []
        for t in range(T):
            # Volatility increases with time: sqrt(t * sigma^2), starting from 0
            vol_factor = np.sqrt(t * sigma**2)  # t=0 gives vol_factor=0
            sample_multiplier = 1 + vol_factor * noise_i[t]
            sample_value = avg_gdp[t] * sample_multiplier

            if i == 0:  # Log details for first sample
                print(
                    f"  t={t}: vol_factor={vol_factor:.6f}, noise={noise_i[t]:.6f}, multiplier={sample_multiplier:.6f}"
                )
                print(
                    f"  t={t}: avg_gdp={avg_gdp[t]:.2f}, sample_value={sample_value:.2f}"
                )

            sample_values.append(sample_value)

        simulation_data[sample_name] = sample_values

        if i == 0:  # Log first few values of first sample
            print(f"Sample 1 first 3 values: {sample_values[:3]}")
        if i < 3:  # Check if samples are identical for first few
            print(f"Sample {i+1} final value: {sample_values[-1]:.2f}")

    # Create DataFrame with samples as rows and dates as columns (matching original format)
    sim_df = pd.DataFrame(simulation_data).T  # Transpose so samples are rows
    sim_df.columns = date_columns  # Set date columns

    # Save to CSV
    output_file = "data/test_gdp.csv"
    print(f"Saving synthetic data to {output_file}...")
    sim_df.to_csv(output_file)

    # Print summary statistics
    print("\n=== Summary Statistics ===")
    print(
        f"Original GDP range: ${actual_gdp_filtered['GDP'].min()/1e12:.2f}T - ${actual_gdp_filtered['GDP'].max()/1e12:.2f}T"
    )
    print(f"Average GDP range: ${avg_gdp.min()/1e12:.2f}T - ${avg_gdp.max()/1e12:.2f}T")
    print(
        f"Simulation median range: ${sim_df.median(axis=0).min()/1e12:.2f}T - ${sim_df.median(axis=0).max()/1e12:.2f}T"
    )
    print(f"Simulation std at final period: ${sim_df.iloc[:, -1].std()/1e12:.2f}T")

    # Show first few rows
    print(f"\nFirst 5 rows of synthetic data:")
    print(sim_df.head())

    print(f"\nSynthetic GDP data saved successfully to {output_file}")
    return sim_df


if __name__ == "__main__":
    # Generate synthetic data
    synthetic_data = generate_synthetic_gdp_data(sigma=0.05, num_samples=100, seed=42)

    print("\nSynthetic GDP simulation data generation complete!")
