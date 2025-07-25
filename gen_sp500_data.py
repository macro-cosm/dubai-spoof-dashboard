import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def generate_sp500_simulation_data(n_samples=100, seed=42):
    """
    Generate S&P 500 volume simulation data and save to CSV:
    - Calibration period starts March 14, 2024
    - Average of samples = S&P 500 Volume + 3% Gaussian noise (becomes the mean)
    - Each sample has 10% noise around that mean during calibration
    - Forecasting period: noise increases sqrt(t) from 10% to 30%
    """
    print("Loading S&P 500 data...")

    # Load S&P 500 data
    sp500_data = pd.read_csv("data/sp500_data.csv")
    sp500_data["Date"] = pd.to_datetime(sp500_data["Date"], utc=True)
    sp500_data = sp500_data.set_index("Date")
    sp500_data = sp500_data.sort_index()

    # Set calibration start date to March 14, 2024
    calibration_start = pd.Timestamp("2024-03-14").tz_localize("UTC")

    # Calculate 3 months ago from today as cutoff between calibration and forecasting
    today = datetime.now()
    cutoff_date = pd.Timestamp(today - timedelta(days=90)).tz_localize("UTC")

    # Filter data from calibration start date onwards
    sp500_data = sp500_data[sp500_data.index >= calibration_start]

    # Split data into calibration and forecasting periods
    calibration_data = sp500_data[sp500_data.index <= cutoff_date]["Volume"]
    forecasting_data = sp500_data[sp500_data.index > cutoff_date]["Volume"]

    print(f"Data from calibration start: {len(sp500_data)} days (from {calibration_start.strftime('%Y-%m-%d')})")
    print(
        f"Calibration period: {len(calibration_data)} days (until {cutoff_date.strftime('%Y-%m-%d')})"
    )
    print(f"Forecasting period: {len(forecasting_data)} days")
    print(f"Generating {n_samples} simulation samples with new noise model...")

    # Set random seed for reproducibility
    np.random.seed(seed)

    # Initialize simulation results
    simulation_results = []

    # First, generate the mean trajectory for each day
    # Mean = S&P 500 Volume + 3% Gaussian noise (this becomes the average for all samples)
    all_dates = list(calibration_data.index) + list(forecasting_data.index)
    all_actual_values = list(calibration_data.values) + list(forecasting_data.values)

    # Sort by date
    combined = list(zip(all_dates, all_actual_values))
    combined.sort(key=lambda x: x[0])
    all_dates = [x[0] for x in combined]
    all_actual_values = [x[1] for x in combined]

    # Generate daily means: actual S&P 500 Volume + 3% noise
    daily_means = []
    for actual_value in all_actual_values:
        mean_noise = np.random.normal(0, 0.03)  # 3% noise for the mean
        daily_mean = actual_value * (1 + mean_noise)
        daily_means.append(daily_mean)

    # Generate samples around these daily means
    for sample_idx in range(n_samples):
        sample_values = []

        # Calibration period: 10% noise around the daily mean
        for i, daily_mean in enumerate(daily_means[: len(calibration_data)]):
            sample_noise = np.random.normal(0, 0.10)  # 10% noise around mean
            simulated_value = daily_mean * (1 + sample_noise)
            sample_values.append(simulated_value)

        # Forecasting period: Error increases from 10% to 30% as sqrt(t)
        n_forecast = len(forecasting_data)
        forecast_means = daily_means[len(calibration_data) :]

        for i, daily_mean in enumerate(forecast_means):
            # t goes from 0 to 1 over the forecasting period
            t = i / max(1, n_forecast - 1) if n_forecast > 1 else 0
            # Error increases as sqrt(t) from 10% to 30%
            error_pct = 0.10 + (0.30 - 0.10) * np.sqrt(t)
            sample_noise = np.random.normal(0, error_pct)
            simulated_value = daily_mean * (1 + sample_noise)
            sample_values.append(simulated_value)

        simulation_results.append(sample_values)

    # Create DataFrame with dates as index and samples as columns
    sim_df = pd.DataFrame(
        np.array(simulation_results).T,  # Transpose to have dates as rows
        index=all_dates,
        columns=[f"Sample_{i+1}" for i in range(n_samples)],
    )

    # Add metadata columns
    metadata_df = pd.DataFrame(
        {
            "Actual_Volume": all_actual_values,
            "Period": [
                "Calibration" if date <= cutoff_date else "Forecasting"
                for date in all_dates
            ],
            "Days_Since_Cutoff": [
                (date - cutoff_date).days if date > cutoff_date else 0
                for date in all_dates
            ],
        },
        index=all_dates,
    )

    # Combine simulation data with metadata
    final_df = pd.concat([metadata_df, sim_df], axis=1)

    # Save to CSV
    output_file = "data/spoof_sp500_data.csv"
    final_df.to_csv(output_file)

    print(f"\n✅ S&P 500 simulation data saved to {output_file}")
    print(f"Data shape: {final_df.shape}")
    print(
        f"Date range: {final_df.index[0].strftime('%Y-%m-%d')} to {final_df.index[-1].strftime('%Y-%m-%d')}"
    )

    # Display sample statistics
    print("\n📊 Sample Statistics:")
    print("Last 5 rows of data:")
    print(final_df.tail())

    # Calculate and display error statistics
    calib_mask = final_df["Period"] == "Calibration"
    forecast_mask = final_df["Period"] == "Forecasting"

    if calib_mask.any():
        calib_actual = final_df.loc[calib_mask, "Actual_Volume"]
        calib_median = final_df.loc[
            calib_mask, [col for col in final_df.columns if col.startswith("Sample_")]
        ].median(axis=1)
        calib_error = np.abs((calib_median - calib_actual) / calib_actual).mean()
        print(
            f"\nCalibration period average relative error: {calib_error*100:.2f}% (10% noise around mean)"
        )

    if forecast_mask.any():
        forecast_actual = final_df.loc[forecast_mask, "Actual_Volume"]
        forecast_median = final_df.loc[
            forecast_mask,
            [col for col in final_df.columns if col.startswith("Sample_")],
        ].median(axis=1)
        forecast_error = np.abs(
            (forecast_median - forecast_actual) / forecast_actual
        ).mean()
        print(
            f"Forecasting period average relative error: {forecast_error*100:.2f}% (10%-30% noise)"
        )

    return final_df, cutoff_date


if __name__ == "__main__":
    try:
        data, cutoff_date = generate_sp500_simulation_data()
        print(f"\n🎯 Cutoff date used: {cutoff_date.strftime('%Y-%m-%d')}")
        print("✅ S&P 500 simulation generation completed successfully!")
    except Exception as e:
        print(f"❌ Error generating S&P 500 simulation data: {str(e)}")