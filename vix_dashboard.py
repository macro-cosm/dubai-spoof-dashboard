import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from macrocosm_visual.viz_setup import setup_plotly, get_macrocosm_colors

# Setup Macrocosm visual styling
setup_plotly()
colors = get_macrocosm_colors()

# Page configuration
st.set_page_config(
    page_title="VIX Model Dashboard", page_icon="📊", layout="wide"
)

# Custom CSS for dark theme
st.markdown(
    """
<style>
    .stApp {
        background-color: #1e1e1e;
        color: #ffffff;
    }
    .main .block-container {
        background-color: #1e1e1e;
        color: #ffffff;
    }
    .stMarkdown {
        color: #ffffff;
    }
    .stInfo {
        background-color: #2d2d2d;
        color: #ffffff;
    }
    .stMetric {
        background-color: #2d2d2d;
        color: #ffffff;
    }
    .stSidebar {
        background-color: #2d2d2d;
    }
    .stSidebar .stMarkdown {
        color: #ffffff;
    }
    div[data-testid="metric-container"] {
        background-color: #2d2d2d;
        border: 1px solid #404040;
        padding: 1rem;
        border-radius: 0.5rem;
        color: #ffffff;
    }
    div[data-testid="metric-container"] > label {
        color: #ffffff !important;
    }
    div[data-testid="metric-container"] > div {
        color: #ffffff !important;
    }
</style>
""",
    unsafe_allow_html=True,
)

def load_vix_simulation_data():
    """Load pre-generated VIX simulation data and filter to last 18 months"""
    # Load simulation data
    sim_data = pd.read_csv("data/spoof_vix_data.csv", index_col=0)
    sim_data.index = pd.to_datetime(sim_data.index, utc=True)
    
    # Extract metadata first
    actual_data_full = sim_data['Actual_VIX']
    period_data_full = sim_data['Period']
    
    # Filter to show data starting from October 2024
    october_2024 = pd.Timestamp("2024-10-01").tz_localize('UTC')
    sim_data = sim_data[sim_data.index >= october_2024]
    
    # Extract metadata from filtered data
    actual_data = sim_data['Actual_VIX']
    period_data = sim_data['Period']
    
    # Determine cutoff date (last calibration date in filtered data)
    calibration_mask = period_data == 'Calibration'
    if calibration_mask.any():
        cutoff_date = sim_data[calibration_mask].index[-1]
    else:
        # If no calibration data in the 18-month window, use 3 months ago
        cutoff_date = pd.Timestamp(today - timedelta(days=90)).tz_localize('UTC')
    
    # Extract simulation columns from filtered data
    sample_cols = [col for col in sim_data.columns if col.startswith('Sample_')]
    simulation_df = sim_data[sample_cols]
    
    # Split actual data by period (from filtered data)
    calibration_actual = actual_data[calibration_mask]
    forecasting_actual = actual_data[period_data == 'Forecasting']
    
    return actual_data, simulation_df, cutoff_date, calibration_actual, forecasting_actual

def create_vix_plot(actual_data, sim_data, cutoff_date, confidence_level=95):
    """Create VIX plot with calibration/forecasting periods"""
    
    fig = go.Figure()
    
    # Calculate statistics from simulations
    median_sim = sim_data.median(axis=1)
    
    # Calculate confidence intervals
    lower_percentile = (100 - confidence_level) / 2
    upper_percentile = 100 - lower_percentile
    
    lower_ci = sim_data.quantile(lower_percentile / 100, axis=1)
    upper_ci = sim_data.quantile(upper_percentile / 100, axis=1)
    
    # Split actual data into calibration and forecasting periods
    calibration_actual = actual_data[actual_data.index <= cutoff_date]
    
    # Add calibration period actual VIX FIRST (so it's behind) with thicker line
    fig.add_trace(
        go.Scatter(
            x=calibration_actual.index,
            y=calibration_actual.values,
            mode="lines",
            name="Actual VIX (Calibration Period)",
            line=dict(color=colors[1], width=6),  # Coral red, 1.5x thicker (4 * 1.5 = 6)
            hovertemplate="<b>Actual VIX (Calibration)</b><br>"
            + "Date: %{x}<br>"
            + "VIX: %{y:.2f}<br>"
            + "<extra></extra>",
        )
    )
    
    # Add confidence interval as filled area
    fig.add_trace(
        go.Scatter(
            x=list(median_sim.index) + list(median_sim.index[::-1]),
            y=list(upper_ci.values) + list(lower_ci.values[::-1]),
            fill="toself",
            fillcolor="rgba(35, 105, 189, 0.2)",  # Semi-transparent blue
            line=dict(color="rgba(255,255,255,0)"),
            name=f"{confidence_level}% Confidence Interval",
            hoverinfo="skip",
            showlegend=True
        )
    )
    
    # Add median simulation line (dashed) - this will be on top
    fig.add_trace(
        go.Scatter(
            x=median_sim.index,
            y=median_sim.values,
            mode="lines",
            name="Median Simulation",
            line=dict(color=colors[0], width=3, dash="dash"),  # Primary blue
            hovertemplate="<b>Median Simulation</b><br>"
            + "Date: %{x}<br>"
            + "VIX: %{y:.2f}<br>"
            + "<extra></extra>",
        )
    )
    
    
    # Add vertical line at calibration cutoff
    fig.add_shape(
        type="line",
        x0=cutoff_date, x1=cutoff_date,
        y0=0, y1=1,
        yref="paper",
        line=dict(
            color="rgba(255, 255, 255, 0.8)",
            width=2,
            dash="dash"
        )
    )
    
    # Add annotation for calibration cutoff
    fig.add_annotation(
        x=cutoff_date,
        y=0.95,
        yref="paper",
        text="Model Calibration Cutoff",
        showarrow=False,
        font=dict(color="#ffffff", size=12),
        bgcolor="rgba(45, 45, 45, 0.8)",
        bordercolor="#ffffff",
        borderwidth=1
    )
    
    # Update layout with dark theme
    fig.update_layout(
        title={
            "text": "VIX Financial Model: Calibration vs. Forecasting",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 20, "color": "#ffffff"},
        },
        xaxis_title="Date",
        yaxis_title="VIX Level",
        hovermode="x unified",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(45, 45, 45, 0.9)",
            bordercolor="#ffffff",
            borderwidth=1,
            font=dict(color="#ffffff"),
        ),
        plot_bgcolor="#1e1e1e",
        paper_bgcolor="#1e1e1e",
        font=dict(color="#ffffff"),
        xaxis=dict(gridcolor="#404040", zerolinecolor="#404040", color="#ffffff"),
        yaxis=dict(gridcolor="#404040", zerolinecolor="#404040", color="#ffffff"),
        height=600,
    )
    
    return fig

@st.cache_data
def load_and_process_data():
    """Load pre-generated VIX simulation data"""
    actual_data, sim_data, cutoff_date, calibration_actual, forecasting_actual = load_vix_simulation_data()
    return actual_data, sim_data, cutoff_date, calibration_actual, forecasting_actual

def main():
    """Main VIX dashboard function"""
    
    # Title and description
    st.title("📊 VIX Financial Model Dashboard")
    st.markdown(
        """
    This dashboard illustrates a financial model calibrated on VIX data:
    - **Calibration Period**: Model trained on data until 3 months ago (3% Gaussian error)
    - **Forecasting Period**: Out-of-sample prediction with increasing uncertainty (3% → 10% error as √t)
    - **Solid coral line**: Actual VIX during calibration period
    - **Dashed blue line**: Median simulation across all periods
    - **Shaded area**: Confidence intervals from Monte Carlo simulations
    """
    )
    
    # Load data
    try:
        actual_data, sim_data, cutoff_date, calibration_data, forecasting_data = load_and_process_data()
        
        # Sidebar controls
        st.sidebar.header("🎛️ Model Controls")
        
        # Confidence interval control
        confidence_level = st.sidebar.slider(
            "Confidence Interval %",
            min_value=50,
            max_value=95,
            value=95,
            step=5,
            help="Select the confidence interval percentage for simulation band",
        )
        
        # Model parameters info
        st.sidebar.markdown("### 📈 Model Parameters")
        st.sidebar.info(
            f"""
        **Data Window**: From October 2024
        **Model**: Mean = VIX + 3% noise
        **Calibration Error**: 10% around mean
        **Forecasting Error**: 10% → 30% as √t
        **Samples**: 100 Monte Carlo paths
        **Cutoff Date**: {cutoff_date.strftime('%Y-%m-%d')}
        """
        )
        
        # Create main plot
        vix_fig = create_vix_plot(actual_data, sim_data, cutoff_date, confidence_level)
        st.plotly_chart(vix_fig, use_container_width=True)
        
        # Summary metrics
        st.markdown("---")
        st.markdown("### 📊 Model Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            latest_actual = actual_data.iloc[-1]
            latest_median = sim_data.median(axis=1).iloc[-1]
            st.metric(
                "Latest Actual VIX",
                f"{latest_actual:.2f}",
                delta=f"{latest_actual - actual_data.iloc[-2]:.2f}",
            )
        
        with col2:
            st.metric(
                "Latest Median Simulation",
                f"{latest_median:.2f}",
                delta=f"{latest_median - sim_data.median(axis=1).iloc[-2]:.2f}",
            )
        
        with col3:
            # Calculate current error level (at end of forecasting period)
            t_final = 1.0  # End of forecasting period
            current_error = 0.10 + (0.30 - 0.10) * np.sqrt(t_final)
            st.metric(
                "Current Model Error",
                f"{current_error*100:.1f}%",
                help="Expected Gaussian error at current time point"
            )
        
        # Data coverage information
        st.markdown("---")
        st.markdown("### 📅 Data Coverage")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(
                f"""
            **Calibration Period**
            - Start: {calibration_data.index[0].strftime('%Y-%m-%d')}
            - End: {calibration_data.index[-1].strftime('%Y-%m-%d')}
            - Days: {len(calibration_data)}
            - Error: 10% around mean
            """
            )
        
        with col2:
            if len(forecasting_data) > 0:
                st.info(
                    f"""
                **Forecasting Period**
                - Start: {forecasting_data.index[0].strftime('%Y-%m-%d')}
                - End: {forecasting_data.index[-1].strftime('%Y-%m-%d')}
                - Days: {len(forecasting_data)}
                - Error: 10% → 30% as √t
                """
                )
            else:
                st.info("**No forecasting data available** (cutoff is today)")
    
    except FileNotFoundError:
        st.error(
            """
        📁 **VIX simulation data file not found!**
        
        Please ensure `data/spoof_vix_data.csv` exists.
        Run `python gen_vix_data.py` to generate the simulation data.
        """
        )
    
    except Exception as e:
        st.error(f"❌ An error occurred while loading the data: {str(e)}")

if __name__ == "__main__":
    main()