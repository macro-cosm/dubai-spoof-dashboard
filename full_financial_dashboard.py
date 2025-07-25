import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from macrocosm_visual.viz_setup import setup_plotly, get_macrocosm_colors

# Setup Macrocosm visual styling
setup_plotly()
colors = get_macrocosm_colors()

# Page configuration
st.set_page_config(
    page_title="Dubai Financial Risk Model", page_icon="🏙️", layout="wide"
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

# ===== ORDER FLOW FUNCTIONS =====


@st.cache_data
def load_order_flow_data():
    """Load the order flow and cancellation rate data"""
    volume_data = np.load("data/dummy_volume.npy")
    rates_data = np.load("data/dummy_rates.npy")
    return volume_data, rates_data


def create_order_flow_plots(volume_data, rates_data):
    """Create order flow analysis plots"""
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Order Flow Imbalance Density", "Cancellation Rate Density"),
        vertical_spacing=0.15,
    )

    # === ORDER FLOW IMBALANCE PLOT ===
    hist_counts, bin_edges = np.histogram(volume_data, bins=100, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    fig.add_trace(
        go.Bar(
            x=bin_centers,
            y=hist_counts,
            name="Realised Data",
            marker=dict(color=colors[0], opacity=0.7),
            width=(bin_edges[1] - bin_edges[0]) * 0.8,
            showlegend=True,
        ),
        row=1,
        col=1,
    )

    xs_volume = np.linspace(0, 6, 100)
    ys_volume = np.exp(-xs_volume)

    fig.add_trace(
        go.Scatter(
            x=xs_volume,
            y=ys_volume,
            mode="lines",
            name="Model",
            line=dict(color=colors[1], width=3),
            showlegend=True,
        ),
        row=1,
        col=1,
    )

    # === CANCELLATION RATE PLOT ===
    hist_counts_rates, bin_edges_rates = np.histogram(
        rates_data, bins=100, density=True
    )
    bin_centers_rates = (bin_edges_rates[:-1] + bin_edges_rates[1:]) / 2

    fig.add_trace(
        go.Bar(
            x=bin_centers_rates,
            y=hist_counts_rates,
            name="Realised Data",
            marker=dict(color=colors[0], opacity=0.7),
            width=(bin_edges_rates[1] - bin_edges_rates[0]) * 0.8,
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    xs_rates = np.linspace(0, 15, 100)
    var = 1
    loc = 5
    ys_rates = (
        1 / np.sqrt(2 * np.pi * var) * np.exp(-((xs_rates - loc) ** 2) / (2 * var))
    )

    fig.add_trace(
        go.Scatter(
            x=xs_rates,
            y=ys_rates,
            mode="lines",
            name="Model",
            line=dict(color=colors[1], width=3),
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        height=600,
        plot_bgcolor="#1e1e1e",
        paper_bgcolor="#1e1e1e",
        font=dict(color="#ffffff"),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(45, 45, 45, 0.9)",
            bordercolor="#ffffff",
            borderwidth=1,
            font=dict(color="#ffffff"),
        ),
    )

    fig.update_xaxes(
        gridcolor="#404040",
        zerolinecolor="#404040",
        color="#ffffff",
        showticklabels=False,
    )
    fig.update_yaxes(
        gridcolor="#404040",
        zerolinecolor="#404040",
        color="#ffffff",
        showticklabels=False,
    )
    fig.update_xaxes(range=[0, 4.5], row=1, col=1)
    fig.update_xaxes(range=[1.5, 10.5], row=2, col=1)

    return fig


# ===== RISK INDEX FUNCTIONS =====


def load_risk_simulation_data():
    """Load pre-generated risk index simulation data and filter to start from October 2024"""
    sim_data = pd.read_csv("data/spoof_vix_data.csv", index_col=0)
    sim_data.index = pd.to_datetime(sim_data.index, utc=True)

    october_2024 = pd.Timestamp("2024-10-01").tz_localize("UTC")
    sim_data = sim_data[sim_data.index >= october_2024]

    actual_data = sim_data["Actual_VIX"]
    period_data = sim_data["Period"]

    calibration_mask = period_data == "Calibration"
    if calibration_mask.any():
        cutoff_date = sim_data[calibration_mask].index[-1]
    else:
        today = datetime.now()
        cutoff_date = pd.Timestamp(today - timedelta(days=90)).tz_localize("UTC")

    sample_cols = [col for col in sim_data.columns if col.startswith("Sample_")]
    simulation_df = sim_data[sample_cols]

    calibration_actual = actual_data[calibration_mask]
    forecasting_actual = actual_data[period_data == "Forecasting"]

    return (
        actual_data,
        simulation_df,
        cutoff_date,
        calibration_actual,
        forecasting_actual,
    )


def format_index_compact(value):
    """Format index values in a compact way"""
    return f"{value:.2f}"


def create_risk_index_plot(actual_data, sim_data, cutoff_date, confidence_level=95):
    """Create Total Risk Index plot"""
    fig = go.Figure()

    median_sim = sim_data.median(axis=1)
    lower_percentile = (100 - confidence_level) / 2
    upper_percentile = 100 - lower_percentile
    lower_ci = sim_data.quantile(lower_percentile / 100, axis=1)
    upper_ci = sim_data.quantile(upper_percentile / 100, axis=1)

    calibration_actual = actual_data[actual_data.index <= cutoff_date]

    # Add actual data first (behind)
    fig.add_trace(
        go.Scatter(
            x=calibration_actual.index,
            y=calibration_actual.values,
            mode="lines",
            name="Actual Risk Index (Calibration Period)",
            line=dict(color=colors[1], width=6),
            hovertemplate="<b>Actual Risk Index (Calibration)</b><br>"
            + "Date: %{x}<br>"
            + "Index: %{y:.2f}<br>"
            + "<extra></extra>",
        )
    )

    # Add confidence interval
    fig.add_trace(
        go.Scatter(
            x=list(median_sim.index) + list(median_sim.index[::-1]),
            y=list(upper_ci.values) + list(lower_ci.values[::-1]),
            fill="toself",
            fillcolor="rgba(35, 105, 189, 0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            name=f"{confidence_level}% Confidence Interval",
            hoverinfo="skip",
            showlegend=True,
        )
    )

    # Add median simulation
    fig.add_trace(
        go.Scatter(
            x=median_sim.index,
            y=median_sim.values,
            mode="lines",
            name="Median Simulation",
            line=dict(color=colors[0], width=3, dash="dash"),
            hovertemplate="<b>Median Simulation</b><br>"
            + "Date: %{x}<br>"
            + "Index: %{y:.2f}<br>"
            + "<extra></extra>",
        )
    )

    # Add calibration cutoff line
    fig.add_shape(
        type="line",
        x0=cutoff_date,
        x1=cutoff_date,
        y0=0,
        y1=1,
        yref="paper",
        line=dict(color="rgba(255, 255, 255, 0.8)", width=2, dash="dash"),
    )

    fig.add_annotation(
        x=cutoff_date,
        y=0.95,
        yref="paper",
        text="Model Calibration Cutoff",
        showarrow=False,
        font=dict(color="#ffffff", size=10),
        bgcolor="rgba(45, 45, 45, 0.8)",
        bordercolor="#ffffff",
        borderwidth=1,
    )

    fig.update_layout(
        title={
            "text": "Total Risk Index",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 16, "color": "#ffffff"},
        },
        xaxis_title="Date",
        yaxis_title="Risk Index",
        hovermode="x unified",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(45, 45, 45, 0.9)",
            bordercolor="#ffffff",
            borderwidth=1,
            font=dict(color="#ffffff", size=10),
        ),
        plot_bgcolor="#1e1e1e",
        paper_bgcolor="#1e1e1e",
        font=dict(color="#ffffff"),
        xaxis=dict(gridcolor="#404040", zerolinecolor="#404040", color="#ffffff"),
        yaxis=dict(gridcolor="#404040", zerolinecolor="#404040", color="#ffffff"),
        height=300,
        margin=dict(l=50, r=50, t=50, b=50),
    )

    return fig


# ===== MARKET DEPTH FUNCTIONS =====


def load_depth_simulation_data():
    """Load pre-generated market depth simulation data and filter to start from October 2024"""
    sim_data = pd.read_csv("data/spoof_sp500_data.csv", index_col=0)
    sim_data.index = pd.to_datetime(sim_data.index, utc=True)

    october_2024 = pd.Timestamp("2024-10-01").tz_localize("UTC")
    sim_data = sim_data[sim_data.index >= october_2024]

    actual_data = sim_data["Actual_Volume"]
    period_data = sim_data["Period"]

    calibration_mask = period_data == "Calibration"
    if calibration_mask.any():
        cutoff_date = sim_data[calibration_mask].index[-1]
    else:
        today = datetime.now()
        cutoff_date = pd.Timestamp(today - timedelta(days=90)).tz_localize("UTC")

    sample_cols = [col for col in sim_data.columns if col.startswith("Sample_")]
    simulation_df = sim_data[sample_cols]

    calibration_actual = actual_data[calibration_mask]
    forecasting_actual = actual_data[period_data == "Forecasting"]

    return (
        actual_data,
        simulation_df,
        cutoff_date,
        calibration_actual,
        forecasting_actual,
    )


def format_depth_compact(value):
    """Format depth values in a compact way"""
    if abs(value) >= 1e12:
        return f"{value/1e12:.1f}T"
    elif abs(value) >= 1e9:
        return f"{value/1e9:.1f}B"
    elif abs(value) >= 1e6:
        return f"{value/1e6:.1f}M"
    else:
        return f"{value:,.0f}"


def create_market_depth_plot(actual_data, sim_data, cutoff_date, confidence_level=95):
    """Create Market Depth Index plot"""
    fig = go.Figure()

    median_sim = sim_data.median(axis=1)
    lower_percentile = (100 - confidence_level) / 2
    upper_percentile = 100 - lower_percentile
    lower_ci = sim_data.quantile(lower_percentile / 100, axis=1)
    upper_ci = sim_data.quantile(upper_percentile / 100, axis=1)

    calibration_actual = actual_data[actual_data.index <= cutoff_date]

    # Add actual data first (behind)
    actual_customdata = []
    for val in calibration_actual:
        actual_customdata.append(format_depth_compact(val))

    fig.add_trace(
        go.Scatter(
            x=calibration_actual.index,
            y=calibration_actual.values,
            mode="lines",
            name="Actual Depth Index (Calibration Period)",
            line=dict(color=colors[1], width=6),
            customdata=actual_customdata,
            hovertemplate="<b>Actual Depth Index (Calibration)</b><br>"
            + "Date: %{x}<br>"
            + "Index: %{customdata}<br>"
            + "<extra></extra>",
        )
    )

    # Add confidence interval
    fig.add_trace(
        go.Scatter(
            x=list(median_sim.index) + list(median_sim.index[::-1]),
            y=list(upper_ci.values) + list(lower_ci.values[::-1]),
            fill="toself",
            fillcolor="rgba(35, 105, 189, 0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            name=f"{confidence_level}% Confidence Interval",
            hoverinfo="skip",
            showlegend=True,
        )
    )

    # Add median simulation
    customdata = []
    for i, idx in enumerate(median_sim.index):
        lower_val = lower_ci.iloc[i]
        upper_val = upper_ci.iloc[i]
        median_val = median_sim.iloc[i]
        customdata.append(
            [
                format_depth_compact(median_val),
                format_depth_compact(lower_val),
                format_depth_compact(upper_val),
                confidence_level,
            ]
        )

    fig.add_trace(
        go.Scatter(
            x=median_sim.index,
            y=median_sim.values,
            mode="lines",
            name="Median Simulation",
            line=dict(color=colors[0], width=3, dash="dash"),
            customdata=customdata,
            hovertemplate="<b>Median Simulation</b><br>"
            + "Date: %{x}<br>"
            + "Index: %{customdata[0]} (CI: %{customdata[1]} - %{customdata[2]})<br>"
            + "<extra></extra>",
        )
    )

    # Add calibration cutoff line
    fig.add_shape(
        type="line",
        x0=cutoff_date,
        x1=cutoff_date,
        y0=0,
        y1=1,
        yref="paper",
        line=dict(color="rgba(255, 255, 255, 0.8)", width=2, dash="dash"),
    )

    fig.add_annotation(
        x=cutoff_date,
        y=0.95,
        yref="paper",
        text="Model Calibration Cutoff",
        showarrow=False,
        font=dict(color="#ffffff", size=10),
        bgcolor="rgba(45, 45, 45, 0.8)",
        bordercolor="#ffffff",
        borderwidth=1,
    )

    fig.update_layout(
        title={
            "text": "Market Depth Index",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 16, "color": "#ffffff"},
        },
        xaxis_title="Date",
        yaxis_title="Depth Index",
        hovermode="x unified",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(45, 45, 45, 0.9)",
            bordercolor="#ffffff",
            borderwidth=1,
            font=dict(color="#ffffff", size=10),
        ),
        plot_bgcolor="#1e1e1e",
        paper_bgcolor="#1e1e1e",
        font=dict(color="#ffffff"),
        xaxis=dict(gridcolor="#404040", zerolinecolor="#404040", color="#ffffff"),
        yaxis=dict(
            gridcolor="#404040",
            zerolinecolor="#404040",
            color="#ffffff",
            tickformat=".2s",
        ),
        height=300,
        margin=dict(l=50, r=50, t=50, b=50),
    )

    return fig


# ===== MAIN DASHBOARD =====


def main():
    """Main Dubai Financial Risk Model dashboard"""

    # Title and description
    st.title("🏙️ Dubai Financial Risk Model")
    st.markdown(
        """
    Comprehensive financial risk analysis dashboard combining order flow dynamics, 
    total risk assessment, and market depth indicators for Dubai financial markets.
    """
    )

    # Load all data
    try:
        # Load order flow data
        volume_data, rates_data = load_order_flow_data()

        # Load risk index data
        (
            risk_actual_data,
            risk_sim_data,
            risk_cutoff_date,
            risk_calibration_data,
            risk_forecasting_data,
        ) = load_risk_simulation_data()

        # Load market depth data
        (
            depth_actual_data,
            depth_sim_data,
            depth_cutoff_date,
            depth_calibration_data,
            depth_forecasting_data,
        ) = load_depth_simulation_data()

        # Sidebar controls
        st.sidebar.header("🎛️ Model Controls")

        confidence_level = st.sidebar.slider(
            "Confidence Interval %",
            min_value=50,
            max_value=95,
            value=95,
            step=5,
            help="Select the confidence interval percentage for simulation bands",
        )

        st.sidebar.markdown("### 📊 Data Summary")
        st.sidebar.info(
            f"""
        **Order Flow Analysis**
        - Flow samples: {len(volume_data):,}
        - Rate samples: {len(rates_data):,}
        
        **Risk Index Model**
        - Calibration: {len(risk_calibration_data)} days
        - Forecasting: {len(risk_forecasting_data)} days
        
        **Market Depth Model**
        - Calibration: {len(depth_calibration_data)} days
        - Forecasting: {len(depth_forecasting_data)} days
        """
        )

        # Create main layout: 2 columns
        col1, col2 = st.columns([1, 1])

        # Left Column: Order Flow Analysis
        with col1:
            st.markdown("### Order Flow Analysis")
            order_flow_fig = create_order_flow_plots(volume_data, rates_data)
            st.plotly_chart(order_flow_fig, use_container_width=True)

        # Right Column: Risk and Depth Indices
        with col2:
            # Top: Total Risk Index
            st.markdown("### Total Risk Index")
            risk_fig = create_risk_index_plot(
                risk_actual_data, risk_sim_data, risk_cutoff_date, confidence_level
            )
            st.plotly_chart(risk_fig, use_container_width=True)

            # Bottom: Market Depth Index
            st.markdown("### Market Depth Index")
            depth_fig = create_market_depth_plot(
                depth_actual_data, depth_sim_data, depth_cutoff_date, confidence_level
            )
            st.plotly_chart(depth_fig, use_container_width=True)

        # Summary metrics at the bottom
        st.markdown("---")
        st.markdown("### 📊 Latest Indicators")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Order Flow Mean", f"{np.mean(volume_data):.3f}")

        with col2:
            st.metric("Cancellation Rate Mean", f"{np.mean(rates_data):.3f}")

        with col3:
            latest_risk = risk_actual_data.iloc[-1]
            st.metric("Latest Risk Index", f"{latest_risk:.2f}")

        with col4:
            latest_depth = depth_actual_data.iloc[-1]
            st.metric("Latest Depth Index", format_depth_compact(latest_depth))

    except FileNotFoundError as e:
        st.error(
            """
        📁 **Data files not found!**
        
        Please ensure the following files exist:
        - `data/dummy_volume.npy`
        - `data/dummy_rates.npy`
        - `data/spoof_vix_data.csv`
        - `data/spoof_sp500_data.csv`
        """
        )

    except Exception as e:
        st.error(f"❌ An error occurred while loading the data: {str(e)}")


if __name__ == "__main__":
    main()
