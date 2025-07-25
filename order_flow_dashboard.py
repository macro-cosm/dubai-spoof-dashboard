import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from macrocosm_visual.viz_setup import setup_plotly, get_macrocosm_colors

# Setup Macrocosm visual styling
setup_plotly()
colors = get_macrocosm_colors()

# Page configuration
st.set_page_config(
    page_title="Order Flow Analysis Dashboard", page_icon="📊", layout="wide"
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

@st.cache_data
def load_order_flow_data():
    """Load the order flow and cancellation rate data"""
    # Load the concatenated data (realised data with spoofed orders)
    volume_data = np.load("data/dummy_volume.npy")
    rates_data = np.load("data/dummy_rates.npy")
    
    return volume_data, rates_data

def create_combined_plot(volume_data, rates_data):
    """Create combined plot with both densities"""
    
    # Create subplots: 2 rows, 1 column
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Order Flow Imbalance Density", "Cancellation Rate Density"),
        vertical_spacing=0.12
    )
    
    # === ORDER FLOW IMBALANCE PLOT (Top) ===
    
    # Create histogram for realised data (top plot)
    hist_counts, bin_edges = np.histogram(volume_data, bins=100, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Add realised data histogram
    fig.add_trace(
        go.Bar(
            x=bin_centers,
            y=hist_counts,
            name="Realised Data",
            marker=dict(color=colors[0], opacity=0.7),
            width=(bin_edges[1] - bin_edges[0]) * 0.8,
            showlegend=True
        ),
        row=1, col=1
    )
    
    # Add exponential model curve (top plot)
    xs_volume = np.linspace(0, 6, 100)
    ys_volume = np.exp(-xs_volume)  # Exponential distribution with scale=1
    
    fig.add_trace(
        go.Scatter(
            x=xs_volume,
            y=ys_volume,
            mode="lines",
            name="Model",
            line=dict(color=colors[1], width=3),
            showlegend=True
        ),
        row=1, col=1
    )
    
    # === CANCELLATION RATE PLOT (Bottom) ===
    
    # Create histogram for cancellation rates (bottom plot)
    hist_counts_rates, bin_edges_rates = np.histogram(rates_data, bins=100, density=True)
    bin_centers_rates = (bin_edges_rates[:-1] + bin_edges_rates[1:]) / 2
    
    # Add realised data histogram
    fig.add_trace(
        go.Bar(
            x=bin_centers_rates,
            y=hist_counts_rates,
            name="Realised Data",
            marker=dict(color=colors[0], opacity=0.7),
            width=(bin_edges_rates[1] - bin_edges_rates[0]) * 0.8,
            showlegend=False  # Don't show legend for duplicate
        ),
        row=2, col=1
    )
    
    # Add gaussian model curve (bottom plot)
    xs_rates = np.linspace(0, 15, 100)
    var = 1
    loc = 5
    ys_rates = 1 / np.sqrt(2 * np.pi * var) * np.exp(-((xs_rates - loc) ** 2) / (2 * var))
    
    fig.add_trace(
        go.Scatter(
            x=xs_rates,
            y=ys_rates,
            mode="lines",
            name="Model",
            line=dict(color=colors[1], width=3),
            showlegend=False  # Don't show legend for duplicate
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        title={
            "text": "Order Flow Analysis: Model vs Realised Data",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 20, "color": "#ffffff"},
        },
        height=800,
        plot_bgcolor="#1e1e1e",
        paper_bgcolor="#1e1e1e",
        font=dict(color="#ffffff"),
        legend=dict(
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(45, 45, 45, 0.9)",
            bordercolor="#ffffff",
            borderwidth=1,
            font=dict(color="#ffffff"),
        )
    )
    
    # Update x and y axes for both subplots
    fig.update_xaxes(
        gridcolor="#404040", 
        zerolinecolor="#404040", 
        color="#ffffff",
        showticklabels=False  # Remove x-axis labels as requested
    )
    fig.update_yaxes(
        gridcolor="#404040", 
        zerolinecolor="#404040", 
        color="#ffffff",
        showticklabels=False  # Remove y-axis labels as requested
    )
    
    # Set x-axis ranges to match the original plots
    fig.update_xaxes(range=[0, 6], row=1, col=1)  # Order flow plot range
    fig.update_xaxes(range=[0, 15], row=2, col=1)  # Cancellation rate plot range
    
    return fig

def main():
    """Main order flow dashboard function"""
    
    # Title and description
    st.title("📊 Order Flow Analysis Dashboard")
    st.markdown(
        """
    This dashboard analyzes order flow imbalance and cancellation rates:
    - **Order Flow Imbalance**: Exponentially distributed with spoofed orders
    - **Cancellation Rates**: Gaussian distribution with spoofed rates at different location
    - **Model**: Analytical theoretical densities
    - **Realised Data**: Combined samples including spoofed components
    """
    )
    
    # Load data
    try:
        volume_data, rates_data = load_order_flow_data()
        
        # Sidebar information
        st.sidebar.header("📈 Data Summary")
        st.sidebar.info(
            f"""
        **Order Flow Data**
        - Samples: {len(volume_data):,}
        - Model: Exponential (λ=1)
        - Includes: Spoofed orders
        
        **Cancellation Rates**
        - Samples: {len(rates_data):,}
        - Model: Gaussian (μ=5, σ=1)
        - Includes: Spoofed rates
        """
        )
        
        # Create and display the combined plot
        combined_fig = create_combined_plot(volume_data, rates_data)
        st.plotly_chart(combined_fig, use_container_width=True)
        
        # Summary statistics
        st.markdown("---")
        st.markdown("### 📊 Data Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Order Flow Mean", f"{np.mean(volume_data):.3f}")
            st.metric("Order Flow Std", f"{np.std(volume_data):.3f}")
            st.metric("Order Flow Max", f"{np.max(volume_data):.3f}")
        
        with col2:
            st.metric("Cancellation Rate Mean", f"{np.mean(rates_data):.3f}")
            st.metric("Cancellation Rate Std", f"{np.std(rates_data):.3f}")
            st.metric("Cancellation Rate Max", f"{np.max(rates_data):.3f}")
    
    except FileNotFoundError as e:
        st.error(
            """
        📁 **Order flow data files not found!**
        
        Please ensure the following files exist in the `data/` directory:
        - `dummy_volume.npy`
        - `dummy_rates.npy`
        
        Run the `dummy_volume.py` script to generate the data.
        """
        )
    
    except Exception as e:
        st.error(f"❌ An error occurred while loading the data: {str(e)}")

if __name__ == "__main__":
    main()