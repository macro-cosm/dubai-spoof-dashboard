import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from macrocosm_visual.viz_setup import setup_plotly, get_macrocosm_colors

# Setup Macrocosm visual styling
setup_plotly()
colors = get_macrocosm_colors()

# Page configuration
st.set_page_config(
    page_title="GDP Analysis Dashboard",
    page_icon="📈",
    layout="wide"
)

# Custom CSS for dark theme
st.markdown("""
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
""", unsafe_allow_html=True)

def parse_quarter_date(date_str):
    """Convert 'YYYY QN' format to datetime"""
    year, quarter = date_str.split(' Q')
    year = int(year)
    quarter = int(quarter)
    month = (quarter - 1) * 3 + 1  # Q1=Jan, Q2=Apr, Q3=Jul, Q4=Oct
    return pd.Timestamp(year, month, 1)

@st.cache_data
def load_data():
    """Load and process GDP data"""
    # Define cutoff date (matching synthetic data)
    cutoff_date = pd.Timestamp(2019, 1, 1)  # 2019 Q1
    
    # Load actual GDP data
    actual_gdp = pd.read_csv('data/actualGDP.csv', index_col=0)
    actual_gdp = actual_gdp.T  # Transpose to have dates as rows
    actual_gdp.index = [parse_quarter_date(date) for date in actual_gdp.index]
    actual_gdp.columns = ['GDP']
    
    # Filter actual GDP data to cutoff date
    actual_gdp = actual_gdp[actual_gdp.index <= cutoff_date]
    
    # Load simulation data
    nominal_gdp = pd.read_csv('data/test_gdp.csv', index_col=0)
    print(f"Loaded simulation data shape (before transpose): {nominal_gdp.shape}")
    print(f"First 3 rows, first 3 columns:")
    print(nominal_gdp.iloc[:3, :3])
    
    # Transpose and convert dates
    nominal_gdp = nominal_gdp.T
    print(f"After transpose shape: {nominal_gdp.shape}")
    nominal_gdp.index = [parse_quarter_date(date) for date in nominal_gdp.index]
    
    # Filter simulation data to cutoff date
    nominal_gdp = nominal_gdp[nominal_gdp.index <= cutoff_date]
    print(f"After filtering shape: {nominal_gdp.shape}")
    print(f"First 3 dates, first 3 samples after processing:")
    print(nominal_gdp.iloc[:3, :3])
    
    return actual_gdp, nominal_gdp

def create_gdp_plot(actual_gdp, nominal_gdp, confidence_level=95):
    """Create GDP plot with actual data, median simulation, and confidence intervals"""
    
    fig = go.Figure()
    
    # Calculate statistics from simulations
    median_sim = nominal_gdp.median(axis=1)
    
    # Calculate confidence intervals
    lower_percentile = (100 - confidence_level) / 2
    upper_percentile = 100 - lower_percentile
    
    lower_ci = nominal_gdp.quantile(lower_percentile/100, axis=1)
    upper_ci = nominal_gdp.quantile(upper_percentile/100, axis=1)
    
    # Log statistics for debugging
    print(f"\n=== DASHBOARD DEBUG INFO ===")
    print(f"Nominal GDP shape: {nominal_gdp.shape}")
    print(f"First few simulation values (first 3 samples, first 3 dates):")
    print(nominal_gdp.iloc[:3, :3])
    print(f"Confidence level: {confidence_level}%")
    print(f"Lower percentile: {lower_percentile}%, Upper percentile: {upper_percentile}%")
    print(f"Standard deviation across simulations (first 5 dates):")
    std_dev = nominal_gdp.std(axis=1)
    print(std_dev.head())
    print(f"Min std dev: {std_dev.min():.2f}, Max std dev: {std_dev.max():.2f}")
    print(f"Median values (first 5 dates): {median_sim.head().values}")
    print(f"Lower CI values (first 5 dates): {lower_ci.head().values}")
    print(f"Upper CI values (first 5 dates): {upper_ci.head().values}")
    print(f"CI band width (first 5 dates): {(upper_ci - lower_ci).head().values}")
    print(f"=== END DEBUG INFO ===\n")
    
    # Add confidence interval as filled area
    fig.add_trace(go.Scatter(
        x=list(median_sim.index) + list(median_sim.index[::-1]),
        y=list(upper_ci.values) + list(lower_ci.values[::-1]),
        fill='toself',
        fillcolor='rgba(35, 105, 189, 0.2)',  # Semi-transparent blue
        line=dict(color='rgba(255,255,255,0)'),
        name=f'{confidence_level}% Confidence Interval',
        hoverinfo='skip'
    ))
    
    # Add median simulation line (dashed)
    fig.add_trace(go.Scatter(
        x=median_sim.index,
        y=median_sim.values,
        mode='lines',
        name='Median Simulation',
        line=dict(
            color=colors[0],  # Primary blue
            width=3,
            dash='dash'
        ),
        hovertemplate='<b>Median Simulation</b><br>' +
                      'Date: %{x}<br>' +
                      'GDP: $%{y:,.0f}<br>' +
                      '<extra></extra>'
    ))
    
    # Add actual GDP line (solid)
    fig.add_trace(go.Scatter(
        x=actual_gdp.index,
        y=actual_gdp['GDP'],
        mode='lines',
        name='Actual GDP',
        line=dict(
            color=colors[1],  # Coral red
            width=4
        ),
        hovertemplate='<b>Actual GDP</b><br>' +
                      'Date: %{x}<br>' +
                      'GDP: $%{y:,.0f}<br>' +
                      '<extra></extra>'
    ))
    
    # Update layout with dark theme
    fig.update_layout(
        title={
            'text': 'GDP Analysis: Actual vs Simulated Data',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24, 'color': '#ffffff'}
        },
        xaxis_title='Date',
        yaxis_title='GDP (USD)',
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(45, 45, 45, 0.9)',
            bordercolor='#ffffff',
            borderwidth=1,
            font=dict(color='#ffffff')
        ),
        plot_bgcolor='#1e1e1e',
        paper_bgcolor='#1e1e1e',
        font=dict(color='#ffffff'),
        xaxis=dict(
            gridcolor='#404040',
            zerolinecolor='#404040',
            color='#ffffff'
        ),
        yaxis=dict(
            gridcolor='#404040',
            zerolinecolor='#404040',
            color='#ffffff'
        ),
        height=600
    )
    
    # Format y-axis to show values in trillions (merge with existing yaxis settings)
    fig.update_layout(
        yaxis=dict(
            tickformat='$,.1s',
            tickmode='linear',
            dtick=50000000000,  # 50 billion intervals
            gridcolor='#404040',
            zerolinecolor='#404040',
            color='#ffffff'
        )
    )
    
    return fig

def main():
    """Main dashboard function"""
    
    # Title and description
    st.title("📈 GDP Analysis Dashboard")
    st.markdown("""
    This dashboard visualizes actual GDP data alongside simulation results, showing:
    - **Solid line**: Actual GDP data
    - **Dashed line**: Median of all simulations  
    - **Shaded area**: Confidence interval from simulations
    """)
    
    # Load data
    try:
        actual_gdp, nominal_gdp = load_data()
        
        # Sidebar controls
        st.sidebar.header("📊 Chart Controls")
        confidence_level = st.sidebar.slider(
            "Confidence Interval %",
            min_value=50,
            max_value=95,
            value=95,
            step=5,
            help="Select the confidence interval percentage for the simulation band"
        )
        
        # Create and display the plot
        fig = create_gdp_plot(actual_gdp, nominal_gdp, confidence_level)
        st.plotly_chart(fig, use_container_width=True)
        
        # Data summary
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Latest Actual GDP",
                f"${actual_gdp['GDP'].iloc[-1]/1e12:.2f}T",
                delta=f"{((actual_gdp['GDP'].iloc[-1] - actual_gdp['GDP'].iloc[-2])/actual_gdp['GDP'].iloc[-2]*100):.1f}%"
            )
        
        with col2:
            latest_median = nominal_gdp.median(axis=1).iloc[-1]
            st.metric(
                "Latest Median Simulation",
                f"${latest_median/1e12:.2f}T"
            )
        
        with col3:
            st.metric(
                "Number of Simulations",
                f"{nominal_gdp.shape[1]:,}"
            )
        
        # Data period information
        st.markdown("---")
        st.markdown("### 📅 Data Coverage")
        col1, col2 = st.columns(2)
        
        with col1:
            start_quarter = (actual_gdp.index[0].month - 1) // 3 + 1
            end_quarter = (actual_gdp.index[-1].month - 1) // 3 + 1
            st.info(f"""
            **Actual GDP Data**
            - Start: {actual_gdp.index[0].year} Q{start_quarter}
            - End: {actual_gdp.index[-1].year} Q{end_quarter}
            - Observations: {len(actual_gdp)}
            """)
        
        with col2:
            sim_start_quarter = (nominal_gdp.index[0].month - 1) // 3 + 1
            sim_end_quarter = (nominal_gdp.index[-1].month - 1) // 3 + 1
            st.info(f"""
            **Simulation Data**
            - Start: {nominal_gdp.index[0].year} Q{sim_start_quarter}
            - End: {nominal_gdp.index[-1].year} Q{sim_end_quarter}
            - Simulations: {nominal_gdp.shape[1]}
            """)
        
    except FileNotFoundError as e:
        st.error(f"""
        📁 **Data files not found!**
        
        Please ensure the following files exist in the `data/` directory:
        - `actualGDP.csv`
        - `test_gdp.csv`
        
        Error details: {str(e)}
        """)
    
    except Exception as e:
        st.error(f"❌ An error occurred while loading the data: {str(e)}")

if __name__ == "__main__":
    main()