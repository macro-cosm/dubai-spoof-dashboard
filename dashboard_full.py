import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from macrocosm_visual.viz_setup import setup_plotly, get_macrocosm_colors

# Setup Macrocosm visual styling
setup_plotly()
colors = get_macrocosm_colors()

# Page configuration
st.set_page_config(
    page_title="Economic Analysis Dashboard", page_icon="📈", layout="wide"
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


def parse_quarter_date(date_str):
    """Convert 'YYYY QN' format to datetime, shifting 2014 to 2024"""
    year, quarter = date_str.split(" Q")
    year = int(year)
    quarter = int(quarter)
    # Shift timeline: 2014 becomes 2024
    shifted_year = year + 10
    month = (quarter - 1) * 3 + 1  # Q1=Jan, Q2=Apr, Q3=Jul, Q4=Oct
    return pd.Timestamp(year=shifted_year, month=month, day=1)


@st.cache_data
def load_gdp_data():
    """Load and process GDP data"""
    # Define cutoff date for all data (2019 Q1 original = 2029 Q1 shifted)
    data_cutoff_date = pd.Timestamp('2029-01-01')  # 2029 Q1 (was 2019 Q1)
    # Define cutoff for real data (2015 Q2 original = 2025 Q2 shifted)
    real_data_cutoff_date = pd.Timestamp('2025-04-01')  # 2025 Q2 (was 2015 Q2)

    # Load actual GDP data
    actual_gdp = pd.read_csv("data/actualGDP.csv", index_col=0)
    actual_gdp = actual_gdp.T  # Transpose to have dates as rows
    actual_gdp.index = [parse_quarter_date(date) for date in actual_gdp.index]
    actual_gdp.columns = ["GDP"]

    # Filter actual GDP data to real data cutoff (don't show real data after 2025 Q2)
    actual_gdp = actual_gdp[actual_gdp.index <= real_data_cutoff_date]

    # Load simulation data
    nominal_gdp = pd.read_csv("data/test_gdp.csv", index_col=0)

    # Transpose and convert dates
    nominal_gdp = nominal_gdp.T
    nominal_gdp.index = [parse_quarter_date(date) for date in nominal_gdp.index]

    # Filter simulation data to full cutoff date
    nominal_gdp = nominal_gdp[nominal_gdp.index <= data_cutoff_date]

    return actual_gdp, nominal_gdp, real_data_cutoff_date


@st.cache_data
def load_cpi_data():
    """Load and process CPI data"""
    # Define cutoff date for all data (2019 Q1 original = 2029 Q1 shifted)
    data_cutoff_date = pd.Timestamp('2029-01-01')  # 2029 Q1 (was 2019 Q1)
    # Define cutoff for real data (2015 Q2 original = 2025 Q2 shifted)
    real_data_cutoff_date = pd.Timestamp('2025-04-01')  # 2025 Q2 (was 2015 Q2)

    # Load actual CPI data
    actual_cpi = pd.read_csv("data/actualCPI.csv", index_col=0)
    actual_cpi = actual_cpi.T  # Transpose to have dates as rows
    actual_cpi.index = [parse_quarter_date(date) for date in actual_cpi.index]
    actual_cpi.columns = ["CPI"]

    # Filter actual CPI data to real data cutoff (don't show real data after 2025 Q2)
    actual_cpi = actual_cpi[actual_cpi.index <= real_data_cutoff_date]

    # Load simulation data
    sim_cpi = pd.read_csv("data/model_outputtest.csv", index_col=0)

    # Transpose and convert dates
    sim_cpi = sim_cpi.T
    sim_cpi.index = [parse_quarter_date(date) for date in sim_cpi.index]

    # Filter simulation data to full cutoff date
    sim_cpi = sim_cpi[sim_cpi.index <= data_cutoff_date]

    return actual_cpi, sim_cpi, real_data_cutoff_date


def format_currency_compact(value):
    """Format currency values in a compact way (e.g., $15.2T instead of $15,200,000,000,000)"""
    if abs(value) >= 1e12:
        return f"${value/1e12:.1f}T"
    elif abs(value) >= 1e9:
        return f"${value/1e9:.1f}B"
    elif abs(value) >= 1e6:
        return f"${value/1e6:.1f}M"
    else:
        return f"${value:,.0f}"

def create_scenario_data(data, forecast_cutoff_date, adjustment_pct, is_gdp=True):
    """
    Create scenario data that matches original up to forecast_cutoff_date,
    then applies exponential adjustment after that point.
    
    Parameters:
    - data: Original simulation data (DataFrame)
    - forecast_cutoff_date: Date after which to apply scenario
    - adjustment_pct: Percentage adjustment per quarter (e.g., 1.0 for 1%)
    - is_gdp: If True, applies positive adjustment; if False (CPI), applies negative
    
    Returns:
    - DataFrame with scenario-adjusted data
    """
    scenario_data = data.copy()
    
    # Find the cutoff index
    cutoff_idx = None
    for i, date in enumerate(data.index):
        if date > forecast_cutoff_date:
            cutoff_idx = i
            break
    
    if cutoff_idx is None:
        # No dates after cutoff, return original data
        return scenario_data
    
    # Apply exponential adjustment after cutoff
    adjustment_factor = 1 + (adjustment_pct / 100)
    if not is_gdp:  # For CPI, invert the adjustment
        adjustment_factor = 1 - (adjustment_pct / 100)
    
    for i in range(cutoff_idx, len(data.index)):
        # T is the number of quarters after the cutoff
        T = i - cutoff_idx + 1
        multiplier = adjustment_factor ** T
        scenario_data.iloc[i] = data.iloc[i] * multiplier
    
    return scenario_data


# @st.cache_data
# def load_unemployment_data():
#     """Load and process unemployment data"""
#     # Define cutoff date
#     cutoff_date = pd.Timestamp(2019, 1, 1)  # 2019 Q1

#     # Load actual unemployment data
#     actual_unem = pd.read_csv("data/actualUnem.csv", index_col=0)
#     actual_unem = actual_unem.T  # Transpose to have dates as rows
#     actual_unem.index = [parse_quarter_date(date) for date in actual_unem.index]
#     actual_unem.columns = ["Unemployment"]

#     # Filter actual unemployment data to cutoff date
#     actual_unem = actual_unem[actual_unem.index <= cutoff_date]

#     # Load simulation data
#     sim_unem = pd.read_csv("data/unemployment.csv", index_col=0)

#     # Transpose and convert dates
#     sim_unem = sim_unem.T
#     sim_unem.index = [parse_quarter_date(date) for date in sim_unem.index]

#     # Filter simulation data to cutoff date
#     sim_unem = sim_unem[sim_unem.index <= cutoff_date]

#     return actual_unem, sim_unem


def create_gdp_plot(actual_gdp, nominal_gdp, forecast_cutoff_date, confidence_level=95, scenario_adjustment=0.0):
    """Create GDP plot with actual data, median simulation, and confidence intervals"""

    fig = go.Figure()

    # Calculate statistics from simulations
    median_sim = nominal_gdp.median(axis=1)

    # Calculate confidence intervals
    lower_percentile = (100 - confidence_level) / 2
    upper_percentile = 100 - lower_percentile

    lower_ci = nominal_gdp.quantile(lower_percentile / 100, axis=1)
    upper_ci = nominal_gdp.quantile(upper_percentile / 100, axis=1)

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
    
    # Store CI data for compact hover info
    ci_data = {
        'upper': upper_ci,
        'lower': lower_ci,
        'level': confidence_level
    }

    # Create scenario data if adjustment is provided
    if scenario_adjustment != 0.0:
        nominal_gdp_scenario = create_scenario_data(nominal_gdp, forecast_cutoff_date, scenario_adjustment, is_gdp=True)
        median_sim_scenario = nominal_gdp_scenario.median(axis=1)
        lower_ci_scenario = nominal_gdp_scenario.quantile(lower_percentile/100, axis=1)
        upper_ci_scenario = nominal_gdp_scenario.quantile(upper_percentile/100, axis=1)
        
        # Add scenario confidence interval as filled area
        fig.add_trace(
            go.Scatter(
                x=list(median_sim_scenario.index) + list(median_sim_scenario.index[::-1]),
                y=list(upper_ci_scenario.values) + list(lower_ci_scenario.values[::-1]),
                fill="toself",
                fillcolor="rgba(255, 165, 0, 0.2)",  # Semi-transparent orange
                line=dict(color="rgba(255,255,255,0)"),
                name=f"{confidence_level}% CI (Scenario 1)",
                hoverinfo="skip",
                showlegend=True
            )
        )
        
        # Store scenario CI data for compact hover info
        scenario_ci_data = {
            'upper': upper_ci_scenario,
            'lower': lower_ci_scenario,
            'level': confidence_level
        }
        
        # Add scenario median simulation line (dashed, orange) with CI info
        scenario_customdata = []
        for i, idx in enumerate(median_sim_scenario.index):
            lower_val = scenario_ci_data['lower'].iloc[i]
            upper_val = scenario_ci_data['upper'].iloc[i]
            median_val = median_sim_scenario.iloc[i]
            scenario_customdata.append([
                format_currency_compact(median_val),
                format_currency_compact(lower_val), 
                format_currency_compact(upper_val),
                scenario_ci_data['level']
            ])
        
        fig.add_trace(
            go.Scatter(
                x=median_sim_scenario.index,
                y=median_sim_scenario.values,
                mode="lines",
                name="Median Simulation (Scenario 1)",
                line=dict(color="orange", width=3, dash="dash"),
                customdata=scenario_customdata,
                hovertemplate="<b>Median Simulation (Scenario 1)</b><br>"
                + "Date: %{x}<br>"
                + "GDP: %{customdata[0]} (CI: %{customdata[1]} - %{customdata[2]})<br>"
                + "<extra></extra>",
            )
        )

    # Add median simulation line (dashed) with CI info
    customdata = []
    for i, idx in enumerate(median_sim.index):
        lower_val = ci_data['lower'].iloc[i]
        upper_val = ci_data['upper'].iloc[i]
        median_val = median_sim.iloc[i]
        customdata.append([
            format_currency_compact(median_val),
            format_currency_compact(lower_val), 
            format_currency_compact(upper_val),
            ci_data['level']
        ])
    
    sim_name = "Median Simulation (Business as Usual)" if scenario_adjustment != 0.0 else "Median Simulation"
    fig.add_trace(
        go.Scatter(
            x=median_sim.index,
            y=median_sim.values,
            mode="lines",
            name=sim_name,
            line=dict(color=colors[0], width=3, dash="dash"),  # Primary blue
            customdata=customdata,
            hovertemplate=f"<b>{sim_name}</b><br>"
            + "Date: %{x}<br>"
            + "GDP: %{customdata[0]} (CI: %{customdata[1]} - %{customdata[2]})<br>"
            + "<extra></extra>",
        )
    )

    # Add actual GDP line (solid) with compact formatting
    actual_customdata = []
    for val in actual_gdp["GDP"]:
        actual_customdata.append(format_currency_compact(val))
    
    fig.add_trace(
        go.Scatter(
            x=actual_gdp.index,
            y=actual_gdp["GDP"],
            mode="lines",
            name="Actual GDP",
            line=dict(color=colors[1], width=4),  # Coral red
            customdata=actual_customdata,
            hovertemplate="<b>Actual GDP</b><br>"
            + "Date: %{x}<br>"
            + "GDP: %{customdata}<br>"
            + "<extra></extra>",
        )
    )

    # Add forecast indicator line using shapes
    fig.add_shape(
        type="line",
        x0=forecast_cutoff_date, x1=forecast_cutoff_date,
        y0=0, y1=1,
        yref="paper",
        line=dict(
            color="rgba(255, 255, 255, 0.6)",
            width=2,
            dash="dash"
        )
    )
    
    # Add forecast annotation
    fig.add_annotation(
        x=forecast_cutoff_date,
        y=0.95,
        yref="paper",
        text="Forecast →",
        showarrow=False,
        font=dict(color="#ffffff", size=12),
        bgcolor="rgba(45, 45, 45, 0.8)",
        bordercolor="#ffffff",
        borderwidth=1
    )

    # Update layout with dark theme
    fig.update_layout(
        title={
            "text": "GDP Analysis: Actual vs Simulated Data",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 20, "color": "#ffffff"},
        },
        xaxis_title="Date",
        yaxis_title="GDP (USD)",
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
        yaxis=dict(
            gridcolor="#404040",
            zerolinecolor="#404040",
            color="#ffffff",
            tickformat="$,.1s",
            tickmode="linear",
            dtick=50000000000,  # 50 billion intervals
        ),
        height=600,
    )

    return fig


def create_cpi_plot(actual_cpi, sim_cpi, forecast_cutoff_date, confidence_level=95, scenario_adjustment=0.0):
    """Create CPI plot with actual data, median simulation, and confidence intervals"""

    fig = go.Figure()

    # Calculate statistics from simulations
    median_sim = sim_cpi.median(axis=1)

    # Calculate confidence intervals
    lower_percentile = (100 - confidence_level) / 2
    upper_percentile = 100 - lower_percentile

    lower_ci = sim_cpi.quantile(lower_percentile / 100, axis=1)
    upper_ci = sim_cpi.quantile(upper_percentile / 100, axis=1)

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
    
    # Store CI data for compact hover info
    ci_data = {
        'upper': upper_ci,
        'lower': lower_ci,
        'level': confidence_level
    }

    # Create scenario data if adjustment is provided
    if scenario_adjustment != 0.0:
        sim_cpi_scenario = create_scenario_data(sim_cpi, forecast_cutoff_date, scenario_adjustment, is_gdp=False)
        median_sim_scenario = sim_cpi_scenario.median(axis=1)
        lower_ci_scenario = sim_cpi_scenario.quantile(lower_percentile/100, axis=1)
        upper_ci_scenario = sim_cpi_scenario.quantile(upper_percentile/100, axis=1)
        
        # Add scenario confidence interval as filled area
        fig.add_trace(
            go.Scatter(
                x=list(median_sim_scenario.index) + list(median_sim_scenario.index[::-1]),
                y=list(upper_ci_scenario.values) + list(lower_ci_scenario.values[::-1]),
                fill="toself",
                fillcolor="rgba(255, 165, 0, 0.2)",  # Semi-transparent orange
                line=dict(color="rgba(255,255,255,0)"),
                name=f"{confidence_level}% CI (Scenario 1)",
                hoverinfo="skip",
                showlegend=True
            )
        )
        
        # Store scenario CI data for compact hover info
        scenario_ci_data = {
            'upper': upper_ci_scenario,
            'lower': lower_ci_scenario,
            'level': confidence_level
        }
        
        # Add scenario median simulation line (dashed, orange) with CI info
        scenario_customdata = []
        for i, idx in enumerate(median_sim_scenario.index):
            lower_val = scenario_ci_data['lower'].iloc[i]
            upper_val = scenario_ci_data['upper'].iloc[i]
            median_val = median_sim_scenario.iloc[i]
            scenario_customdata.append([
                f"{median_val:.3f}",
                f"{lower_val:.3f}", 
                f"{upper_val:.3f}",
                scenario_ci_data['level']
            ])
        
        fig.add_trace(
            go.Scatter(
                x=median_sim_scenario.index,
                y=median_sim_scenario.values,
                mode="lines",
                name="Median Simulation (Scenario 1)",
                line=dict(color="orange", width=3, dash="dash"),
                customdata=scenario_customdata,
                hovertemplate="<b>Median Simulation (Scenario 1)</b><br>"
                + "Date: %{x}<br>"
                + "CPI: %{customdata[0]} (CI: %{customdata[1]} - %{customdata[2]})<br>"
                + "<extra></extra>",
            )
        )

    # Add median simulation line (dashed) with CI info
    customdata = []
    for i, idx in enumerate(median_sim.index):
        lower_val = ci_data['lower'].iloc[i]
        upper_val = ci_data['upper'].iloc[i]
        median_val = median_sim.iloc[i]
        customdata.append([
            f"{median_val:.3f}",
            f"{lower_val:.3f}", 
            f"{upper_val:.3f}",
            ci_data['level']
        ])
    
    sim_name = "Median Simulation (Business as Usual)" if scenario_adjustment != 0.0 else "Median Simulation"
    fig.add_trace(
        go.Scatter(
            x=median_sim.index,
            y=median_sim.values,
            mode="lines",
            name=sim_name,
            line=dict(color=colors[0], width=3, dash="dash"),  # Primary blue
            customdata=customdata,
            hovertemplate=f"<b>{sim_name}</b><br>"
            + "Date: %{x}<br>"
            + "CPI: %{customdata[0]} (CI: %{customdata[1]} - %{customdata[2]})<br>"
            + "<extra></extra>",
        )
    )

    # Add actual CPI line (solid) with compact formatting
    actual_customdata = []
    for val in actual_cpi["CPI"]:
        actual_customdata.append(f"{val:.3f}")
    
    fig.add_trace(
        go.Scatter(
            x=actual_cpi.index,
            y=actual_cpi["CPI"],
            mode="lines",
            name="Actual CPI",
            line=dict(color=colors[1], width=4),  # Coral red
            customdata=actual_customdata,
            hovertemplate="<b>Actual CPI</b><br>"
            + "Date: %{x}<br>"
            + "CPI: %{customdata}<br>"
            + "<extra></extra>",
        )
    )

    # Add forecast indicator line using shapes
    fig.add_shape(
        type="line",
        x0=forecast_cutoff_date, x1=forecast_cutoff_date,
        y0=0, y1=1,
        yref="paper",
        line=dict(
            color="rgba(255, 255, 255, 0.6)",
            width=2,
            dash="dash"
        )
    )
    
    # Add forecast annotation
    fig.add_annotation(
        x=forecast_cutoff_date,
        y=0.95,
        yref="paper",
        text="Forecast →",
        showarrow=False,
        font=dict(color="#ffffff", size=12),
        bgcolor="rgba(45, 45, 45, 0.8)",
        bordercolor="#ffffff",
        borderwidth=1
    )

    # Update layout with dark theme
    fig.update_layout(
        title={
            "text": "CPI Analysis: Actual vs Simulated Data",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 18, "color": "#ffffff"},
        },
        xaxis_title="Date",
        yaxis_title="CPI (Index)",
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


# def create_unemployment_plot(actual_unem, sim_unem, confidence_level=95):
#     """Create unemployment plot with actual data, median simulation, and confidence intervals"""

#     fig = go.Figure()

#     # Calculate statistics from simulations
#     median_sim = sim_unem.median(axis=1)

#     # Calculate confidence intervals
#     lower_percentile = (100 - confidence_level) / 2
#     upper_percentile = 100 - lower_percentile

#     lower_ci = sim_unem.quantile(lower_percentile / 100, axis=1)
#     upper_ci = sim_unem.quantile(upper_percentile / 100, axis=1)

#     # Add confidence interval as filled area
#     fig.add_trace(
#         go.Scatter(
#             x=list(median_sim.index) + list(median_sim.index[::-1]),
#             y=list(upper_ci.values) + list(lower_ci.values[::-1]),
#             fill="toself",
#             fillcolor="rgba(35, 105, 189, 0.2)",  # Semi-transparent blue
#             line=dict(color="rgba(255,255,255,0)"),
#             name=f"{confidence_level}% Confidence Interval",
#             hoverinfo="skip",
#         )
#     )

#     # Add median simulation line (dashed)
#     fig.add_trace(
#         go.Scatter(
#             x=median_sim.index,
#             y=median_sim.values * 100,  # Convert to percentage
#             mode="lines",
#             name="Median Simulation",
#             line=dict(color=colors[0], width=3, dash="dash"),  # Primary blue
#             hovertemplate="<b>Median Simulation</b><br>"
#             + "Date: %{x}<br>"
#             + "Unemployment: %{y:.2f}%<br>"
#             + "<extra></extra>",
#         )
#     )

#     # Add actual unemployment line (solid)
#     fig.add_trace(
#         go.Scatter(
#             x=actual_unem.index,
#             y=actual_unem["Unemployment"] * 100,  # Convert to percentage
#             mode="lines",
#             name="Actual Unemployment",
#             line=dict(color=colors[1], width=4),  # Coral red
#             hovertemplate="<b>Actual Unemployment</b><br>"
#             + "Date: %{x}<br>"
#             + "Unemployment: %{y:.2f}%<br>"
#             + "<extra></extra>",
#         )
#     )

#     # Update layout with dark theme
#     fig.update_layout(
#         title={
#             "text": "Unemployment Analysis: Actual vs Simulated Data",
#             "x": 0.5,
#             "xanchor": "center",
#             "font": {"size": 18, "color": "#ffffff"},
#         },
#         xaxis_title="Date",
#         yaxis_title="Unemployment Rate (%)",
#         hovermode="x unified",
#         legend=dict(
#             yanchor="top",
#             y=0.99,
#             xanchor="left",
#             x=0.01,
#             bgcolor="rgba(45, 45, 45, 0.9)",
#             bordercolor="#ffffff",
#             borderwidth=1,
#             font=dict(color="#ffffff"),
#         ),
#         plot_bgcolor="#1e1e1e",
#         paper_bgcolor="#1e1e1e",
#         font=dict(color="#ffffff"),
#         xaxis=dict(gridcolor="#404040", zerolinecolor="#404040", color="#ffffff"),
#         yaxis=dict(gridcolor="#404040", zerolinecolor="#404040", color="#ffffff"),
#         height=600,
#     )

#     return fig


def main():
    """Main dashboard function"""

    # Title and description
    st.title("📈 Economic Analysis Dashboard")
    st.markdown(
        """
    This dashboard visualizes economic indicators alongside simulation results:
    - **Solid lines**: Actual data
    - **Dashed lines**: Median of all simulations  
    - **Shaded areas**: Confidence intervals from simulations
    """
    )

    # Load data
    try:
        actual_gdp, nominal_gdp, forecast_cutoff_date = load_gdp_data()
        actual_cpi, sim_cpi, _ = load_cpi_data()  # CPI has same cutoff date
        # actual_unem, sim_unem = load_unemployment_data()

        # Sidebar controls
        st.sidebar.header("📊 Chart Controls")

        # Confidence interval controls
        gdp_confidence = st.sidebar.slider(
            "GDP Confidence Interval %",
            min_value=50,
            max_value=95,
            value=95,
            step=5,
            help="Select the confidence interval percentage for GDP simulation band",
        )

        cpi_confidence = st.sidebar.slider(
            "CPI Confidence Interval %",
            min_value=50,
            max_value=95,
            value=95,
            step=5,
            help="Select the confidence interval percentage for CPI simulation band",
        )

        # unemployment_confidence = st.sidebar.slider(
        #     "Unemployment Confidence Interval %",
        #     min_value=50,
        #     max_value=95,
        #     value=95,
        #     step=5,
        #     help="Select the confidence interval percentage for unemployment simulation band",
        # )
        
        # Hardcoded scenario adjustments
        gdp_scenario_adjustment = 1.0  # 1% quarterly growth for GDP
        cpi_scenario_adjustment = 0.1  # 0.1% quarterly decline for CPI

        # Create layout with equal column ratio
        col1, col2 = st.columns([1, 1])

        # Left column - GDP plot
        with col1:
            gdp_fig = create_gdp_plot(actual_gdp, nominal_gdp, forecast_cutoff_date, gdp_confidence, gdp_scenario_adjustment)
            st.plotly_chart(gdp_fig, use_container_width=True)

        # Right column - CPI plot
        with col2:
            # CPI plot
            cpi_fig = create_cpi_plot(actual_cpi, sim_cpi, forecast_cutoff_date, cpi_confidence, cpi_scenario_adjustment)
            st.plotly_chart(cpi_fig, use_container_width=True)

            # # Unemployment plot (bottom)
            # unem_fig = create_unemployment_plot(
            #     actual_unem, sim_unem, unemployment_confidence
            # )
            # st.plotly_chart(unem_fig, use_container_width=True)

        # Data summary metrics
        st.markdown("---")
        st.markdown("### 📊 Latest Values")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Latest Actual GDP",
                f"${actual_gdp['GDP'].iloc[-1]/1e12:.2f}T",
                delta=f"{((actual_gdp['GDP'].iloc[-1] - actual_gdp['GDP'].iloc[-2])/actual_gdp['GDP'].iloc[-2]*100):.1f}%",
            )
            latest_median_gdp = nominal_gdp.median(axis=1).iloc[-1]
            st.metric("Latest Median GDP Simulation", f"${latest_median_gdp/1e12:.2f}T")

        with col2:
            st.metric(
                "Latest Actual CPI",
                f"{actual_cpi['CPI'].iloc[-1]:.4f}",
                delta=f"{((actual_cpi['CPI'].iloc[-1] - actual_cpi['CPI'].iloc[-2])/actual_cpi['CPI'].iloc[-2]*100):.2f}%",
            )
            latest_median_cpi = sim_cpi.median(axis=1).iloc[-1]
            st.metric("Latest Median CPI Simulation", f"{latest_median_cpi:.4f}")

        # with col3:
        #     st.metric(
        #         "Latest Actual Unemployment",
        #         f"{actual_unem['Unemployment'].iloc[-1]*100:.2f}%",
        #         delta=f"{((actual_unem['Unemployment'].iloc[-1] - actual_unem['Unemployment'].iloc[-2])*100):.2f}pp",
        #     )
        #     latest_median_unem = sim_unem.median(axis=1).iloc[-1]
        #     st.metric(
        #         "Latest Median Unemployment Simulation",
        #         f"{latest_median_unem*100:.2f}%",
        #     )

        # Data period information
        st.markdown("---")
        st.markdown("### 📅 Data Coverage")

        col1, col2 = st.columns(2)

        with col1:
            start_quarter = (actual_gdp.index[0].month - 1) // 3 + 1
            end_quarter = (actual_gdp.index[-1].month - 1) // 3 + 1
            st.info(
                f"""
            **GDP Data**
            - Period: {actual_gdp.index[0].year} Q{start_quarter} to {actual_gdp.index[-1].year} Q{end_quarter}
            - Observations: {len(actual_gdp)}
            - Simulations: {nominal_gdp.shape[1]}
            """
            )

        with col2:
            cpi_start_quarter = (actual_cpi.index[0].month - 1) // 3 + 1
            cpi_end_quarter = (actual_cpi.index[-1].month - 1) // 3 + 1
            st.info(
                f"""
            **CPI Data**
            - Period: {actual_cpi.index[0].year} Q{cpi_start_quarter} to {actual_cpi.index[-1].year} Q{cpi_end_quarter}
            - Observations: {len(actual_cpi)}
            - Simulations: {sim_cpi.shape[1]}
            """
            )

        # with col3:
        #     unem_start_quarter = (actual_unem.index[0].month - 1) // 3 + 1
        #     unem_end_quarter = (actual_unem.index[-1].month - 1) // 3 + 1
        #     st.info(
        #         f"""
        #     **Unemployment Data**
        #     - Period: {actual_unem.index[0].year} Q{unem_start_quarter} to {actual_unem.index[-1].year} Q{unem_end_quarter}
        #     - Observations: {len(actual_unem)}
        #     - Simulations: {sim_unem.shape[1]}
        #     """
        #     )

    except FileNotFoundError as e:
        st.error(
            f"""
        📁 **Data files not found!**
        
        Please ensure the following files exist in the `data/` directory:
        - `actualGDP.csv` and `test_gdp.csv`
        - `actualCPI.csv` and `model_outputtest.csv`
        # - `actualUnem.csv` and `unemployment.csv`
        
        Error details: {str(e)}
        """
        )

    except Exception as e:
        st.error(f"❌ An error occurred while loading the data: {str(e)}")


if __name__ == "__main__":
    main()
