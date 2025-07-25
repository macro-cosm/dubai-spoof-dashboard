# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an **Economic Analysis Dashboard** built with Python and Streamlit for visualizing economic indicators (GDP, CPI, unemployment) alongside simulation results. The project uses the `macrocosm_visual` package for company-branded visualizations and supports scenario modeling with confidence intervals.

## Development Commands

### Environment Setup
```bash
# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Application
```bash
# Main comprehensive dashboard (recommended)
streamlit run dashboard_full.py

# GDP-only simplified dashboard
streamlit run gdp_plot.py

# Test/debug dashboard functionality
python test_dashboard.py
```

### Data Management
```bash
# Download fresh market data (VIX, S&P 500)
python download_data.py

# Generate synthetic GDP simulation data
python gen_gdp_data.py
```

## Architecture Overview

### Core Components

**Main Dashboard (`dashboard_full.py`)**
- Primary entry point with GDP, CPI analysis and scenario modeling
- Features timeline shifting (2014 data → 2024 for forecasting)
- Supports business-as-usual vs. alternative scenario comparisons
- Dark theme with company branding via `macrocosm_visual`

**Data Pipeline**
- CSV files with quarterly economic data ("YYYY QN" format)
- Custom date parsing with 10-year timeline shift capability
- Monte Carlo simulation approach with configurable confidence intervals
- Statistical aggregation (median, quantiles) across simulation samples

**Visualization System**
- Plotly interactive charts with custom Macrocosm color palette
- Confidence intervals as shaded areas, actual data as solid lines, simulations as dashed
- Hover tooltips with compact currency formatting ($15.2T instead of full numbers)
- Forecast indicators with vertical lines and annotations

### Data Structure

**Input Format**: CSV files with samples as rows, time periods as columns
```
Index: Sample1, Sample2, ...
Columns: 2014 Q1, 2014 Q2, ...
Values: Economic indicators (GDP in USD, CPI index, unemployment rates)
```

**Processing Flow**: Load → Transpose → Parse Dates → Filter Time Range → Calculate Statistics → Visualize

### Key Data Files
- `data/actualGDP.csv`, `data/actualCPI.csv` - Historical economic data
- `data/test_gdp.csv`, `data/model_outputtest.csv` - Simulation outputs
- `data/sp500_data.csv`, `data/vix_data.csv` - Market data from Yahoo Finance

## Working with Economic Data

### Time Series Handling
- Quarterly data converted from "YYYY QN" string format to pandas Timestamps
- Timeline shifting: `shifted_year = year + 10` (e.g., 2014 → 2024)
- Consistent cutoff dates: real data stops at 2025 Q2, forecasts extend to 2029

### Scenario Modeling
The `create_scenario_data()` function applies exponential adjustments after forecast cutoff:
- GDP scenarios: `multiplier = (1 + adjustment_pct/100) ** T`
- CPI scenarios: `multiplier = (1 - adjustment_pct/100) ** T` (inverted)
- Where T = quarters after cutoff date

### Confidence Intervals
- Monte Carlo simulations with 50-95% configurable confidence levels
- Statistical calculation: `lower_ci = data.quantile(lower_percentile/100, axis=1)`
- Visualization as filled areas between upper/lower bounds

## Macrocosm Visual Branding

### Setup Pattern
```python
from macrocosm_visual.viz_setup import setup_plotly, get_macrocosm_colors
setup_plotly()
colors = get_macrocosm_colors()
```

### Color Palette
- Primary Blue: `#153995`, Coral Red: `#fa6060`, Teal Green: `#199079`
- Golden Yellow: `#f4d35e`, Sky Blue: `#83bdcf`, Light Pink: `#ffe8eb`
- Mint Green: `#83f0ca`, Bright Yellow: `#f5f749`

### Dark Theme Implementation
Custom CSS in Streamlit with `#1e1e1e` background and `#ffffff` text for professional appearance.

## Adding New Economic Indicators

1. **Data Loading**: Create function following pattern in `load_gdp_data()` with proper date parsing and filtering
2. **Visualization**: Implement plotting function with confidence intervals using existing `create_gdp_plot()` as template
3. **Dashboard Integration**: Add to main layout with appropriate column sizing and controls
4. **Metrics**: Update summary statistics section with new indicator's latest values

## Common Patterns

### Currency Formatting
Use `format_currency_compact()` for readable display: `$15.2T` instead of `$15,200,000,000,000`

### Hover Templates
Include confidence interval data in customdata arrays for compact hover information

### Date Filtering
Always apply both real data cutoff (2025 Q2) and full simulation cutoff (2029 Q1) appropriately

### Error Handling
The codebase has minimal error handling - add try/catch blocks around data loading operations when extending functionality