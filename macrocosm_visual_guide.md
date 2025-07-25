# Macrocosm Visual Guide

This guide explains how to use the `macrocosm_visual` package to apply your company's visual branding to matplotlib and plotly visualizations.

## Package Overview

The `macrocosm_visual` package provides:
- Pre-configured styling for matplotlib and plotly
- Consistent color palette across all visualizations
- Easy setup functions for both plotting libraries
- Company branding compliance

## Installation

The package is already installed in your virtual environment. To use it:

```bash
source venv/bin/activate
```

## Quick Start

### For Matplotlib

```python
from macrocosm_visual.viz_setup import setup_matplotlib
import matplotlib.pyplot as plt

# Apply Macrocosm styling
setup_matplotlib()

# Now create plots with company styling
fig, ax = plt.subplots()
ax.plot([1, 2, 3, 4], [1, 4, 2, 3])
plt.show()
```

### For Plotly

```python
from macrocosm_visual.viz_setup import setup_plotly
import plotly.graph_objects as go

# Apply Macrocosm styling
setup_plotly()

# Now create plots with company styling
fig = go.Figure(data=go.Scatter(x=[1, 2, 3, 4], y=[1, 4, 2, 3]))
fig.show()
```

### Using Both Libraries

```python
from macrocosm_visual.viz_setup import VizSetup

# Setup both libraries at once
viz = VizSetup()
viz.setup_matplotlib()
viz.setup_plotly()
```

## Color Palette

The Macrocosm color palette includes:

1. **Primary Blue**: `#153995`
2. **Coral Red**: `#fa6060` 
3. **Teal Green**: `#199079`
4. **Golden Yellow**: `#f4d35e`
5. **Sky Blue**: `#83bdcf`
6. **Light Pink**: `#ffe8eb`
7. **Mint Green**: `#83f0ca`
8. **Bright Yellow**: `#f5f749`

### Accessing Colors Programmatically

```python
from macrocosm_visual.viz_setup import get_macrocosm_colors

colors = get_macrocosm_colors()
print(colors)  # Returns list of hex color codes
```

## Visual Style Features

### Matplotlib Styling
- **Font**: Monospace family, size 14
- **Figure size**: 16x9 inches (300 DPI)
- **Colors**: Dark gray (`#2D2D2D`) for text and axes
- **Spines**: Only left and bottom axes visible
- **Line width**: 2px default
- **Title**: Bold, size 16

### Plotly Styling
- **Font**: Monospace family, size 14
- **Colors**: Same dark gray theme
- **Grid**: Visible on both axes
- **Line width**: 3px for scatter plots
- **Consistent colorway**: Matches matplotlib cycle

## Advanced Usage

### Custom Color Rotation

```python
from macrocosm_visual.viz_setup import VizSetup

viz = VizSetup()
viz.setup_matplotlib()
viz.setup_plotly()

# Use custom colors for specific visualizations
custom_colors = ['#153995', '#fa6060', '#199079']
viz.to_custom_rotation(custom_colors)
```

### Using Custom Config Files

```python
from macrocosm_visual.viz_setup import setup_matplotlib, setup_plotly

# Use your own YAML config file
setup_matplotlib(configs='path/to/your/config.yaml')
setup_plotly(configs='path/to/your/config.yaml')
```

## Best Practices

1. **Always setup styling first** - Call setup functions before creating any plots
2. **Consistent usage** - Use the same setup across all visualizations in your project
3. **Color accessibility** - The provided palette is designed for good contrast and readability
4. **High resolution** - Default DPI is set to 300 for crisp output

## Example Complete Script

```python
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from macrocosm_visual.viz_setup import VizSetup, get_macrocosm_colors

# Setup both libraries
viz = VizSetup()
viz.setup_matplotlib()
viz.setup_plotly()

# Get company colors
colors = get_macrocosm_colors()

# Create matplotlib plot
fig, ax = plt.subplots()
ax.plot([1, 2, 3, 4], [1, 4, 2, 3], label='Data Series 1')
ax.plot([1, 2, 3, 4], [2, 3, 1, 4], label='Data Series 2')
ax.set_title('Macrocosm Styled Plot')
ax.legend()
plt.savefig('matplotlib_example.png')

# Create plotly plot
fig = go.Figure()
fig.add_trace(go.Scatter(x=[1, 2, 3, 4], y=[1, 4, 2, 3], name='Data Series 1'))
fig.add_trace(go.Scatter(x=[1, 2, 3, 4], y=[2, 3, 1, 4], name='Data Series 2'))
fig.update_layout(title='Macrocosm Styled Plot')
fig.write_html('plotly_example.html')
```

This ensures all your visualizations maintain the Macrocosm brand identity and professional appearance.