#!/usr/bin/env python3
"""
Test script to debug dashboard data loading and plotting
"""

import sys
sys.path.append('.')

from gdp_plot import load_data, create_gdp_plot

# Test the data loading and plot creation
print("Testing data loading...")
actual_gdp, nominal_gdp = load_data()

print("\nTesting plot creation...")
fig = create_gdp_plot(actual_gdp, nominal_gdp, confidence_level=95)

print("\nTest completed!")