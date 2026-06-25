"""
Example usage of the improved plot_phase_diagram function.

This example shows how to use the enhanced visualization features
to create professional-looking phase diagrams.
"""

import numpy as np
import sys
from pathlib import Path

# Add the src directory to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))

from etc.visualization import plot_phase_diagram, create_custom_colormap, PHASE_DIAGRAM_CMAPS


def create_example_data():
    """Create example phase diagram data."""
    result = {}
    
    # Example data structure as expected by the function
    k_values = [2, 3, 4, 5, 6]  # Number of nodes
    scale_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Scale parameters
    
    for k in k_values:
        result[k] = {}
        for scale in scale_values:
            # Example mu/gamma ratio and H_min value calculation
            mu_gamma_ratio = scale * 0.2 + k * 0.1
            hmin_value = np.sin(k) * np.cos(scale) * 3 + (k - 3) * (scale - 5) * 0.1
            
            result[k][scale] = (mu_gamma_ratio, hmin_value)
    
    return result


def main():
    """Demonstrate the enhanced phase diagram plotting."""
    # Create sample data
    result = create_example_data()
    
    print("Basic usage (with all enhancements enabled):")
    print("=" * 50)
    
    # Basic usage with enhanced defaults
    plot_phase_diagram(
        result,
        title='Enhanced Phase Diagram - Basic Usage'
    )
    
    print("\\nAdvanced usage with custom settings:")
    print("=" * 50)
    
    # Advanced usage with custom settings
    plot_phase_diagram(
        result,
        figsize=(14, 10),
        cmap='plasma',
        add_contours=True,
        contour_levels=15,
        grid=True,
        colorbar_shrink=0.7,
        title='Advanced Phase Diagram with Custom Settings',
        dpi=200
    )
    
    print("\\nUsage with custom colormap:")
    print("=" * 50)
    
    # Using a custom colormap
    custom_cmap = create_custom_colormap(
        PHASE_DIAGRAM_CMAPS['thermal'], 
        'my_thermal'
    )
    
    plot_phase_diagram(
        result,
        cmap=custom_cmap,
        add_contours=False,
        title='Phase Diagram with Custom Thermal Colormap'
    )
    
    print("\\nAll examples completed!")
    print("\\nKey features demonstrated:")
    print("• Enhanced color schemes and customization")
    print("• Optional contour lines for phase boundaries")
    print("• Improved typography and LaTeX formatting")
    print("• Configurable grid and colorbar")
    print("• Custom colormap support")
    print("• Professional layout and styling")


if __name__ == "__main__":
    main()