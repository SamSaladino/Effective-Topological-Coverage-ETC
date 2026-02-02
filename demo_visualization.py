#!/usr/bin/env python3
"""
Demo script to showcase the enhanced phase diagram visualization.

This script creates sample phase diagram data and demonstrates both
the original and improved plotting functions side by side.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

# Add the src directory to path for imports
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from etc.visualization import plot_phase_diagram, create_custom_colormap, PHASE_DIAGRAM_CMAPS


def create_realistic_phase_data():
    """Create more realistic phase diagram data with interesting features."""
    result = {}
    
    # Create a larger grid for more interesting visualization
    k_values = range(2, 12)  # k from 2 to 11
    scale_values = range(1, 21)  # scale from 1 to 20
    
    for k in k_values:
        result[k] = {}
        for scale in scale_values:
            # Create realistic mu/gamma ratio
            mu_gamma_ratio = scale * 0.1 + k * 0.05
            
            # Create interesting hmin pattern with phase transitions
            hmin_value = (
                np.sin(k * 0.5) * np.cos(scale * 0.3) * 5 +
                (k - 6) * (scale - 10) * 0.01 +
                np.random.normal(0, 0.5)  # Add some noise
            )
            
            result[k][scale] = (mu_gamma_ratio, hmin_value)
    
    return result


def original_plot_phase_diagram(result):
    """Original plotting function from the problem statement."""
    if not result:
        raise ValueError("The result dictionary is empty.")

    k_values = list(result.keys())
    if not all(isinstance(k, (int, float)) for k in k_values):
        raise ValueError("Keys of the result dictionary must be numeric (e.g., int or float).")

    scale_values = list(result[k_values[0]].keys())
    if not all(isinstance(scale, (int, float)) for scale in scale_values):
        raise ValueError("Scale values in the result dictionary must be numeric (e.g., int or float).")

    mu_gamma_ratios = []
    hmin_values = []
    for k in k_values:
        for scale in scale_values:
            ratio, hmin = result[k][scale]
            mu_gamma_ratios.append(ratio)
            hmin_values.append(hmin)

    # Reshape data for heatmap
    try:
        mu_gamma_ratios = np.array(mu_gamma_ratios).reshape(len(k_values), len(scale_values))
        hmin_values = np.array(hmin_values).reshape(len(k_values), len(scale_values))
    except ValueError as e:
        raise ValueError("Mismatch in the dimensions of the result data. Ensure the data is consistent.") from e

    # Plot the phase diagram
    plt.figure(figsize=(10, 6))
    plt.imshow(hmin_values, aspect='auto', origin='lower',
               extent=[mu_gamma_ratios.min(), mu_gamma_ratios.max(),
                       min(k_values), max(k_values)], cmap='viridis')
    plt.colorbar(label='H_min')
    plt.xlabel(r'$\mu / \gamma$')  # This could cause SyntaxWarning
    plt.ylabel(r'$k$')  # This could cause SyntaxWarning
    plt.title('Phase Diagram')
    plt.show()


def demo_visualizations():
    """Demonstrate the visualization improvements."""
    # Create sample data
    print("Creating sample phase diagram data...")
    result = create_realistic_phase_data()
    
    # Create output directory
    output_dir = Path('/tmp/phase_diagram_demo')
    output_dir.mkdir(exist_ok=True)
    
    print(f"Generated data for {len(result)} k values and {len(result[2])} scale values")
    
    # 1. Original visualization (with fixes for LaTeX warnings)
    print("\n1. Creating original-style visualization...")
    plt.figure(figsize=(10, 6))
    
    k_values = sorted(list(result.keys()))
    scale_values = sorted(list(result[k_values[0]].keys()))
    
    mu_gamma_ratios = []
    hmin_values = []
    for k in k_values:
        for scale in scale_values:
            ratio, hmin = result[k][scale]
            mu_gamma_ratios.append(ratio)
            hmin_values.append(hmin)
    
    mu_gamma_ratios = np.array(mu_gamma_ratios).reshape(len(k_values), len(scale_values))
    hmin_values = np.array(hmin_values).reshape(len(k_values), len(scale_values))
    
    plt.imshow(hmin_values, aspect='auto', origin='lower',
               extent=[mu_gamma_ratios.min(), mu_gamma_ratios.max(),
                       min(k_values), max(k_values)], cmap='viridis')
    plt.colorbar(label='H_min')
    plt.xlabel(r'$\mu / \gamma$')  # Fixed LaTeX string
    plt.ylabel(r'$k$')  # Fixed LaTeX string
    plt.title('Original Phase Diagram (Fixed)')
    plt.savefig(output_dir / 'original_phase_diagram.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Enhanced visualization with default settings
    print("2. Creating enhanced visualization with default settings...")
    plot_phase_diagram(
        result, 
        title='Enhanced Phase Diagram (Default)',
        save_path=output_dir / 'enhanced_default.png'
    )
    plt.close()
    
    # 3. Enhanced visualization with custom colormap and contours
    print("3. Creating enhanced visualization with custom colormap...")
    plot_phase_diagram(
        result,
        cmap='plasma',
        add_contours=True,
        contour_levels=15,
        title='Enhanced Phase Diagram (Plasma + Contours)',
        save_path=output_dir / 'enhanced_plasma_contours.png'
    )
    plt.close()
    
    # 4. Enhanced visualization with custom thermal colormap
    print("4. Creating visualization with custom thermal colormap...")
    thermal_cmap = create_custom_colormap(PHASE_DIAGRAM_CMAPS['thermal'], 'thermal')
    
    plot_phase_diagram(
        result,
        cmap=thermal_cmap,  # Pass the colormap object directly
        add_contours=True,
        contour_levels=12,
        grid=True,
        title='Enhanced Phase Diagram (Custom Thermal)',
        save_path=output_dir / 'enhanced_thermal.png'
    )
    plt.close()
    
    # 5. Enhanced visualization with topological colormap
    print("5. Creating visualization with topological colormap...")
    topo_cmap = create_custom_colormap(PHASE_DIAGRAM_CMAPS['topological'], 'topological')
    
    plot_phase_diagram(
        result,
        cmap=topo_cmap,  # Pass the colormap object directly
        figsize=(14, 10),
        add_contours=False,
        grid=True,
        colorbar_shrink=0.8,
        title='Enhanced Phase Diagram (Topological Theme)',
        save_path=output_dir / 'enhanced_topological.png'
    )
    plt.close()
    
    print(f"\nAll visualizations saved to: {output_dir}")
    print("Files created:")
    for file_path in sorted(output_dir.glob('*.png')):
        print(f"  - {file_path.name}")
    
    print("\nKey improvements made:")
    print("✓ Fixed LaTeX string warnings using raw strings")
    print("✓ Improved color schemes for better phase distinction")
    print("✓ Added optional contour lines for phase boundaries")
    print("✓ Enhanced figure layout and typography")
    print("✓ Added customizable grid and colorbar formatting")
    print("✓ Improved axis scaling and labels")
    print("✓ Made function highly configurable")
    print("✓ Added proper error handling and validation")
    print("✓ Smooth interpolation for better visual quality")


if __name__ == "__main__":
    # Set random seed for reproducible results
    np.random.seed(42)
    
    print("Phase Diagram Visualization Demo")
    print("=" * 40)
    
    demo_visualizations()
    
    print("\nDemo completed successfully!")