#!/usr/bin/env python3
"""
Simple comparison script showing the visual improvements.
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add the src directory to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))

from etc.visualization import plot_phase_diagram


def create_simple_data():
    """Create simple phase diagram data for comparison."""
    result = {}
    k_values = [2, 3, 4, 5, 6]
    scale_values = [1, 2, 3, 4, 5, 6, 7, 8]
    
    for k in k_values:
        result[k] = {}
        for scale in scale_values:
            mu_gamma_ratio = scale * 0.15 + k * 0.05
            hmin_value = np.sin(k * 0.8) * np.cos(scale * 0.4) * 2 + (k - 3) * (scale - 4) * 0.05
            result[k][scale] = (mu_gamma_ratio, hmin_value)
    
    return result


def original_style_plot(result):
    """Recreate the original plotting style for comparison."""
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
    
    plt.figure(figsize=(10, 6))
    plt.imshow(hmin_values, aspect='auto', origin='lower',
               extent=[mu_gamma_ratios.min(), mu_gamma_ratios.max(),
                       min(k_values), max(k_values)], cmap='viridis')
    plt.colorbar(label='H_min')
    plt.xlabel(r'$\mu / \gamma$')  # Fixed LaTeX warning
    plt.ylabel(r'$k$')  # Fixed LaTeX warning
    plt.title('Original Style Phase Diagram')
    
    return plt.gcf()


def main():
    """Show side-by-side comparison."""
    data = create_simple_data()
    
    # Create comparison figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Original style plot
    plt.sca(ax1)
    k_values = sorted(list(data.keys()))
    scale_values = sorted(list(data[k_values[0]].keys()))
    
    mu_gamma_ratios = []
    hmin_values = []
    for k in k_values:
        for scale in scale_values:
            ratio, hmin = data[k][scale]
            mu_gamma_ratios.append(ratio)
            hmin_values.append(hmin)
    
    mu_gamma_ratios = np.array(mu_gamma_ratios).reshape(len(k_values), len(scale_values))
    hmin_values = np.array(hmin_values).reshape(len(k_values), len(scale_values))
    
    im1 = ax1.imshow(hmin_values, aspect='auto', origin='lower',
                     extent=[mu_gamma_ratios.min(), mu_gamma_ratios.max(),
                             min(k_values), max(k_values)], cmap='viridis')
    plt.colorbar(im1, ax=ax1, label='H_min')
    ax1.set_xlabel(r'$\mu / \gamma$')
    ax1.set_ylabel(r'$k$')
    ax1.set_title('Original Style', fontsize=14, fontweight='bold')
    
    # Enhanced plot
    plt.sca(ax2)
    x_min, x_max = mu_gamma_ratios.min(), mu_gamma_ratios.max()
    y_min, y_max = min(k_values), max(k_values)
    
    im2 = ax2.imshow(hmin_values, aspect='auto', origin='lower',
                     extent=[x_min, x_max, y_min, y_max], 
                     cmap='RdYlBu_r', interpolation='bilinear')
    
    # Add contours
    x_coords = np.linspace(x_min, x_max, mu_gamma_ratios.shape[1])
    y_coords = np.linspace(y_min, y_max, mu_gamma_ratios.shape[0])
    X, Y = np.meshgrid(x_coords, y_coords)
    contours = ax2.contour(X, Y, hmin_values, levels=8, colors='black', alpha=0.4, linewidths=0.5)
    ax2.clabel(contours, inline=True, fontsize=8, fmt='%.1f')
    
    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
    cbar2.set_label(r'$H_{\min}$', rotation=270, labelpad=15, fontsize=12)
    ax2.set_xlabel(r'$\mu / \gamma$', fontsize=12)
    ax2.set_ylabel(r'$k$', fontsize=12)
    ax2.set_title('Enhanced Style', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    plt.suptitle('Phase Diagram Visualization Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/tmp/phase_diagram_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Comparison saved to: /tmp/phase_diagram_comparison.png")
    print("\\nKey Improvements Made:")
    print("✓ Better colormap (RdYlBu_r vs viridis)")
    print("✓ Smooth interpolation for better visual quality")  
    print("✓ Contour lines to highlight phase boundaries")
    print("✓ Enhanced grid and typography")
    print("✓ Better colorbar formatting and positioning")
    print("✓ Fixed LaTeX string warnings")
    print("✓ Professional layout and styling")


if __name__ == "__main__":
    main()