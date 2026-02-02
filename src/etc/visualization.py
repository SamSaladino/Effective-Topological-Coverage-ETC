"""
Enhanced visualization functions for ETC phase diagrams and other plots.

This module provides improved plotting functions with better aesthetics,
proper LaTeX string handling, and more customization options.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from typing import Dict, Optional, Tuple, Union


def plot_phase_diagram(
    result: Dict, 
    figsize: Tuple[int, int] = (12, 8),
    cmap: Union[str, LinearSegmentedColormap] = 'RdYlBu_r',
    add_contours: bool = True,
    contour_levels: Optional[int] = 10,
    grid: bool = True,
    colorbar_shrink: float = 0.8,
    title: str = 'Phase Diagram',
    save_path: Optional[str] = None,
    dpi: int = 150
) -> None:
    """
    Plot an enhanced phase diagram with improved visualization features.
    
    Parameters
    ----------
    result : dict
        Dictionary containing phase diagram data with structure:
        {k_value: {scale_value: (mu_gamma_ratio, hmin_value), ...}, ...}
    figsize : tuple, optional
        Figure size as (width, height). Default is (12, 8).
    cmap : str or LinearSegmentedColormap, optional
        Colormap name or colormap object. Default is 'RdYlBu_r' for better phase distinction.
    add_contours : bool, optional
        Whether to add contour lines. Default is True.
    contour_levels : int, optional
        Number of contour levels. Default is 10.
    grid : bool, optional
        Whether to add a grid. Default is True.
    colorbar_shrink : float, optional
        Shrink factor for colorbar. Default is 0.8.
    title : str, optional
        Plot title. Default is 'Phase Diagram'.
    save_path : str, optional
        Path to save the figure. If None, figure is not saved.
    dpi : int, optional
        DPI for saving figure. Default is 150.
        
    Raises
    ------
    ValueError
        If result dictionary is empty or has invalid structure.
    """
    if not result:
        raise ValueError("The result dictionary is empty.")

    k_values = list(result.keys())
    if not all(isinstance(k, (int, float)) for k in k_values):
        raise ValueError("Keys of the result dictionary must be numeric (e.g., int or float).")

    scale_values = list(result[k_values[0]].keys())
    if not all(isinstance(scale, (int, float)) for scale in scale_values):
        raise ValueError("Scale values in the result dictionary must be numeric (e.g., int or float).")

    # Extract and reshape data
    mu_gamma_ratios = []
    hmin_values = []
    
    # Sort keys for consistent ordering
    k_values_sorted = sorted(k_values)
    scale_values_sorted = sorted(scale_values)
    
    for k in k_values_sorted:
        for scale in scale_values_sorted:
            ratio, hmin = result[k][scale]
            mu_gamma_ratios.append(ratio)
            hmin_values.append(hmin)

    # Reshape data for heatmap
    try:
        mu_gamma_ratios = np.array(mu_gamma_ratios).reshape(len(k_values_sorted), len(scale_values_sorted))
        hmin_values = np.array(hmin_values).reshape(len(k_values_sorted), len(scale_values_sorted))
    except ValueError as e:
        raise ValueError("Mismatch in the dimensions of the result data. Ensure the data is consistent.") from e

    # Create the plot with improved styling
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate proper extent for the axes
    # For x-axis, we use the actual mu/gamma ratio range
    x_min, x_max = mu_gamma_ratios.min(), mu_gamma_ratios.max()
    # For y-axis, we use the k values range
    y_min, y_max = min(k_values_sorted), max(k_values_sorted)
    
    # Create the heatmap
    im = ax.imshow(
        hmin_values, 
        aspect='auto', 
        origin='lower',
        extent=[x_min, x_max, y_min, y_max], 
        cmap=cmap,
        interpolation='bilinear'  # Smooth interpolation for better appearance
    )
    
    # Add contour lines if requested
    if add_contours and contour_levels is not None:
        # Create coordinate grids for contour plotting
        x_coords = np.linspace(x_min, x_max, mu_gamma_ratios.shape[1])
        y_coords = np.linspace(y_min, y_max, mu_gamma_ratios.shape[0])
        X, Y = np.meshgrid(x_coords, y_coords)
        
        # Add contour lines
        contours = ax.contour(
            X, Y, hmin_values, 
            levels=contour_levels, 
            colors='black', 
            alpha=0.4, 
            linewidths=0.5
        )
        
        # Add contour labels for key levels
        ax.clabel(contours, inline=True, fontsize=8, fmt='%.1f')
    
    # Enhanced colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=colorbar_shrink)
    cbar.set_label(r'$H_{\min}$', rotation=270, labelpad=15, fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    
    # Improved axis labels with proper LaTeX formatting
    ax.set_xlabel(r'$\mu / \gamma$', fontsize=14)
    ax.set_ylabel(r'$k$', fontsize=14)
    ax.set_title(title, fontsize=16, pad=20)
    
    # Add grid if requested
    if grid:
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Improve tick formatting
    ax.tick_params(labelsize=11)
    
    # Add a subtle border
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
    
    # Tight layout for better spacing
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
    
    plt.show()


def plot_phase_diagram_with_annotations(
    result: Dict,
    phase_boundaries: Optional[Dict] = None,
    figsize: Tuple[int, int] = (12, 8),
    **kwargs
) -> None:
    """
    Plot phase diagram with optional phase boundary annotations.
    
    Parameters
    ----------
    result : dict
        Phase diagram data dictionary.
    phase_boundaries : dict, optional
        Dictionary defining phase boundaries for annotation.
    figsize : tuple, optional
        Figure size. Default is (12, 8).
    **kwargs
        Additional arguments passed to plot_phase_diagram.
    """
    # Plot the base phase diagram
    plot_phase_diagram(result, figsize=figsize, **kwargs)
    
    # Add phase boundary annotations if provided
    if phase_boundaries:
        ax = plt.gca()
        
        for boundary_name, boundary_data in phase_boundaries.items():
            if 'line' in boundary_data:
                x_line, y_line = boundary_data['line']
                ax.plot(x_line, y_line, 
                       color=boundary_data.get('color', 'white'),
                       linewidth=boundary_data.get('linewidth', 2),
                       linestyle=boundary_data.get('linestyle', '-'),
                       label=boundary_name)
        
        # Add legend if we have boundaries
        ax.legend(loc='best', framealpha=0.9)


def create_custom_colormap(colors: list, name: str = 'custom') -> LinearSegmentedColormap:
    """
    Create a custom colormap for phase diagrams.
    
    Parameters
    ----------
    colors : list
        List of colors for the colormap.
    name : str, optional
        Name for the custom colormap.
        
    Returns
    -------
    LinearSegmentedColormap
        Custom colormap object.
    """
    return LinearSegmentedColormap.from_list(name, colors)


# Predefined color schemes for different types of phase diagrams
PHASE_DIAGRAM_CMAPS = {
    'thermal': ['#000428', '#004e92', '#009ffd', '#00d2ff', '#ffffff'],
    'magnetic': ['#8B0000', '#FF4500', '#FFD700', '#FFFF00', '#FFFFFF'],
    'topological': ['#2E003E', '#512DA8', '#3F51B5', '#2196F3', '#00BCD4', '#4CAF50'],
}