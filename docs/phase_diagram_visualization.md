# Enhanced Phase Diagram Visualization

This document describes the improved phase diagram visualization functions provided in the `etc.visualization` module.

## Overview

The `plot_phase_diagram` function has been significantly enhanced with the following improvements:

### Key Improvements

1. **Fixed LaTeX String Issues**: All LaTeX strings now use raw string formatting (`r"$\mu / \gamma$"`) to prevent SyntaxWarning messages.

2. **Enhanced Visual Quality**: 
   - Better default colormap (`RdYlBu_r`) for improved phase distinction
   - Smooth bilinear interpolation for better visual quality
   - Professional typography and layout

3. **Contour Lines**: Optional contour lines to highlight phase boundaries and transitions.

4. **Customization Options**: 
   - Configurable figure size, colormap, and DPI
   - Adjustable colorbar positioning and sizing
   - Optional grid display
   - Custom titles and labels

5. **Custom Colormap Support**: 
   - Predefined colormaps for different physics domains
   - Support for custom colormap objects
   - Easy creation of domain-specific color schemes

6. **Better Error Handling**: Comprehensive input validation and clear error messages.

## Function Signature

```python
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
) -> None
```

## Usage Examples

### Basic Usage
```python
from etc.visualization import plot_phase_diagram

# Your phase diagram data
result = {
    2: {1: (0.3, -1.2), 2: (0.6, -0.8), ...},
    3: {1: (0.3, -0.9), 2: (0.6, -0.5), ...},
    ...
}

# Plot with enhanced defaults
plot_phase_diagram(result)
```

### Advanced Customization
```python
# Highly customized plot
plot_phase_diagram(
    result,
    figsize=(14, 10),
    cmap='plasma',
    add_contours=True,
    contour_levels=15,
    grid=True,
    colorbar_shrink=0.7,
    title='Custom Phase Diagram',
    save_path='my_phase_diagram.png',
    dpi=300
)
```

### Custom Colormap
```python
from etc.visualization import create_custom_colormap, PHASE_DIAGRAM_CMAPS

# Use predefined thermal colormap
thermal_cmap = create_custom_colormap(PHASE_DIAGRAM_CMAPS['thermal'])
plot_phase_diagram(result, cmap=thermal_cmap)

# Create completely custom colormap
custom_colors = ['#000080', '#0000FF', '#00FFFF', '#FFFF00', '#FF0000']
my_cmap = create_custom_colormap(custom_colors, 'my_custom')
plot_phase_diagram(result, cmap=my_cmap)
```

## Data Format

The `result` dictionary should have the following structure:
```python
{
    k_value: {
        scale_value: (mu_gamma_ratio, hmin_value),
        ...
    },
    ...
}
```

Where:
- `k_value`: Integer representing the number of nodes (y-axis)
- `scale_value`: Numeric scale parameter  
- `mu_gamma_ratio`: Float representing μ/γ ratio (x-axis)
- `hmin_value`: Float representing H_min value (colormap)

## Predefined Colormaps

The module provides three predefined colormaps optimized for different physics domains:

- `'thermal'`: Blue to white gradient for temperature-based phase diagrams
- `'magnetic'`: Red to white gradient for magnetic phase diagrams  
- `'topological'`: Purple to cyan gradient for topological phase diagrams

## Migration from Original Function

The enhanced function maintains full backward compatibility with the original `plot_phase_diagram` function. Simply replace:

```python
# Old function call
plot_phase_diagram(result)

# New function call (same result, but enhanced)
from etc.visualization import plot_phase_diagram
plot_phase_diagram(result)
```

All existing code will work without changes, but will benefit from the improved visual quality and fixed LaTeX warnings.

## Requirements

- numpy
- matplotlib
- scipy (for some colormap functionality)

The visualization module has no additional dependencies beyond those already required by the ETC package.