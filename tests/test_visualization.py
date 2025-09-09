"""
Test the enhanced visualization module.
"""

import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt

from etc.visualization import plot_phase_diagram, create_custom_colormap, PHASE_DIAGRAM_CMAPS


def create_sample_data():
    """Create sample phase diagram data for testing."""
    result = {}
    
    # Create a simple 3x3 grid of data
    k_values = [2, 3, 4]
    scale_values = [1, 2, 3]
    
    for k in k_values:
        result[k] = {}
        for scale in scale_values:
            # Create some synthetic phase diagram data
            mu_gamma_ratio = scale * 0.5  # Simple relationship
            hmin_value = np.sin(k + scale) * 10  # Some variation
            result[k][scale] = (mu_gamma_ratio, hmin_value)
    
    return result


def test_plot_phase_diagram_basic():
    """Test basic functionality of plot_phase_diagram."""
    result = create_sample_data()
    
    # Should not raise any exceptions
    plot_phase_diagram(result, figsize=(8, 6))
    plt.close('all')


def test_plot_phase_diagram_empty_data():
    """Test plot_phase_diagram with empty data."""
    with pytest.raises(ValueError, match="The result dictionary is empty"):
        plot_phase_diagram({})


def test_plot_phase_diagram_invalid_keys():
    """Test plot_phase_diagram with invalid key types."""
    invalid_result = {'invalid_key': {1: (0.5, 1.0)}}
    
    with pytest.raises(ValueError, match="Keys of the result dictionary must be numeric"):
        plot_phase_diagram(invalid_result)


def test_plot_phase_diagram_invalid_scale_values():
    """Test plot_phase_diagram with invalid scale values."""
    invalid_result = {1: {'invalid_scale': (0.5, 1.0)}}
    
    with pytest.raises(ValueError, match="Scale values in the result dictionary must be numeric"):
        plot_phase_diagram(invalid_result)


def test_plot_phase_diagram_customization():
    """Test plot_phase_diagram with custom parameters."""
    result = create_sample_data()
    
    # Test with various customization options
    plot_phase_diagram(
        result,
        figsize=(10, 8),
        cmap='plasma',
        add_contours=False,
        grid=False,
        title='Custom Phase Diagram',
        colorbar_shrink=0.9
    )
    plt.close('all')


def test_create_custom_colormap():
    """Test custom colormap creation."""
    colors = ['red', 'yellow', 'blue']
    cmap = create_custom_colormap(colors, 'test_cmap')
    
    assert cmap.name == 'test_cmap'
    assert cmap.N == 256  # Default resolution


def test_predefined_colormaps():
    """Test predefined colormap definitions."""
    assert 'thermal' in PHASE_DIAGRAM_CMAPS
    assert 'magnetic' in PHASE_DIAGRAM_CMAPS
    assert 'topological' in PHASE_DIAGRAM_CMAPS
    
    # Check that each colormap has valid color definitions
    for cmap_name, colors in PHASE_DIAGRAM_CMAPS.items():
        assert len(colors) >= 2  # Need at least 2 colors for a gradient
        cmap = create_custom_colormap(colors, cmap_name)
        assert cmap is not None


def test_plot_phase_diagram_with_contours():
    """Test plot_phase_diagram with contour lines."""
    result = create_sample_data()
    
    # Test with contours enabled
    plot_phase_diagram(
        result,
        add_contours=True,
        contour_levels=5
    )
    plt.close('all')


if __name__ == "__main__":
    pytest.main([__file__])