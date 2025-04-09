"""
User Interface Module.

Provides the user interface components for application.
"""

# Import all UI components
from PyQt5.QtWidgets import QApplication
from .app import ClusteringApp
from .data_loader import load_data_from_file
from ._perform_clustering import perform_clustering, update_data_info, preprocess_data
from ._find_optimal_k import find_optimal_k
from .results_visualizer import visualize_results, update_cluster_visualization, update_tabs_visualization
from .data_saver import save_results
from ._init_ui import init_ui, _create_menu_bar, _show_about_dialog, MatplotlibCanvas
from ._update_visualization_language import update_visualization_language

def create_app(
    data_loader_class, 
    preprocessor_class, 
    clustering_class, 
    cluster_analyzer_class, 
    visualizer_class,
    translator=None
):
    """
    Create and return a application instance.
    
    Initializes all components of the application and assembles them
    into a functioning UI.
    
    Args:
        data_loader_class: Class for loading data from files.
        preprocessor_class: Class for preprocessing and transforming raw data.
        clustering_class: Class implementing the clustering algorithm.
        cluster_analyzer_class: Class for analyzing and evaluating clustering results.
        visualizer_class: Class for visualizing data and clustering results.
        translator: Optional translator object for localization support. If None,
                   a default translator will be used.
        
    Returns:
        ClusteringApp: A fully initialized application instance ready to be displayed.
    """

    # Import localization module
    from .localization import (
        update_ui_language,
        update_menu_language,
        update_tab_language,
        update_ui_elements_language,
        change_language,
    )
    
    # Create a dictionary of UI components that will be attached to the app
    ui_components = {
        'MatplotlibCanvas': MatplotlibCanvas,
        'init_ui': init_ui,
        '_create_menu_bar': _create_menu_bar,
        '_show_about_dialog': _show_about_dialog,
        'update_ui_elements_language': update_ui_elements_language,
        'update_ui_language': update_ui_language,
        'update_menu_language': update_menu_language,
        'update_tab_language': update_tab_language,
        'change_language': change_language,
        'update_visualization_language': update_visualization_language,
        'perform_clustering': perform_clustering,
        'update_data_info': update_data_info,
        'preprocess_data': preprocess_data,
        'find_optimal_k': find_optimal_k,
        'load_data_from_file': load_data_from_file,
        'save_results': save_results,
        'visualize_results': visualize_results
    }
    
    # Initialize and return application instance
    app = ClusteringApp(
        data_loader_class,
        preprocessor_class,
        clustering_class,
        cluster_analyzer_class,
        visualizer_class,
        ui_components,
        translator
    )
    
    return app

# Export public components
__all__ = [
    'create_app',
    'ClusteringApp',
    'load_data_from_file',
    'save_results',
    'visualize_results'
]