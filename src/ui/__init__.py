"""
User interface module.
"""

# Import all UI components
from ._init_ui import init_ui, MatplotlibCanvas
from ._perform_clustering import perform_clustering, update_data_info, preprocess_data
from ._find_optimal_k import find_optimal_k
from .data_loader import load_data_from_file, generate_test_data
from .data_saver import save_results
from .results_visualizer import visualize_results
from .app import ClusteringApp

def create_app(
    data_loader_class, 
    preprocessor_class, 
    clustering_class, 
    cluster_analyzer_class, 
    visualizer_class
):
    """
    Create and return the clustering application.
    
    Args:
        data_loader_class: Class for loading data
        preprocessor_class: Class for preprocessing data
        clustering_class: Class for clustering
        cluster_analyzer_class: Class for analyzing clusters
        visualizer_class: Class for visualization
        
    Returns:
        The clustering application instance
    """
    
    app = ClusteringApp(
        data_loader_class=data_loader_class,
        preprocessor_class=preprocessor_class,
        clustering_class=clustering_class,
        cluster_analyzer_class=cluster_analyzer_class,
        visualizer_class=visualizer_class,
        ui_components={
            'init_ui': init_ui,
            'perform_clustering': perform_clustering,
            'update_data_info': update_data_info,
            'preprocess_data': preprocess_data,
            'find_optimal_k': find_optimal_k,
            'load_data_from_file': load_data_from_file,
            'generate_test_data': generate_test_data,
            'save_results': save_results,
            'visualize_results': visualize_results,
            'MatplotlibCanvas': MatplotlibCanvas
        }
    )
    return app

__all__ = [
    'create_app',
    'ClusteringApp',
    'MatplotlibCanvas',
    'init_ui',
    'perform_clustering',
    'update_data_info',
    'preprocess_data',
    'find_optimal_k',
    'load_data_from_file',
    'generate_test_data',
    'save_results',
    'visualize_results'
]