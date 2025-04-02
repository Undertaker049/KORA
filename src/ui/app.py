"""
Main application module with GUI.
"""

from PyQt5.QtWidgets import QMainWindow

class ClusteringApp(QMainWindow):
    """
    Main application for data clustering with GUI.
    """
    
    def __init__(self, 
                 data_loader_class, 
                 preprocessor_class, 
                 clustering_class, 
                 cluster_analyzer_class, 
                 visualizer_class,
                 ui_components):
        """
        Initialize the application.
        
        Args:
            data_loader_class: Class for loading data
            preprocessor_class: Class for preprocessing data
            clustering_class: Class for clustering
            cluster_analyzer_class: Class for analyzing clusters
            visualizer_class: Class for visualization
            ui_components: Dictionary with UI component functions
        """
        super().__init__()
        
        # Initialize UI components
        self.MatplotlibCanvas = ui_components['MatplotlibCanvas']
        
        # Component initialization
        self.data_loader = data_loader_class()
        self.preprocessor = preprocessor_class()
        self.clustering_class = clustering_class
        self.cluster_analyzer = cluster_analyzer_class()
        self.visualizer = visualizer_class()
        
        # Data
        self.data = None
        self.processed_data = None
        self.labels = None
        self.reduced_data = None
        self.original_columns = None
        self.kmeans = None
        
        # Attach methods from UI components
        self.init_ui = ui_components['init_ui'].__get__(self)
        self.perform_clustering = ui_components['perform_clustering'].__get__(self)
        self.update_data_info = ui_components['update_data_info'].__get__(self)
        self.preprocess_data = ui_components['preprocess_data'].__get__(self)
        self.find_optimal_k = ui_components['find_optimal_k'].__get__(self)
        self.load_data_from_file = ui_components['load_data_from_file'].__get__(self)
        self.save_results = ui_components['save_results'].__get__(self)
        self.visualize_results = ui_components['visualize_results'].__get__(self)
        
        # Interface setup
        self.init_ui()