"""
Main application module with GUI.

Defines the core application class for the app.
"""

from PyQt5.QtWidgets import QMainWindow, QAction, QActionGroup

class ClusteringApp(QMainWindow):
    """
    Main application class for data clustering with graphical user interface.
    
    Creates and manages the main window, menu system, and all UI components.
    
    Attributes:
        translator: Localization handler providing translation services
        current_language: Currently active language code
        available_languages: List of supported language codes
        data_loader: Component for loading datasets from various sources
        preprocessor: Component for data cleaning and transformation
        clustering_class: Class providing clustering algorithms
        cluster_analyzer: Component for evaluating clustering results
        visualizer: Component for generating visualizations
        data: Original loaded dataset
        processed_data: Dataset after preprocessing
        labels: Cluster assignments from clustering algorithm
        reduced_data: Dimensionality-reduced data for visualization
        language_actions: Dictionary of language selection menu actions
    """
    
    def __init__(self, 
                 data_loader_class, 
                 preprocessor_class, 
                 clustering_class, 
                 cluster_analyzer_class, 
                 visualizer_class,
                 ui_components,
                 translator=None):
        """
        Initialize the application with all required components.
        
        Sets up the main app window and initializes all
        required components and data structures.
        
        Parameters:
            data_loader_class: Class for loading and parsing datasets from files
                Must implement methods for reading various file formats
            preprocessor_class: Class for data preprocessing operations
                Must implement methods for scaling, handling missing values, etc.
            clustering_class: Class implementing clustering algorithms
                Must provide fit_predict() and evaluate() methods
            cluster_analyzer_class: Class for analyzing clustering results
                Must provide methods for computing quality metrics
            visualizer_class: Class for generating data visualizations
                Must implement methods for creating various plot types
            ui_components: Dictionary containing UI initialization functions
                Must include all required UI component functions as key-value pairs
            translator: Optional translator object for localization
                If None, a default translator will be created from ui_components
        """
        super().__init__()
        
        # Initialize translator
        if translator is None:

            if 'get_translator' in ui_components:
                self.translator = ui_components['get_translator']()

            else:
                raise ValueError("Translator not provided and get_translator not in ui_components")
        
        else:
            self.translator = translator
            
        self.current_language = self.translator.language
        self.available_languages = self.translator.available_languages
        
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
        
        # Language actions
        self.language_actions = {}
        
        # Attach methods from UI components
        self.init_ui = ui_components['init_ui'].__get__(self)
        self._create_menu_bar = ui_components['_create_menu_bar'].__get__(self)
        self._show_about_dialog = ui_components['_show_about_dialog'].__get__(self)
        self.update_ui_elements_language = ui_components['update_ui_elements_language'].__get__(self)
        self.perform_clustering = ui_components['perform_clustering'].__get__(self)
        self.update_data_info = ui_components['update_data_info'].__get__(self)
        self.preprocess_data = ui_components['preprocess_data'].__get__(self)
        self.find_optimal_k = ui_components['find_optimal_k'].__get__(self)
        self.load_data_from_file = ui_components['load_data_from_file'].__get__(self)
        self.save_results = ui_components['save_results'].__get__(self)
        self.visualize_results = ui_components['visualize_results'].__get__(self)
        self.update_visualization_language = ui_components['update_visualization_language'].__get__(self)
        
        # Import localization functions from provided components
        self.change_language = ui_components['change_language'].__get__(self)
        self.update_ui_language = ui_components['update_ui_language'].__get__(self)
        self.update_menu_language = ui_components['update_menu_language'].__get__(self)
        self.update_tab_language = ui_components['update_tab_language'].__get__(self)
        
        # Interface setup
        self.init_ui()