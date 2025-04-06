"""
User Interface Initialization Module.

Handles the creation of UI elements, layout design, and menu structure.
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QSpinBox,
    QTabWidget, QCheckBox, QGroupBox, QFormLayout,
    QDoubleSpinBox, QSplitter, QTextEdit,
    QMenuBar, QMenu, QAction, QActionGroup
)
from PyQt5.QtCore import Qt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class MatplotlibCanvas(FigureCanvas):
    """
    Custom canvas class for embedding Matplotlib figures in PyQt applications.
    
    Extends FigureCanvas to provide a way to integrate
    matplotlib visualizations directly into PyQt GUIs. 
    Each canvas contains a figure and axes that can be used for plotting.
    
    Attributes:
        axes: The Matplotlib Axes object where plots are drawn.
        figure: The Matplotlib Figure object containing the axes.
        
    Parameters:
        parent (QWidget): The parent widget, default is None.
        width (float): Figure width in inches, default is 5.
        height (float): Figure height in inches, default is 4.
        dpi (int): Dots per inch (resolution), default is 100.
    """

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        """
        Initialize a new MatplotlibCanvas instance.
        
        Creates a new Figure with the specified dimensions and a single Axes
        object (subplot) that can be used for plotting data.
        
        Args:
            parent (QWidget, optional): Parent widget for the canvas. Defaults to None.
            width (float, optional): Width of the figure in inches. Defaults to 5.
            height (float, optional): Height of the figure in inches. Defaults to 4.
            dpi (int, optional): Dots per inch (resolution). Defaults to 100.
        """
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MatplotlibCanvas, self).__init__(fig)

def init_ui(self):
    """
    Initialize the complete user interface for the application.
    
    Creates and configures all UI components, including:
    - Main window setup with title and size
    - Menu bar with all actions
    - Left panel with settings for data, preprocessing, and clustering
    - Right panel with visualization tabs
    - Splitter to allow resizing of panels
    
    Each UI element is connected to the appropriate callback function,
    and translations are applied based on the current locale.
    
    Parameters:
        self: The parent application instance containing:
            - translator: For localizing UI text
            - Various callback methods for UI actions
            
    Returns:
        None: UI elements are attached to the application instance.
    """

    # Get translator for localization
    tr = self.translator.translate

    # Main window setup
    self.setWindowTitle(tr('app_title'))
    self.setGeometry(100, 100, 1200, 800)
    
    # Create menu bar
    self._create_menu_bar()
    
    # Create central widget
    central_widget = QWidget()
    self.setCentralWidget(central_widget)
    
    # Create main layout
    main_layout = QHBoxLayout(central_widget)
    
    # Create splitter for settings and visualization area
    splitter = QSplitter(Qt.Horizontal)
    main_layout.addWidget(splitter)
    
    # Left panel with settings
    left_panel = QWidget()
    left_layout = QVBoxLayout(left_panel)
    
    # Data settings group
    data_group = QGroupBox(tr('tab_data'))
    data_layout = QVBoxLayout()
    
    # Data loading button
    self.load_data_btn = QPushButton(tr('menu_open'))
    self.load_data_btn.clicked.connect(self.load_data_from_file)
    
    # Add widgets to data group
    data_layout.addWidget(self.load_data_btn)
    data_group.setLayout(data_layout)
    
    # Preprocessing settings group
    preprocess_group = QGroupBox(tr('menu_preprocess'))
    preprocess_layout = QFormLayout()
    
    # Data scaling
    self.scale_check = QCheckBox()
    self.scale_check.setChecked(True)
    preprocess_layout.addRow(tr('preprocess_scale') + ":", self.scale_check)
    
    # Scaling method
    self.scale_method_combo = QComboBox()
    self.scale_method_combo.addItems(["standard", "minmax", "robust"])
    preprocess_layout.addRow(tr('preprocess_scale') + ":", self.scale_method_combo)
    
    # Missing values handling
    self.missing_check = QCheckBox()
    self.missing_check.setChecked(True)
    preprocess_layout.addRow(tr('preprocess_missing') + ":", self.missing_check)
    
    # Missing values handling method
    self.missing_method_combo = QComboBox()
    self.missing_method_combo.addItems(["mean", "median", "most_frequent"])
    preprocess_layout.addRow(tr('preprocess_missing') + ":", self.missing_method_combo)
    
    # Dimensionality reduction
    self.dim_reduce_check = QCheckBox()
    self.dim_reduce_check.setChecked(True)
    preprocess_layout.addRow(tr('viz_method') + ":", self.dim_reduce_check)
    
    # Dimensionality reduction method
    self.dim_reduce_method_combo = QComboBox()
    self.dim_reduce_method_combo.addItems(["pca", "tsne"])
    preprocess_layout.addRow(tr('viz_method') + ":", self.dim_reduce_method_combo)
    
    # Number of components
    self.n_components_spin = QSpinBox()
    self.n_components_spin.setRange(2, 100)
    self.n_components_spin.setValue(2)
    preprocess_layout.addRow(tr('data_features') + ":", self.n_components_spin)
    
    # Preprocessing button
    self.preprocess_btn = QPushButton(tr('preprocess_button'))
    self.preprocess_btn.clicked.connect(self.preprocess_data)
    
    preprocess_layout.addRow(self.preprocess_btn)
    preprocess_group.setLayout(preprocess_layout)
    
    # Clustering settings group
    cluster_group = QGroupBox(tr('tab_clustering'))
    cluster_layout = QFormLayout()
    
    # Number of clusters
    self.n_clusters_spin = QSpinBox()
    self.n_clusters_spin.setRange(2, 20)
    self.n_clusters_spin.setValue(3)
    cluster_layout.addRow(tr('clustering_k') + ":", self.n_clusters_spin)
    
    # Maximum number of iterations
    self.max_iter_spin = QSpinBox()
    self.max_iter_spin.setRange(100, 1000)
    self.max_iter_spin.setValue(300)
    self.max_iter_spin.setSingleStep(100)
    cluster_layout.addRow(tr('clustering_max_iter') + ":", self.max_iter_spin)
    
    # Clustering buttons
    self.cluster_btn = QPushButton(tr('clustering_button'))
    self.cluster_btn.clicked.connect(self.perform_clustering)
    
    self.find_optimal_k_btn = QPushButton(tr('optimal_k_button'))
    self.find_optimal_k_btn.clicked.connect(self.find_optimal_k)
    
    # Results saving button
    self.save_results_btn = QPushButton(tr('button_save'))
    self.save_results_btn.clicked.connect(self.save_results)
    
    cluster_layout.addRow(self.cluster_btn)
    cluster_layout.addRow(self.find_optimal_k_btn)
    cluster_layout.addRow(self.save_results_btn)
    
    cluster_group.setLayout(cluster_layout)
    
    # Data and Results Information Group
    info_group = QGroupBox(tr('data_info'))
    info_layout = QVBoxLayout()
    
    # Data information
    self.data_info_label = QLabel(tr('data_preview'))
    info_layout.addWidget(self.data_info_label)
    
    # Results information
    self.results_label = QLabel(tr('clustering_results') + ":")
    info_layout.addWidget(self.results_label)
    
    self.results_text = QTextEdit()
    self.results_text.setReadOnly(True)
    self.results_text.setMinimumHeight(150)
    info_layout.addWidget(self.results_text)
    
    info_group.setLayout(info_layout)
    
    # Add groups to left panel
    left_layout.addWidget(data_group)
    left_layout.addWidget(preprocess_group)
    left_layout.addWidget(cluster_group)
    left_layout.addWidget(info_group)
    left_layout.addStretch()
    
    # Right panel with visualization
    right_panel = QWidget()
    right_layout = QVBoxLayout(right_panel)
    
    # Create tabs for different visualizations
    self.tabs = QTabWidget()
    
    # Tab with cluster visualization
    self.clusters_tab = QWidget()
    clusters_layout = QVBoxLayout(self.clusters_tab)
    self.clusters_canvas = MatplotlibCanvas(width=8, height=6)
    clusters_layout.addWidget(self.clusters_canvas)
    
    # Tab with elbow method
    self.elbow_tab = QWidget()
    elbow_layout = QVBoxLayout(self.elbow_tab)
    self.elbow_canvas = MatplotlibCanvas(width=8, height=6)
    elbow_layout.addWidget(self.elbow_canvas)
    
    # Tab with silhouette coefficients
    self.silhouette_tab = QWidget()
    silhouette_layout = QVBoxLayout(self.silhouette_tab)
    self.silhouette_canvas = MatplotlibCanvas(width=8, height=6)
    silhouette_layout.addWidget(self.silhouette_canvas)
    
    # Tab with feature importance
    self.features_tab = QWidget()
    features_layout = QVBoxLayout(self.features_tab)
    self.features_canvas = MatplotlibCanvas(width=8, height=6)
    features_layout.addWidget(self.features_canvas)
    
    # Add tabs
    self.tabs.addTab(self.clusters_tab, tr('plot_cluster'))
    self.tabs.addTab(self.elbow_tab, tr('plot_elbow_title'))
    self.tabs.addTab(self.silhouette_tab, tr('plot_silhouette_title'))
    self.tabs.addTab(self.features_tab, tr('data_features'))
    
    # Add tabs to right panel
    right_layout.addWidget(self.tabs)
    
    # Add panels to splitter
    splitter.addWidget(left_panel)
    splitter.addWidget(right_panel)
    
    # Set initial splitter proportions (30% left, 70% right)
    splitter.setSizes([300, 700])

def _create_menu_bar(self):
    """
    Create and configure the application menu bar.
    
    Menu structure includes:
    - File menu (open, exit)
    - Data menu (preprocessing)
    - Analysis menu (optimal K, clustering)
    - Results menu (save, visualize)
    - Settings menu (language selection)
    - Help menu (about)
    
    Parameters:
        self: The parent application instance containing:
            - translator: For localizing menu text
            - Callback methods for menu actions
            - Language settings and available languages
            
    Returns:
        None: Menu items are attached to the application instance.
    """
    tr = self.translator.translate
    
    # Create menu bar
    menubar = self.menuBar()
    
    # File menu
    self.menu_file = menubar.addMenu(tr('menu_file'))
    
    # Open action
    self.action_open = QAction(tr('menu_open'), self)
    self.action_open.triggered.connect(self.load_data_from_file)
    self.menu_file.addAction(self.action_open)
    
    # Exit action
    self.action_exit = QAction(tr('menu_exit'), self)
    self.action_exit.triggered.connect(self.close)
    self.menu_file.addAction(self.action_exit)
    
    # Data menu
    self.menu_data = menubar.addMenu(tr('menu_data'))
    
    # Preprocess action
    self.action_preprocess = QAction(tr('menu_preprocess'), self)
    self.action_preprocess.triggered.connect(self.preprocess_data)
    self.menu_data.addAction(self.action_preprocess)
    
    # Analysis menu
    self.menu_analysis = menubar.addMenu(tr('menu_analysis'))
    
    # Find optimal k action
    self.action_find_optimal_k = QAction(tr('menu_find_optimal_k'), self)
    self.action_find_optimal_k.triggered.connect(self.find_optimal_k)
    self.menu_analysis.addAction(self.action_find_optimal_k)
    
    # Perform clustering action
    self.action_perform_clustering = QAction(tr('menu_perform_clustering'), self)
    self.action_perform_clustering.triggered.connect(self.perform_clustering)
    self.menu_analysis.addAction(self.action_perform_clustering)
    
    # Results menu
    self.menu_results = menubar.addMenu(tr('menu_results'))
    
    # Save results action
    self.action_save_results = QAction(tr('menu_save_results'), self)
    self.action_save_results.triggered.connect(self.save_results)
    self.menu_results.addAction(self.action_save_results)
    
    # Visualize action
    self.action_visualize = QAction(tr('menu_visualize'), self)
    self.action_visualize.triggered.connect(self.visualize_results)
    self.menu_results.addAction(self.action_visualize)
    
    # Settings menu
    self.menu_settings = menubar.addMenu(tr('menu_settings'))
    
    # Language menu
    self.menu_language = self.menu_settings.addMenu(tr('menu_language'))
    
    # Create language action group to ensure only one language is selected
    language_group = QActionGroup(self)
    
    # Dynamically create language menu items
    for lang in self.available_languages:

        # Try to get the translated name for the language
        lang_key = f'menu_language_{lang}'
        lang_name = tr(lang_key)
        
        # Create language action
        action = QAction(lang_name, self)
        action.setCheckable(True)
        action.setChecked(self.current_language == lang)
        
        # Use lambda with default argument to prevent closure issues
        action.triggered.connect(lambda checked, l=lang: self.change_language(l))
        
        # Add to action group and menu
        language_group.addAction(action)
        self.menu_language.addAction(action)
        
        # Store the action for later reference
        self.language_actions[lang] = action
    
    # Help menu
    self.menu_help = menubar.addMenu(tr('menu_help'))
    
    # About action
    self.action_about = QAction(tr('menu_about'), self)
    self.action_about.triggered.connect(self._show_about_dialog)
    self.menu_help.addAction(self.action_about)

def _show_about_dialog(self):
    """
    Display a dialog with information about the application.
    
    Parameters:
        self: The parent application instance containing:
            - translator: For localizing dialog text
            
    Returns:
        None: Dialog is displayed and control returns when user closes it.
    """
    from PyQt5.QtWidgets import QMessageBox
    
    about_box = QMessageBox(self)
    about_box.setWindowTitle(self.translator.translate('about_title'))
    about_box.setText(self.translator.translate('about_text'))
    about_box.setIcon(QMessageBox.Information)
    about_box.exec_()