"""
User interface initialization module.
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QSpinBox,
    QTabWidget, QCheckBox, QGroupBox, QFormLayout,
    QDoubleSpinBox, QSplitter, QTextEdit
)
from PyQt5.QtCore import Qt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class MatplotlibCanvas(FigureCanvas):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MatplotlibCanvas, self).__init__(fig)

def init_ui(self):
    """
    Initialize the user interface.
    """

    # Main window setup
    self.setWindowTitle('Data Clustering Application')
    self.setGeometry(100, 100, 1200, 800)
    
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
    data_group = QGroupBox("Data")
    data_layout = QVBoxLayout()
    
    # Data loading button
    load_data_btn = QPushButton("Load Data from File...")
    load_data_btn.clicked.connect(self.load_data_from_file)
    
    # Add widgets to data group
    data_layout.addWidget(load_data_btn)
    data_group.setLayout(data_layout)
    
    # Preprocessing settings group
    preprocess_group = QGroupBox("Preprocessing")
    preprocess_layout = QFormLayout()
    
    # Data scaling
    self.scale_check = QCheckBox()
    self.scale_check.setChecked(True)
    preprocess_layout.addRow("Scale data:", self.scale_check)
    
    # Scaling method
    self.scale_method_combo = QComboBox()
    self.scale_method_combo.addItems(["standard", "minmax", "robust"])
    preprocess_layout.addRow("Scaling method:", self.scale_method_combo)
    
    # Missing values handling
    self.missing_check = QCheckBox()
    self.missing_check.setChecked(True)
    preprocess_layout.addRow("Handle missing values:", self.missing_check)
    
    # Missing values handling method
    self.missing_method_combo = QComboBox()
    self.missing_method_combo.addItems(["mean", "median", "most_frequent"])
    preprocess_layout.addRow("Handling method:", self.missing_method_combo)
    
    # Dimensionality reduction
    self.dim_reduce_check = QCheckBox()
    self.dim_reduce_check.setChecked(True)
    preprocess_layout.addRow("Reduce dimensionality:", self.dim_reduce_check)
    
    # Dimensionality reduction method
    self.dim_reduce_method_combo = QComboBox()
    self.dim_reduce_method_combo.addItems(["pca", "tsne"])
    preprocess_layout.addRow("Reduction method:", self.dim_reduce_method_combo)
    
    # Number of components
    self.n_components_spin = QSpinBox()
    self.n_components_spin.setRange(2, 100)
    self.n_components_spin.setValue(2)
    preprocess_layout.addRow("Number of components:", self.n_components_spin)
    
    # Preprocessing button
    preprocess_btn = QPushButton("Preprocess Data")
    preprocess_btn.clicked.connect(self.preprocess_data)
    
    preprocess_layout.addRow(preprocess_btn)
    preprocess_group.setLayout(preprocess_layout)
    
    # Clustering settings group
    cluster_group = QGroupBox("Clustering")
    cluster_layout = QFormLayout()
    
    # Number of clusters
    self.n_clusters_spin = QSpinBox()
    self.n_clusters_spin.setRange(2, 20)
    self.n_clusters_spin.setValue(3)
    cluster_layout.addRow("Number of clusters (k):", self.n_clusters_spin)
    
    # Maximum number of iterations
    self.max_iter_spin = QSpinBox()
    self.max_iter_spin.setRange(100, 1000)
    self.max_iter_spin.setValue(300)
    self.max_iter_spin.setSingleStep(100)
    cluster_layout.addRow("Max iterations:", self.max_iter_spin)
    
    # Clustering buttons
    cluster_btn = QPushButton("Perform Clustering")
    cluster_btn.clicked.connect(self.perform_clustering)
    
    find_optimal_k_btn = QPushButton("Find Optimal k")
    find_optimal_k_btn.clicked.connect(self.find_optimal_k)
    
    # Results saving button
    save_results_btn = QPushButton("Save Results...")
    save_results_btn.clicked.connect(self.save_results)
    
    cluster_layout.addRow(cluster_btn)
    cluster_layout.addRow(find_optimal_k_btn)
    cluster_layout.addRow(save_results_btn)
    
    cluster_group.setLayout(cluster_layout)
    
    # Data and Results Information Group
    info_group = QGroupBox("Information")
    info_layout = QVBoxLayout()
    
    # Data information
    self.data_info_label = QLabel("No data loaded")
    info_layout.addWidget(self.data_info_label)
    
    # Results information
    results_label = QLabel("Clustering Results:")
    info_layout.addWidget(results_label)
    
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
    self.tabs.addTab(self.clusters_tab, "Clusters")
    self.tabs.addTab(self.elbow_tab, "Elbow Method")
    self.tabs.addTab(self.silhouette_tab, "Silhouette Analysis")
    self.tabs.addTab(self.features_tab, "Feature Importance")
    
    right_layout.addWidget(self.tabs)
    
    # Add panels to splitter
    splitter.addWidget(left_panel)
    splitter.addWidget(right_panel)
    
    # Set initial splitter sizes
    splitter.setSizes([300, 900])
    
    # Show window
    self.show()