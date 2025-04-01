"""
Main application module with graphical user interface.
"""

import sys
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QComboBox, QSpinBox,
    QTabWidget, QCheckBox, QGroupBox, QFormLayout, QMessageBox,
    QDoubleSpinBox, QSplitter
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from ..data_processing import DataLoader, DataPreprocessor
from ..clustering import KMeansClustering, ClusterAnalyzer
from ..visualization import ClusterVisualizer

class MatplotlibCanvas(FigureCanvas):
    """
    Class for displaying Matplotlib plots in PyQt5.
    """
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MatplotlibCanvas, self).__init__(fig)

class ClusteringApp(QMainWindow):
    
    def __init__(self):
        """
        Initialize the application.
        """
        super().__init__()
        
        # Component initialization
        self.data_loader = DataLoader()
        self.preprocessor = DataPreprocessor()
        self.kmeans = KMeansClustering()
        self.cluster_analyzer = ClusterAnalyzer()
        self.visualizer = ClusterVisualizer()
        
        # Data
        self.data = None
        self.processed_data = None
        self.labels = None
        self.reduced_data = None
        self.original_columns = None  # For storing original column names
        
        # Interface setup
        self.init_ui()
        
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
        
        # Data loading buttons
        load_data_btn = QPushButton("Load Data from File...")
        load_data_btn.clicked.connect(self.load_data_from_file)
        
        generate_data_btn = QPushButton("Generate Test Data")
        generate_data_btn.clicked.connect(self.generate_test_data)
        
        # Data generation settings
        gen_data_settings = QGroupBox("Generation Settings")
        gen_data_layout = QFormLayout()
        
        self.n_samples_spin = QSpinBox()
        self.n_samples_spin.setRange(10, 10000)
        self.n_samples_spin.setValue(1000)
        gen_data_layout.addRow("Number of points:", self.n_samples_spin)
        
        self.n_features_spin = QSpinBox()
        self.n_features_spin.setRange(2, 100)
        self.n_features_spin.setValue(2)
        gen_data_layout.addRow("Number of features:", self.n_features_spin)
        
        self.n_clusters_gen_spin = QSpinBox()
        self.n_clusters_gen_spin.setRange(2, 20)
        self.n_clusters_gen_spin.setValue(3)
        gen_data_layout.addRow("Number of clusters:", self.n_clusters_gen_spin)
        
        gen_data_settings.setLayout(gen_data_layout)
        
        # Add widgets to data group
        data_layout.addWidget(load_data_btn)
        data_layout.addWidget(generate_data_btn)
        data_layout.addWidget(gen_data_settings)
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
        
        # Data information
        self.data_info_label = QLabel("No data loaded")
        
        # Add groups to left panel
        left_layout.addWidget(data_group)
        left_layout.addWidget(preprocess_group)
        left_layout.addWidget(cluster_group)
        left_layout.addWidget(self.data_info_label)
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
        
    def load_data_from_file(self):
        """
        Load data from a file.
        """
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Data",
            "",
            "CSV files (*.csv);;Excel files (*.xlsx *.xls);;NumPy files (*.npy);;All files (*)",
            options=options
        )
        
        if not file_path:
            return
            
        try:

            if file_path.endswith('.csv'):
                data_df = self.data_loader.load_csv(file_path)
                self.original_columns = data_df.columns.tolist()
                self.data = data_df.values

            elif file_path.endswith(('.xlsx', '.xls')):
                data_df = self.data_loader.load_excel(file_path)
                self.original_columns = data_df.columns.tolist()
                self.data = data_df.values

            elif file_path.endswith('.npy'):
                self.data = self.data_loader.load_numpy(file_path)
                self.original_columns = [f"Feature_{i}" for i in range(self.data.shape[1])]
            
            else:
                self.data = self.data_loader.load_csv(file_path)

                if isinstance(self.data, pd.DataFrame):
                    self.original_columns = self.data.columns.tolist()
                    self.data = self.data.values

                else:
                    self.original_columns = [f"Feature_{i}" for i in range(self.data.shape[1])]
                
            self.update_data_info()
            QMessageBox.information(self, "Success", "Data successfully loaded.")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading data: {str(e)}")
            
    def generate_test_data(self):
        """
        Generate test data.
        """

        try:
            n_samples = self.n_samples_spin.value()
            n_features = self.n_features_spin.value()
            n_clusters = self.n_clusters_gen_spin.value()
            
            self.data, self.true_labels = self.data_loader.generate_sample_data(
                n_samples=n_samples,
                n_features=n_features,
                n_clusters=n_clusters,
                random_state=42
            )
            
            # Create column names for generated data
            self.original_columns = [f"Feature_{i}" for i in range(self.data.shape[1])]
            
            self.update_data_info()
            QMessageBox.information(self, "Success", "Test data successfully generated.")
       
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error generating data: {str(e)}")
            
    def update_data_info(self):
        """
        Update data information.
        """

        if self.data is not None:
            n_samples, n_features = self.data.shape
            self.data_info_label.setText(f"Data: {n_samples} points, {n_features} features")
        
        else:
            self.data_info_label.setText("No data loaded")
            
    def preprocess_data(self):
        """
        Preprocess data.
        """

        if self.data is None:
            QMessageBox.warning(self, "Warning", "Load data first.")
            return
            
        try:
            # Get preprocessing parameters
            scale = self.scale_check.isChecked()
            scaling_method = self.scale_method_combo.currentText()
            handle_missing = self.missing_check.isChecked()
            missing_strategy = self.missing_method_combo.currentText()
            reduce_dims = self.dim_reduce_check.isChecked()
            n_components = self.n_components_spin.value()
            dim_reduce_method = self.dim_reduce_method_combo.currentText()
            
            # Preprocess data
            self.processed_data = self.preprocessor.preprocess_pipeline(
                self.data,
                scale=scale,
                scaling_method=scaling_method,
                handle_missing=handle_missing,
                missing_strategy=missing_strategy,
                reduce_dims=False  # Dimensionality reduction is done separately
            )
            
            # Reduce dimensionality if needed
            if reduce_dims:
                self.reduced_data = self.visualizer.reduce_dimensions(
                    self.processed_data,
                    method=dim_reduce_method,
                    n_components=n_components
                )
                
            else:
                self.reduced_data = self.processed_data
                
            QMessageBox.information(self, "Success", "Data successfully preprocessed.")
        
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error preprocessing data: {str(e)}")
            
    def perform_clustering(self):
        """
        Perform clustering.
        """

        if self.processed_data is None:
            QMessageBox.warning(self, "Warning", "Preprocess data first.")
            return
            
        try:
            # Get clustering parameters
            n_clusters = self.n_clusters_spin.value()
            max_iter = self.max_iter_spin.value()
            
            # Create model
            self.kmeans = KMeansClustering(
                n_clusters=n_clusters,
                max_iter=max_iter,
                random_state=42
            )
            
            # Perform clustering
            self.labels = self.kmeans.fit_predict(self.processed_data)
            
            # Evaluate results
            evaluation = self.kmeans.evaluate(self.processed_data)
            
            # Visualize results
            self.visualize_results()
            
            # Results information
            info_text = "Clustering Results:\n"
            info_text += f"Number of clusters: {n_clusters}\n"
            info_text += f"Inertia: {evaluation['inertia']:.4f}\n"
            
            if 'silhouette_score' in evaluation:
                info_text += f"Silhouette coefficient: {evaluation['silhouette_score']:.4f}\n"
                
            if 'calinski_harabasz_score' in evaluation:
                info_text += f"Calinski-Harabasz index: {evaluation['calinski_harabasz_score']:.4f}\n"
                
            if 'davies_bouldin_score' in evaluation:
                info_text += f"Davies-Bouldin index: {evaluation['davies_bouldin_score']:.4f}\n"
                
            QMessageBox.information(self, "Clustering Results", info_text)
        
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error performing clustering: {str(e)}")
            
    def find_optimal_k(self):
        """
        Find the optimal number of clusters.
        """

        if self.processed_data is None:
            QMessageBox.warning(self, "Warning", "Preprocess data first.")
            return
            
        try:
            # Create temporary model
            temp_kmeans = KMeansClustering()
            
            # Find optimal k
            k_range, inertia_values = temp_kmeans.optimal_k_elbow(
                self.processed_data,
                k_range=list(range(1, 11))
            )
            
            # Visualize elbow method
            fig = self.visualizer.plot_elbow_method(k_range, inertia_values)
            self.elbow_canvas.figure = fig
            self.elbow_canvas.draw()
            
            # Switch to elbow method tab
            self.tabs.setCurrentIndex(1)
            
            QMessageBox.information(self, "Optimal k", "Elbow method completed. Examine the plot to determine the optimal number of clusters.")
       
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error finding optimal k: {str(e)}")
    
    def save_results(self):
        """
        Save clustering results to a file.
        """

        if self.data is None or self.labels is None:
            QMessageBox.warning(self, "Warning", "No data to save. Perform clustering first.")
            return
            
        try:
            options = QFileDialog.Options()
            file_path, selected_filter = QFileDialog.getSaveFileName(
                self,
                "Save Clustering Results",
                "",
                "CSV files (*.csv);;Excel files (*.xlsx);;NumPy files (*.npy)",
                options=options
            )
            
            if not file_path:
                return
                
            # Create DataFrame with original data and cluster labels
            if self.original_columns is None:
                self.original_columns = [f"Feature_{i}" for i in range(self.data.shape[1])]
                
            results_df = pd.DataFrame(self.data, columns=self.original_columns)
            results_df['cluster'] = self.labels
            
            # Add coordinates in reduced dimensionality if available
            if self.reduced_data is not None and self.reduced_data.shape[1] == 2:
                results_df['x_reduced'] = self.reduced_data[:, 0]
                results_df['y_reduced'] = self.reduced_data[:, 1]
            
            # Save according to selected format
            if file_path.endswith('.csv') or selected_filter == "CSV files (*.csv)":

                if not file_path.endswith('.csv'):
                    file_path += '.csv'

                results_df.to_csv(file_path, index=False)

            elif file_path.endswith('.xlsx') or selected_filter == "Excel files (*.xlsx)":

                if not file_path.endswith('.xlsx'):
                    file_path += '.xlsx'

                results_df.to_excel(file_path, index=False)

            elif file_path.endswith('.npy') or selected_filter == "NumPy files (*.npy)":

                if not file_path.endswith('.npy'):
                    file_path += '.npy'

                # For NumPy, save only the array with cluster labels
                np.save(file_path, self.labels)

            else:
                # Default save as CSV
                if not file_path.endswith('.csv'):
                    file_path += '.csv'

                results_df.to_csv(file_path, index=False)
                
            QMessageBox.information(self, "Success", f"Clustering results successfully saved to\n{file_path}")
       
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error saving results: {str(e)}")
            
    def visualize_results(self):
        """
        Visualize clustering results.
        """
        
        if self.labels is None:
            return
            
        try:
            # Cluster visualization
            if self.reduced_data.shape[1] > 2:

                # If data has more than 2 dimensions, apply PCA for visualization
                vis_data = self.visualizer.reduce_dimensions(self.processed_data, n_components=2)
                centers = None  # Don't visualize centers as they're in a different space
            
            else:
                vis_data = self.reduced_data

                # If dimensionality was reduced, transform centers too
                if self.reduced_data.shape[1] == 2 and self.kmeans.cluster_centers_ is not None:
                    
                    if self.kmeans.cluster_centers_.shape[1] == self.reduced_data.shape[1]:
                        centers = self.kmeans.cluster_centers_
                    
                    else:
                        centers = None
                
                else:
                    centers = None
                    
            # Cluster visualization
            fig_clusters = self.visualizer.plot_clusters_2d(
                vis_data,
                self.labels,
                centers=centers
            )
            self.clusters_canvas.figure = fig_clusters
            self.clusters_canvas.draw()
            
            # Silhouette analysis (if more than one cluster)
            if len(np.unique(self.labels)) > 1:

                try:
                    fig_silhouette = self.visualizer.plot_silhouette(
                        self.processed_data,
                        self.labels
                    )
                    self.silhouette_canvas.figure = fig_silhouette
                    self.silhouette_canvas.draw()

                except Exception as e:
                    print(f"Error visualizing silhouette analysis: {str(e)}")
                    
            # Feature importance (if cluster centers exist)
            if self.kmeans.cluster_centers_ is not None:
                feature_names = [f"Feature {i+1}" for i in range(self.kmeans.cluster_centers_.shape[1])]
                fig_features = self.visualizer.plot_feature_importance(
                    self.kmeans.cluster_centers_,
                    feature_names
                )
                self.features_canvas.figure = fig_features
                self.features_canvas.draw()
                
            # Switch to clusters tab
            self.tabs.setCurrentIndex(0)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error visualizing results: {str(e)}")

def main():
    """
    Application entry point.
    """
    app = QApplication(sys.argv)
    clustering_app = ClusteringApp()
    sys.exit(app.exec_())