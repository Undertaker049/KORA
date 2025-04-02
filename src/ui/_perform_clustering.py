"""
Module for performing clustering operations.
"""

import numpy as np
from PyQt5.QtWidgets import QMessageBox

def perform_clustering(self):

    if self.processed_data is None:
        QMessageBox.warning(self, "Warning", "Preprocess data first.")
        return
        
    try:

        # Get clustering parameters
        n_clusters = self.n_clusters_spin.value()
        max_iter = self.max_iter_spin.value()
        
        # Create model
        self.kmeans = self.clustering_class(
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