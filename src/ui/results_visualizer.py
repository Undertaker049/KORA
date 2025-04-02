"""
Module for visualizing clustering results.
"""

import numpy as np
from PyQt5.QtWidgets import QMessageBox

def visualize_results(self):

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