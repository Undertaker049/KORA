"""
Module for finding the optimal number of clusters.
"""

from PyQt5.QtWidgets import QMessageBox

def find_optimal_k(self):
    """
    Find the optimal number of clusters.
    """
    
    if self.processed_data is None:
        QMessageBox.warning(self, "Warning", "Preprocess data first.")
        return
        
    try:
        # Create temporary model
        temp_kmeans = self.clustering_class()
        
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