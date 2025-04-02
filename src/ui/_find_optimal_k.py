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
        
        # Create results text
        info_text = "Elbow Method Analysis\n\n"
        info_text += "Inertia values for different k:\n"
        
        for k, inertia in zip(k_range, inertia_values):
            info_text += f"k = {k}: {inertia:.2f}\n"
            
        # Display results in the text field
        self.results_text.setText(info_text)
   
    except Exception as e:
        error_msg = f"Error finding optimal k: {str(e)}"
        QMessageBox.critical(self, "Error", error_msg)
        self.results_text.setText(error_msg)