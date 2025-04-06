"""
Module for finding the optimal number of clusters.

Provides functionality to determine the optimal K parameter
for K-means clustering using the Elbow method.
"""

from PyQt5.QtWidgets import QMessageBox

def find_optimal_k(self):
    """
    Find the optimal number of clusters using the Elbow method.
    
    Computes inertia values for a range of cluster counts (K values),
    visualizes the results using the Elbow method plot, and presents the findings
    to the user. The optimal K is typically at the "elbow" point where adding more
    clusters yields diminishing returns in reducing inertia.
    
    Parameters:
        self: The parent application instance containing:
            - processed_data: The preprocessed dataset ready for clustering
            - clustering_class: The K-means implementation to use
            - visualizer: Component for creating visualizations
            - translator: Localization component for UI text
            - elbow_canvas: Canvas for displaying the elbow method plot
            - tabs: Tab widget to switch to the visualization
            - results_text: Text widget to display numerical results
    
    Returns:
        None: Results are displayed in the UI rather than returned.
    
    Raises:
        Exception: If any error occurs during the clustering process or visualization,
                  an error message is displayed to the user.
    """
    tr = self.translator.translate
    
    # Check if preprocessed data is available
    if self.processed_data is None:
        QMessageBox.warning(self, tr('msg_warning'), tr('msg_no_processed_data'))
        return
        
    try:

        # Create temporary model for finding optimal K
        temp_kmeans = self.clustering_class()
        
        # Compute inertia values for a range of K values (1-10)
        k_range, inertia_values = temp_kmeans.optimal_k_elbow(
            self.processed_data,
            k_range=list(range(1, 11))
        )
        
        # Visualize the elbow method results
        fig = self.visualizer.plot_elbow_method(
            k_range, 
            inertia_values,
            title=tr('plot_elbow_title'),
            x_label=tr('plot_elbow_x'),
            y_label=tr('plot_elbow_y')
        )
        self.elbow_canvas.figure = fig
        self.elbow_canvas.draw()
        
        # Switch to the elbow method tab to display results
        self.tabs.setCurrentIndex(1)
        
        # Create and format results text for detailed display
        info_text = f"{tr('optimal_k_results')}\n\n"
        info_text += f"{tr('metric_inertia')}:\n"
        
        for k, inertia in zip(k_range, inertia_values):
            info_text += f"k = {k}: {inertia:.2f}\n"
            
        # Display numerical results in the text field
        self.results_text.setText(info_text)
   
    except Exception as e:
        
        # Handle any errors and display them to the user
        error_msg = f"{tr('msg_error')}: {str(e)}"
        QMessageBox.critical(self, tr('msg_error'), error_msg)
        self.results_text.setText(error_msg)