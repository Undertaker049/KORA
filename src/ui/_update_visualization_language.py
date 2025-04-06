"""
Module for updating visualization language settings.

Provides functionality for dynamically updating all visualization
elements when the application language is changed.
"""

import traceback

def update_visualization_language(self):
    """
    Update all visualization elements to reflect the current language setting.
    
    Parameters:
        self: The parent application instance containing:
            - reduced_data: The dimensionality-reduced dataset
            - labels: Cluster assignments for data points
            - clusters_canvas: Canvas for cluster visualization
            - elbow_canvas: Canvas for elbow method plot
            - silhouette_canvas: Canvas for silhouette analysis
            - features_canvas: Canvas for feature importance
            - translator: Object providing translation functionality
            - elbow_k_range, elbow_curve: Data for elbow method plot
            
    Returns:
        None: Visualizations are updated in-place
    """

    # Ensure all required data and canvases exist
    required_attrs = ['reduced_data', 'labels', 'clusters_canvas']
    if not all(hasattr(self, attr) for attr in required_attrs):
        print("Skipping visualization update - missing required attributes")
        return
        
    # Check if data is available and not empty
    if self.reduced_data is None or self.labels is None:
        print("Skipping visualization update - no data available")
        return
        
    if len(self.reduced_data) == 0 or len(self.labels) == 0:
        print("Skipping visualization update - empty data")
        return
    
    # Check if canvases are properly initialized
    for canvas_name in ['clusters_canvas', 'elbow_canvas', 'silhouette_canvas', 'features_canvas']:
        
        if hasattr(self, canvas_name):
            canvas = getattr(self, canvas_name)
            
            if not hasattr(canvas, 'axes') or canvas.axes is None:
                print(f"Skipping visualization update - {canvas_name} is not properly initialized")
                return
    
    # Update all visualizations when language changes
    try:

        # Update cluster visualization
        try:

            # Call function that completely rebuilds the plot
            from .results_visualizer import update_cluster_visualization
            update_cluster_visualization(self)

        except Exception as e:
            print(f"Error updating cluster visualization: {str(e)}")
            traceback.print_exc()
        
        # Update elbow method plot if data is available
        if hasattr(self, 'elbow_k_range') and hasattr(self, 'elbow_curve') and \
           self.elbow_k_range is not None and self.elbow_curve is not None:
            
            try:
                import matplotlib.pyplot as plt
                
                # Save reference to current figure and canvas
                current_canvas = self.elbow_canvas
                current_figure = current_canvas.figure
                
                # Create new figure with the same size
                fig_size = current_figure.get_size_inches()
                new_figure = plt.figure(figsize=fig_size)
                
                # Create new axes on the new figure
                new_axes = new_figure.add_subplot(111)
                
                tr = self.translator.translate  # Use translate method instead of object
                
                # Plot elbow method on new axes
                new_axes.plot(self.elbow_k_range, self.elbow_curve, 'bo-')
                new_axes.set_xlabel(tr('plot_elbow_x'))
                new_axes.set_ylabel(tr('plot_elbow_y'))
                new_axes.set_title(tr('plot_elbow_title'))
                
                # Add labels to points
                for k, inertia in zip(self.elbow_k_range, self.elbow_curve):
                    new_axes.annotate(
                        f'k={k}',
                        xy=(k, inertia),
                        xytext=(5, 0),
                        textcoords='offset points',
                        fontsize=10
                    )
                
                # Set integer ticks on x-axis
                new_axes.set_xticks(self.elbow_k_range)
                
                # Apply tight_layout for optimal space usage
                new_figure.tight_layout()
                
                # Update canvas with new figure
                current_canvas.figure = new_figure
                current_canvas.draw()
                
                # Close old figure to free resources
                plt.close(current_figure)

            except Exception as e:
                print(f"Error updating elbow plot: {str(e)}")
                traceback.print_exc()
        
        # Update other plots
        try:

            # Call function that rebuilds silhouette and feature importance plots
            from .results_visualizer import update_tabs_visualization
            update_tabs_visualization(self)

        except Exception as e:
            print(f"Error updating visualization tabs: {str(e)}")
            traceback.print_exc()
        
    except Exception as e:
        print(f"Error in visualization language update: {str(e)}")
        traceback.print_exc()