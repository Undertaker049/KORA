"""
Module for visualizing clustering results.

Provides functionality for creating and updating visualization
of clustering results.
"""

import numpy as np
from PyQt5.QtWidgets import QMessageBox
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def visualize_results(self):
    """
    Visualize clustering results.
    
    Creates or updates plots for cluster distribution, elbow method curve,
    silhouette analysis, and feature importance based on available data.
    
    Parameters:
        self: Parent application with visualization components and data:
            - reduced_data: Dimensionally reduced dataset for visualization
            - labels: Cluster assignments for each data point
            - translator: Localization handler
            - tabs: Tab widget for visualization panels
    """
    tr = self.translator.translate
    
    # Check if data exists for visualization
    if not hasattr(self, 'reduced_data') or self.reduced_data is None or not hasattr(self, 'labels') or self.labels is None:
        
        # Show error message
        QMessageBox.warning(
            self,
            tr('error_title'),
            tr('error_no_clustering_results'),
            QMessageBox.Ok
        )

        return
    
    # Check if data is empty
    if len(self.reduced_data) == 0 or len(self.labels) == 0:
        QMessageBox.warning(
            self,
            tr('error_title'),
            tr('error_empty_data'),
            QMessageBox.Ok
        )

        return
        
    try:

        # Switch to visualization tab if it exists
        if hasattr(self, 'tabs') and self.tabs.count() > 2:
            self.tabs.setCurrentIndex(2)
        
        # Reset existing plots
        if hasattr(self, 'clusters_canvas') and self.clusters_canvas:

            # Clear cluster plot before new visualization
            self.clusters_canvas.axes.clear()
            
            # Remove colorbar if it exists
            if hasattr(self, 'clusters_canvas') and self.clusters_canvas and hasattr(self, 'clusters_colorbar'):
                
                if self.clusters_colorbar is not None:
                    self.clusters_colorbar.remove()
                    self.clusters_colorbar = None
        
        # Update cluster visualization
        update_cluster_visualization(self)
        
        # Update elbow method plot if data is available
        if hasattr(self, 'elbow_k_range') and hasattr(self, 'elbow_curve') and \
           self.elbow_k_range is not None and self.elbow_curve is not None:
            
            try:

                # Clear plot before new visualization
                if hasattr(self, 'elbow_canvas') and self.elbow_canvas:
                    self.elbow_canvas.axes.clear()
                
                # Get axes for elbow method plot
                ax = self.elbow_canvas.axes
                
                # Plot elbow method
                ax.plot(self.elbow_k_range, self.elbow_curve, 'bo-')
                ax.set_xlabel(tr('plot_elbow_x'))
                ax.set_ylabel(tr('plot_elbow_y'))
                ax.set_title(tr('plot_elbow_title'))
                
                # Add labels to points
                for k, inertia in zip(self.elbow_k_range, self.elbow_curve):
                    ax.annotate(
                        f'k={k}',
                        xy=(k, inertia),
                        xytext=(5, 0),
                        textcoords='offset points',
                        fontsize=10
                    )
                
                # Set integer ticks on X axis
                ax.set_xticks(self.elbow_k_range)
                
                # Update plot
                self.elbow_canvas.draw()

            except Exception as e:
                print(f"Error updating elbow plot: {str(e)}")
        
        # Update other plots
        update_tabs_visualization(self)
        
        # Switch to clusters tab
        if hasattr(self, 'tabs'):
            self.tabs.setCurrentIndex(0)
        
    except Exception as e:
        print(f"Error in visualization: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Show error message
        QMessageBox.warning(
            self,
            tr('error_title'),
            tr('error_visualization') + f": {str(e)}",
            QMessageBox.Ok
        )

def update_cluster_visualization(self):
    """
    Update cluster visualization.
    
    Creates a new figure showing the cluster distribution
    with different colors for each cluster and centroids if available.
    
    Parameters:
        self: Parent application with:
            - clusters_canvas: Canvas for cluster visualization
            - reduced_data: 2D data for visualization
            - labels: Cluster assignments
            - translator: Localization handler
            - kmeans: Clustering model with centroids (optional)
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        
        tr = self.translator.translate
        
        # Create new figure and axes, completely replacing the old one
        current_canvas = self.clusters_canvas
        current_figure = current_canvas.figure
        
        # Create new figure with same size
        fig_size = current_figure.get_size_inches()
        new_figure = plt.figure(figsize=fig_size)
        
        # Create new axes on new figure
        new_axes = new_figure.add_subplot(111)
        
        # Create color map and scatter points by class
        unique_labels = sorted(set(self.labels))
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
        cmap = ListedColormap(colors)
        
        # Plot scatter graph
        scatter = new_axes.scatter(
            self.reduced_data[:, 0],
            self.reduced_data[:, 1],
            c=self.labels,
            cmap=cmap,
            s=30,
            alpha=0.8
        )
        
        # Add colorbar to the right of the plot
        cbar = new_figure.colorbar(scatter, ax=new_axes)
        cbar.set_label(tr('plot_cluster'))
        
        # Configure axes
        new_axes.set_xlabel(tr('plot_component1'))
        new_axes.set_ylabel(tr('plot_component2'))
        new_axes.set_title(tr('plot_cluster_distribution'))
        
        # Add centroids if available
        if hasattr(self, 'kmeans') and self.kmeans is not None and hasattr(self.kmeans, 'cluster_centers_'):
            
            try:
                # Transform centroids to the same space as reduced_data
                reduced_centers = None
                
                # If we have a dimensionality reducer and it was trained
                if hasattr(self, 'dimensionality_reducer') and self.dimensionality_reducer is not None:
                    
                    # Check that centroids have the same dimension as original data
                    if (hasattr(self, 'processed_data') and self.processed_data is not None and 
                        self.kmeans.cluster_centers_.shape[1] == self.processed_data.shape[1]):
                        
                        # Transform centroids using the same dimensionality reducer
                        reduced_centers = self.dimensionality_reducer.transform(self.kmeans.cluster_centers_)
                        
                        # Add centroids to plot
                        new_axes.scatter(
                            reduced_centers[:, 0], 
                            reduced_centers[:, 1],
                            s=150, 
                            marker='X',
                            c=range(len(reduced_centers)),
                            cmap=cmap,
                            edgecolors='k',
                            linewidths=1.5
                        )

            except Exception as e:
                print(f"Error plotting cluster centers: {str(e)}")
        
        # Apply tight_layout for optimal space usage
        new_figure.tight_layout()
        
        # Update canvas with new figure
        current_canvas.figure = new_figure
        current_canvas.draw()
        
        # Close old figure to free resources
        plt.close(current_figure)
        
    except Exception as e:
        print(f"Detailed error in cluster visualization: {str(e)}")
        import traceback
        traceback.print_exc()

def update_tabs_visualization(self):
    """
    Update silhouette and feature importance visualizations.
    
    Creates plots for silhouette analysis and feature importance
    if the corresponding data is available.
    
    Parameters:
        self: Parent application with visualization canvases and data:
            - silhouette_canvas: Canvas for silhouette plot
            - features_canvas: Canvas for feature importance plot
            - silhouette_values: Silhouette scores for each sample
            - labels: Cluster assignments
            - feature_importance: Importance scores for features
            - translator: Localization handler
    """
    tr = self.translator.translate
    import matplotlib.pyplot as plt
    
    # Update silhouette coefficients plot if data is available
    if hasattr(self, 'silhouette_values') and self.silhouette_values is not None:
        
        # Check that we have necessary data for plotting
        if len(self.silhouette_values) > 0 and self.labels is not None:
            
            try:
                
                # Create new figure and axes, completely replacing the old one
                current_canvas = self.silhouette_canvas
                current_figure = current_canvas.figure
                
                # Create new figure with same size
                fig_size = current_figure.get_size_inches()
                new_figure = plt.figure(figsize=fig_size)
                
                # Create new axes on new figure
                new_axes = new_figure.add_subplot(111)
                
                y_lower = 10
                
                # Get unique clusters
                unique_clusters = np.unique(self.labels)
                
                # Plot silhouette coefficients for each cluster
                for i in unique_clusters:

                    # Get silhouette coefficients for current cluster
                    cluster_silhouette_values = self.silhouette_values[self.labels == i]
                    
                    # Skip if empty cluster
                    if len(cluster_silhouette_values) == 0:
                        continue
                        
                    cluster_silhouette_values.sort()
                    
                    size_cluster_i = len(cluster_silhouette_values)
                    y_upper = y_lower + size_cluster_i
                    
                    color = plt.cm.viridis(float(i) / len(unique_clusters))
                    new_axes.fill_betweenx(
                        np.arange(y_lower, y_upper),
                        0, cluster_silhouette_values,
                        facecolor=color, edgecolor=color, alpha=0.7
                    )
                    
                    # Add cluster labels
                    new_axes.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
                    
                    # Update y_lower for next cluster
                    y_lower = y_upper + 10
                
                # Configure plot
                new_axes.set_xlabel(tr('plot_silhouette_x'))
                new_axes.set_ylabel(tr('plot_silhouette_y'))
                new_axes.set_title(tr('plot_silhouette_title'))
                
                # Vertical line for average silhouette coefficient
                if np.mean(self.silhouette_values) != 0:
                    new_axes.axvline(
                        x=np.mean(self.silhouette_values),
                        color="red",
                        linestyle="--"
                    )
                
                new_axes.set_yticks([])
                
                # Apply tight_layout for optimal space usage
                new_figure.tight_layout()
                
                # Update canvas with new figure
                current_canvas.figure = new_figure
                current_canvas.draw()
                
                # Close old figure to free resources
                plt.close(current_figure)

            except Exception as e:
                print(f"Error updating silhouette plot: {str(e)}")
                import traceback
                traceback.print_exc()
    
    # Update feature importance plot if data is available
    if hasattr(self, 'feature_importance') and self.feature_importance is not None:

        try:

            # Create new figure and axes, completely replacing the old one
            current_canvas = self.features_canvas
            current_figure = current_canvas.figure
            
            # Create new figure with same size
            fig_size = current_figure.get_size_inches()
            new_figure = plt.figure(figsize=fig_size)
            
            # Create new axes on new figure
            new_axes = new_figure.add_subplot(111)
            
            # Sort features by importance
            indices = np.argsort(self.feature_importance)[::-1]
            
            # Get feature names if available, else use indices
            if hasattr(self, 'original_columns') and self.original_columns is not None:
                feature_names = [self.original_columns[i] for i in indices]
                
            else:
                feature_names = [f"Feature {i}" for i in indices]
            
            # Plot bar chart
            new_axes.bar(
                range(len(self.feature_importance)),
                self.feature_importance[indices],
                align='center'
            )
            
            # Set labels on X axis if not too many
            if len(feature_names) < 30:  # Limit to prevent overlap
                new_axes.set_xticks(range(len(self.feature_importance)))
                new_axes.set_xticklabels(feature_names, rotation=90)
            
            # Configure plot
            new_axes.set_xlabel(tr('data_features'))
            new_axes.set_ylabel(tr('clustering_metrics'))
            new_axes.set_title(tr('data_features'))
            
            # Apply tight_layout for optimal space usage
            new_figure.tight_layout()
            
            # Update canvas with new figure
            current_canvas.figure = new_figure
            current_canvas.draw()
            
            # Close old figure to free resources
            plt.close(current_figure)
        
        except Exception as e:
                print(f"Error updating feature importance plot: {str(e)}")
                import traceback
                traceback.print_exc()