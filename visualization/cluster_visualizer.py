"""
Visualization module for clustering results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Union, List, Tuple, Dict
from matplotlib.figure import Figure
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

class ClusterVisualizer:
    
    def __init__(self, figsize: Tuple[int, int] = (10, 8), style: str = 'whitegrid'):
        """
        Initialize the ClusterVisualizer object.
        
        Args:
            figsize: Figure size for visualization
            style: Seaborn style for visualization
        """
        self.figsize = figsize
        sns.set_style(style)
        
    def plot_clusters_2d(self, 
                       data: np.ndarray, 
                       labels: np.ndarray,
                       centers: Optional[np.ndarray] = None,
                       title: str = "Clustering Results",
                       x_label: str = "Component 1",
                       y_label: str = "Component 2",
                       alpha: float = 0.7,
                       s: int = 50) -> Figure:
        """
        Visualize clusters in two-dimensional space.
        
        Args:
            data: Data for visualization (must be 2D)
            labels: Cluster labels
            centers: Cluster centers (optional)
            title: Plot title
            x_label: X-axis label
            y_label: Y-axis label
            alpha: Point transparency
            s: Point size
            
        Returns:
            Matplotlib Figure with visualization
        """

        if data.shape[1] != 2:
            raise ValueError("Data must be two-dimensional for visualization. Use the reduce_dimensions method.")
            
        # Create dataframe for seaborn
        df = pd.DataFrame({
            'x': data[:, 0],
            'y': data[:, 1],
            'Cluster': labels.astype(str)
        })
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Build scatter plot using seaborn
        sns.scatterplot(
            data=df,
            x='x', y='y',
            hue='Cluster',
            palette='viridis',
            alpha=alpha,
            s=s,
            ax=ax
        )
        
        # Add cluster centers if provided
        if centers is not None:
            plt.scatter(
                centers[:, 0], centers[:, 1],
                marker='X',
                c='red',
                s=200,
                alpha=1,
                label='Cluster Centers'
            )
            plt.legend()
            
        # Configure plot
        plt.title(title, fontsize=15)
        plt.xlabel(x_label, fontsize=12)
        plt.ylabel(y_label, fontsize=12)
        plt.tight_layout()
        
        return fig
    
    def reduce_dimensions(self, 
                         data: np.ndarray, 
                         method: str = 'pca',
                         n_components: int = 2,
                         random_state: Optional[int] = 42) -> np.ndarray:
        """
        Reduce data dimensionality for visualization.
        
        Args:
            data: High-dimensional input data
            method: Dimensionality reduction method ('pca' or 'tsne')
            n_components: Number of components to retain
            random_state: Random state for reproducibility
            
        Returns:
            Reduced-dimensionality data
        """

        if method == 'pca':
            reducer = PCA(n_components=n_components, random_state=random_state)
        
        elif method == 'tsne':
            reducer = TSNE(n_components=n_components, random_state=random_state)
       
        else:
            raise ValueError(f"Unknown dimensionality reduction method: {method}")
            
        return reducer.fit_transform(data)
    
    def plot_elbow_method(self,
                         k_values: List[int],
                         inertia_values: List[float],
                         title: str = "Elbow Method for Determining Optimal Number of Clusters",
                         x_label: str = "Number of clusters (k)",
                         y_label: str = "Inertia") -> Figure:
        """
        Visualize the elbow method for determining the optimal number of clusters.
        
        Args:
            k_values: List of k values
            inertia_values: List of inertia values for each k
            title: Plot title
            x_label: X-axis label
            y_label: Y-axis label
            
        Returns:
            Matplotlib Figure with elbow method visualization
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Build line graph
        plt.plot(k_values, inertia_values, 'bo-')
        plt.grid(True)
        
        # Add data points
        for k, inertia in zip(k_values, inertia_values):
            plt.annotate(
                f'k={k}',
                xy=(k, inertia),
                xytext=(5, 0),
                textcoords='offset points',
                fontsize=10
            )
            
        # Configure plot
        plt.title(title, fontsize=15)
        plt.xlabel(x_label, fontsize=12)
        plt.ylabel(y_label, fontsize=12)
        plt.xticks(k_values)
        plt.tight_layout()
        
        return fig
    
    def plot_silhouette(self,
                       data: np.ndarray,
                       labels: np.ndarray,
                       title: str = "Silhouette Coefficient Analysis",
                       figsize: Optional[Tuple[int, int]] = None) -> Figure:
        """
        Visualize silhouette coefficients to evaluate clustering quality.
        
        Args:
            data: Input data
            labels: Cluster labels
            title: Plot title
            figsize: Figure size (optional)
            
        Returns:
            Matplotlib Figure with silhouette coefficient visualization
        """
        from sklearn.metrics import silhouette_samples, silhouette_score
        
        # If only one cluster, silhouette coefficients cannot be computed
        if len(np.unique(labels)) <= 1:
            raise ValueError("More than one cluster is required to compute silhouette coefficients.")
            
        # Calculate silhouette coefficients
        silhouette_avg = silhouette_score(data, labels)
        sample_silhouette_values = silhouette_samples(data, labels)
        
        # Create figure
        if figsize is None:
            figsize = self.figsize
            
        fig, ax = plt.subplots(figsize=figsize)
        
        y_lower = 10
        n_clusters = len(np.unique(labels))
        
        # Color palette
        cmap = plt.cm.get_cmap("viridis", n_clusters)
        
        # For each cluster, build a silhouette plot
        for i in range(n_clusters):
            # Get silhouette values for i-th cluster
            ith_cluster_values = sample_silhouette_values[labels == i]
            ith_cluster_values.sort()
            
            size_cluster_i = ith_cluster_values.shape[0]
            y_upper = y_lower + size_cluster_i
            
            color = cmap(i / n_clusters)
            ax.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7
            )
            
            # Add cluster label
            ax.text(
                -0.05,
                y_lower + 0.5 * size_cluster_i,
                str(i)
            )
            
            y_lower = y_upper + 10
            
        # Configure plot
        ax.set_title(title, fontsize=15)
        ax.set_xlabel("Silhouette Coefficients", fontsize=12)
        ax.set_ylabel("Cluster", fontsize=12)
        
        # Add vertical line for average silhouette coefficient
        ax.axvline(x=silhouette_avg, color="red", linestyle="--")
        ax.text(
            silhouette_avg + 0.02,
            plt.ylim()[0] + 0.7 * (plt.ylim()[1] - plt.ylim()[0]),
            f"Average silhouette coefficient: {silhouette_avg:.3f}",
            color="red"
        )
        
        ax.set_yticks([])  # Hide Y-axis ticks
        ax.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
        plt.tight_layout()
        
        return fig
    
    def plot_feature_importance(self,
                              cluster_centers: np.ndarray,
                              feature_names: List[str],
                              title: str = "Feature Importance for Clusters",
                              figsize: Optional[Tuple[int, int]] = None) -> Figure:
        """
        Visualize feature importance for clusters based on cluster centers.
        
        Args:
            cluster_centers: Cluster centers
            feature_names: Feature names
            title: Plot title
            figsize: Figure size (optional)
            
        Returns:
            Matplotlib Figure with feature importance visualization
        """
        if figsize is None:
            figsize = (self.figsize[0], self.figsize[1] * 0.8)
            
        n_clusters = cluster_centers.shape[0]
        n_features = cluster_centers.shape[1]
        
        if len(feature_names) != n_features:
            raise ValueError(f"Number of feature names ({len(feature_names)}) does not match number of features in cluster centers ({n_features}).")
            
        # Create dataframe with cluster centers
        df = pd.DataFrame(cluster_centers, columns=feature_names)
        df['Cluster'] = [f"Cluster {i}" for i in range(n_clusters)]
        
        # Transform dataframe to "long" format for seaborn
        df_melted = pd.melt(df, id_vars=['Cluster'], var_name='Feature', value_name='Value')
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Build heatmap to visualize cluster centers
        pivoted = df_melted.pivot(index='Cluster', columns='Feature', values='Value')
        sns.heatmap(pivoted, annot=True, cmap='viridis', linewidths=0.5, ax=ax)
        
        # Configure plot
        plt.title(title, fontsize=15)
        plt.tight_layout()
        
        return fig