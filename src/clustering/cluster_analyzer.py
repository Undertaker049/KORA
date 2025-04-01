"""
Module for analyzing clustering results.
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, List, Tuple, Dict
from sklearn.metrics import silhouette_samples
from collections import Counter

class ClusterAnalyzer:
    
    @staticmethod
    def get_cluster_sizes(labels: np.ndarray) -> Dict[int, int]:
        """
        Get the sizes of each cluster.
        
        Args:
            labels: Cluster labels
            
        Returns:
            Dictionary with the number of elements in each cluster
        """
        return dict(Counter(labels))
    
    @staticmethod
    def get_cluster_stats(data: np.ndarray, 
                        labels: np.ndarray, 
                        feature_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Get statistics for each cluster.
        
        Args:
            data: Original data
            labels: Cluster labels
            feature_names: Feature names
            
        Returns:
            DataFrame with statistics for each cluster
        """

        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(data.shape[1])]
            
        # Create DataFrame with data and cluster labels
        df = pd.DataFrame(data, columns=feature_names)
        df['cluster'] = labels
        
        # Calculate statistics for each cluster
        stats = []
        
        for cluster_id in sorted(df['cluster'].unique()):
            cluster_data = df[df['cluster'] == cluster_id]
            cluster_stats = {
                'cluster_id': cluster_id,
                'size': len(cluster_data),
                'percentage': len(cluster_data) / len(df) * 100
            }
            
            # Add statistics for each feature
            for feature in feature_names:
                feature_values = cluster_data[feature]
                cluster_stats[f"{feature}_mean"] = feature_values.mean()
                cluster_stats[f"{feature}_std"] = feature_values.std()
                cluster_stats[f"{feature}_min"] = feature_values.min()
                cluster_stats[f"{feature}_max"] = feature_values.max()
                
            stats.append(cluster_stats)
            
        return pd.DataFrame(stats)
    
    @staticmethod
    def get_silhouette_values(data: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Calculate silhouette values for each data point.
        
        Args:
            data: Original data
            labels: Cluster labels
            
        Returns:
            Array of silhouette values for each data point
        """

        # If only one cluster, return zero values
        if len(np.unique(labels)) <= 1:
            return np.zeros(len(data))
            
        return silhouette_samples(data, labels)
    
    @staticmethod
    def find_outliers(data: np.ndarray, 
                     labels: np.ndarray,
                     method: str = 'distance',
                     threshold: float = 2.0) -> List[int]:
        """
        Find outliers in clusters.
        
        Args:
            data: Original data
            labels: Cluster labels
            method: Outlier detection method ('distance', 'silhouette')
            threshold: Threshold for determining outliers
            
        Returns:
            List of outlier indices
        """
        outliers = []
        
        if method == 'distance':

            # Calculate centroids for each cluster
            centroids = {}

            for cluster_id in np.unique(labels):
                cluster_points = data[labels == cluster_id]
                centroids[cluster_id] = np.mean(cluster_points, axis=0)
                
            # Calculate distances to centroids
            for i, (point, label) in enumerate(zip(data, labels)):
                distance = np.linalg.norm(point - centroids[label])
                
                # Calculate standard deviation of distances for the cluster
                cluster_points = data[labels == label]
                cluster_centroid = centroids[label]
                distances = [np.linalg.norm(p - cluster_centroid) for p in cluster_points]
                std_distance = np.std(distances)
                
                # If distance is greater than threshold * standard deviations, consider point as outlier
                if distance > threshold * std_distance:
                    outliers.append(i)
                    
        elif method == 'silhouette':
            
            # Use silhouette coefficients to find outliers
            silhouette_values = ClusterAnalyzer.get_silhouette_values(data, labels)
            
            # Points with negative silhouette coefficient close to -1 may be outliers
            # threshold in this case should be a negative value (e.g., -0.3)
            outliers = [i for i, s in enumerate(silhouette_values) if s < threshold]
            
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")
            
        return outliers