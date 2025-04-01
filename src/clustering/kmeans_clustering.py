"""
Module implementing the k-means clustering algorithm.
"""

import numpy as np
from typing import Optional, Union, List, Tuple
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

class KMeansClustering:
    
    def __init__(self, 
                n_clusters: int = 3, 
                max_iter: int = 300, 
                random_state: Optional[int] = None):
        """
        Initialize the KMeansClustering object.
        
        Args:
            n_clusters: Number of clusters
            max_iter: Maximum number of iterations
            random_state: Random state for reproducibility
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.model = KMeans(
            n_clusters=n_clusters,
            max_iter=max_iter,
            random_state=random_state
        )
        self.labels_ = None
        self.cluster_centers_ = None
        self.inertia_ = None
        
    def fit(self, X: np.ndarray) -> 'KMeansClustering':
        """
        Train the model on input data.
        
        Args:
            X: Input data for clustering
            
        Returns:
            self: Trained model
        """
        self.model.fit(X)
        self.labels_ = self.model.labels_
        self.cluster_centers_ = self.model.cluster_centers_
        self.inertia_ = self.model.inertia_
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict clusters for new data.
        
        Args:
            X: New data for clustering
            
        Returns:
            Cluster labels for input data
        """
        return self.model.predict(X)
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Combine fit and predict methods.
        
        Args:
            X: Input data for clustering
            
        Returns:
            Cluster labels for input data
        """
        self.fit(X)
        return self.labels_
    
    def evaluate(self, X: np.ndarray, labels: Optional[np.ndarray] = None) -> dict:
        """
        Evaluate clustering quality using various metrics.
        
        Args:
            X: Input data
            labels: Cluster labels (if None, self.labels_ is used)
            
        Returns:
            Dictionary with clustering quality metrics
        """

        if labels is None:

            if self.labels_ is None:
                raise ValueError("Model is not trained. Call fit() or fit_predict() method first.")
            
            labels = self.labels_
        
        metrics = {}
        
        # Inertia (sum of squared distances to the nearest cluster center)
        metrics['inertia'] = self.inertia_
        
        # Clustering quality evaluation (only if number of clusters > 1)
        if len(np.unique(labels)) > 1:

            # Silhouette Score (from -1 to 1, higher is better)
            metrics['silhouette_score'] = silhouette_score(X, labels)
            
            # Calinski-Harabasz Index (higher is better)
            metrics['calinski_harabasz_score'] = calinski_harabasz_score(X, labels)
            
            # Davies-Bouldin Index (lower is better)
            metrics['davies_bouldin_score'] = davies_bouldin_score(X, labels)
        
        return metrics
    
    def optimal_k_elbow(self, 
                      X: np.ndarray, 
                      k_range: List[int] = None) -> Tuple[List[int], List[float]]:
        """
        Find the optimal number of clusters using the elbow method.
        
        Args:
            X: Input data
            k_range: Range of k values to test
            
        Returns:
            Tuple of two lists: k values and corresponding inertia values
        """

        if k_range is None:
            k_range = list(range(1, 11))
        
        inertia_values = []
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, max_iter=self.max_iter, random_state=self.random_state)
            kmeans.fit(X)
            inertia_values.append(kmeans.inertia_)
            
        return k_range, inertia_values