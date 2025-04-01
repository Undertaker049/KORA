"""
Module for data preprocessing before clustering.
"""

import pandas as pd
import numpy as np
from typing import Optional, Union, List, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

class DataPreprocessor:
    
    def __init__(self):
        """Initialize the DataPreprocessor object."""
        self.scaler = None
        self.imputer = None
        self.pca = None
    
    def scale_data(self, 
                  data: Union[pd.DataFrame, np.ndarray], 
                  method: str = 'standard',
                  feature_range: Tuple[int, int] = (0, 1)) -> np.ndarray:
        """
        Scale data using the selected method.
        
        Args:
            data: Input data for scaling
            method: Scaling method ('standard', 'minmax', 'robust')
            feature_range: Range for MinMaxScaler
            
        Returns:
            Scaled data
        """
        
        if method == 'standard':
            self.scaler = StandardScaler()

        elif method == 'minmax':
            self.scaler = MinMaxScaler(feature_range=feature_range)

        elif method == 'robust':
            self.scaler = RobustScaler()

        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        # Convert DataFrame to numpy array if necessary
        if isinstance(data, pd.DataFrame):
            data = data.values
            
        return self.scaler.fit_transform(data)
    
    def handle_missing_values(self, 
                            data: Union[pd.DataFrame, np.ndarray],
                            strategy: str = 'mean') -> np.ndarray:
        """
        Handle missing values in data.
        
        Args:
            data: Input data with missing values
            strategy: Imputation strategy ('mean', 'median', 'most_frequent', 'constant')
            
        Returns:
            Data without missing values
        """
        self.imputer = SimpleImputer(strategy=strategy)
        
        # Convert DataFrame to numpy array if necessary
        if isinstance(data, pd.DataFrame):
            data = data.values
            
        return self.imputer.fit_transform(data)
    
    def reduce_dimensions(self, 
                         data: Union[pd.DataFrame, np.ndarray],
                         n_components: int = 2) -> np.ndarray:
        """
        Reduce data dimensionality using PCA.
        
        Args:
            data: High-dimensional input data
            n_components: Number of components to keep
            
        Returns:
            Reduced dimensionality data
        """
        self.pca = PCA(n_components=n_components)
        
        # Convert DataFrame to numpy array if necessary
        if isinstance(data, pd.DataFrame):
            data = data.values
            
        return self.pca.fit_transform(data)
    
    def preprocess_pipeline(self, 
                           data: Union[pd.DataFrame, np.ndarray],
                           scale: bool = True,
                           scaling_method: str = 'standard',
                           handle_missing: bool = True,
                           missing_strategy: str = 'mean',
                           reduce_dims: bool = False,
                           n_components: int = 2) -> np.ndarray:
        """
        Complete data preprocessing pipeline.
        
        Args:
            data: Original data for preprocessing
            scale: Flag to enable scaling
            scaling_method: Scaling method
            handle_missing: Flag to handle missing values
            missing_strategy: Missing value handling strategy
            reduce_dims: Flag to reduce dimensionality
            n_components: Number of components for dimensionality reduction
            
        Returns:
            Preprocessed data
        """
        processed_data = data.copy() if isinstance(data, pd.DataFrame) else data.copy()
        
        if handle_missing:
            processed_data = self.handle_missing_values(processed_data, strategy=missing_strategy)
            
        if scale:
            processed_data = self.scale_data(processed_data, method=scaling_method)
            
        if reduce_dims:
            processed_data = self.reduce_dimensions(processed_data, n_components=n_components)
            
        return processed_data