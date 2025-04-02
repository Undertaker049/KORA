"""
Module for loading data from various sources.
"""

import pandas as pd
import numpy as np
from typing import Optional, Union, List, Tuple

class DataLoader:
    
    @staticmethod
    def load_csv(file_path: str, **kwargs) -> pd.DataFrame:
        """
        Load data from a CSV file.
        
        Args:
            file_path: Path to the CSV file
            **kwargs: Additional arguments for pd.read_csv
            
        Returns:
            DataFrame with loaded data
        """
        
        try:
            return pd.read_csv(file_path, **kwargs)
        
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            raise
    
    @staticmethod
    def load_excel(file_path: str, **kwargs) -> pd.DataFrame:
        """
        Load data from an Excel file.
        
        Args:
            file_path: Path to the Excel file
            **kwargs: Additional arguments for pd.read_excel
            
        Returns:
            DataFrame with loaded data
        """

        try:
            return pd.read_excel(file_path, **kwargs)
        
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            raise
    
    @staticmethod
    def load_numpy(file_path: str) -> np.ndarray:
        """
        Load data from a NumPy file.
        
        Args:
            file_path: Path to the .npy file
            
        Returns:
            NumPy array with loaded data
        """

        try:
            return np.load(file_path)
        
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            raise