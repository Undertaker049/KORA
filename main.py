"""
Main module for launching the clustering application.
"""

import sys
import os
from PyQt5.QtWidgets import QApplication
from src.data_processing import DataLoader, DataPreprocessor
from src.clustering import KMeansClustering, ClusterAnalyzer
from src.visualization import ClusterVisualizer
from src.ui import create_app

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def main():
    """
    Application entry point.
    
    Creates and launches the clustering application.
    """
    app_instance = QApplication(sys.argv)
    clustering_app = create_app(
        DataLoader,
        DataPreprocessor,
        KMeansClustering,
        ClusterAnalyzer,
        ClusterVisualizer
    )
    sys.exit(app_instance.exec_())

if __name__ == '__main__':
    main()