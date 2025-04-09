"""
Main module for launching the clustering application.
"""

import sys
import os
import argparse
from PyQt5.QtWidgets import QApplication
from src.data_processing import DataLoader, DataPreprocessor
from src.clustering import KMeansClustering, ClusterAnalyzer
from src.visualization import ClusterVisualizer
from src.ui import create_app
from src.localization import get_translator, set_language

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def main():
    """
    Application entry point.
    
    Creates and launches the clustering application.
    """

    # Get translator instance
    translator = get_translator()
    
    # Get available languages
    available_languages = translator.available_languages
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='KORA - K-means Optimization and Recursive Analysis Framework')
    parser.add_argument(
        '--lang', 
        choices=available_languages, 
        default='ru',
        help=f'Application language ({"/".join(available_languages)}), default: ru'
    )
    args = parser.parse_args()
    
    # Set language from command line argument
    set_language(args.lang)
    
    # Update translator with the selected language
    translator = get_translator()
    
    # Initialize application
    app_instance = QApplication(sys.argv)
    clustering_app = create_app(
        DataLoader,
        DataPreprocessor,
        KMeansClustering,
        ClusterAnalyzer,
        ClusterVisualizer,
        translator
    )
    
    # Show the application window
    clustering_app.show()
    
    # Start application event loop
    sys.exit(app_instance.exec_())

if __name__ == '__main__':
    main()