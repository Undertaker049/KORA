"""
Main module for launching the clustering application.
"""

import sys
import os
from src.ui import main

# Add the project root directory to the import path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

if __name__ == '__main__':
    main()