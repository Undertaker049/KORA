"""
Main module for launching the application.
"""

import sys
import os

# Add the project root directory to the import path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.interface.app import main

if __name__ == '__main__':
    main()