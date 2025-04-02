"""
Test file generation.
"""

import os
import subprocess
import shutil
import sys
import numpy as np
import pandas as pd
from pathlib import Path

def ensure_directory_exists(directory_path):
    """
    Ensure that a directory exists, create it if necessary.
    
    Args:
        directory_path: Path to the directory
    """
    Path(directory_path).mkdir(parents=True, exist_ok=True)
    print(f"Directory prepared: {directory_path}")

def clean_directory(directory_path):
    """
    Clean the contents of a directory.
    
    Args:
        directory_path: Path to the directory
    """

    if not os.path.exists(directory_path):
        return
    
    for item in os.listdir(directory_path):
        item_path = os.path.join(directory_path, item)

        if os.path.isfile(item_path):
            os.unlink(item_path)

        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)
    
    print(f"Directory cleaned: {directory_path}")

def generate_blobs_data(n_samples=100, n_features=2, n_clusters=3, random_state=42):
    """
    Generate 'blobs' type data (groups of points).
    
    Args:
        n_samples: Number of points
        n_features: Number of features
        n_clusters: Number of clusters
        random_state: Parameter for reproducibility
        
    Returns:
        Tuple (X, y) with data and cluster labels
    """
    from sklearn.datasets import make_blobs
    
    X, y = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_clusters,
        random_state=random_state
    )
    
    return X, y

def generate_moons_data(n_samples=100, noise=0.1, random_state=42):
    """
    Generate 'moons' type data (half-moons).
    
    Args:
        n_samples: Number of points
        noise: Noise level
        random_state: Parameter for reproducibility
        
    Returns:
        Tuple (X, y) with data and cluster labels
    """
    from sklearn.datasets import make_moons
    
    X, y = make_moons(
        n_samples=n_samples,
        noise=noise,
        random_state=random_state
    )
    
    return X, y

def generate_circles_data(n_samples=100, noise=0.1, factor=0.5, random_state=42):
    """
    Generate 'circles' type data (concentric circles).
    
    Args:
        n_samples: Number of points
        noise: Noise level
        factor: Scale factor between circles
        random_state: Parameter for reproducibility
        
    Returns:
        Tuple (X, y) with data and cluster labels
    """
    from sklearn.datasets import make_circles
    
    X, y = make_circles(
        n_samples=n_samples,
        noise=noise,
        factor=factor,
        random_state=random_state
    )
    
    return X, y

def generate_random_data(n_samples=100, n_features=2, random_state=42):
    """
    Generate random data.
    
    Args:
        n_samples: Number of points
        n_features: Number of features
        random_state: Parameter for reproducibility
        
    Returns:
        Array X with data
    """
    np.random.seed(random_state)
    X = np.random.randn(n_samples, n_features)
    
    return X

def save_to_csv(X, y=None, output_file="test_data.csv"):
    """
    Save data to a CSV file.
    
    Args:
        X: Feature array
        y: Cluster labels array (optional)
        output_file: Path to save the file
    """

    # Create DataFrame with features
    columns = [f"feature_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=columns)
    
    # Add cluster labels if provided
    if y is not None:
        df["cluster"] = y
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Data saved to CSV: {output_file}")

def save_to_excel(X, y=None, output_file="test_data.xlsx", sheet_name="Data"):
    """
    Save data to an Excel file.
    
    Args:
        X: Feature array
        y: Cluster labels array (optional)
        output_file: Path to save the file
        sheet_name: Name of the sheet in the Excel file
    """

    # Create DataFrame with features
    columns = [f"feature_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=columns)
    
    # Add cluster labels if provided
    if y is not None:
        df["cluster"] = y
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    
    # Save to Excel
    df.to_excel(output_file, sheet_name=sheet_name, index=False)
    print(f"Data saved to Excel: {output_file}")

def save_to_numpy(X, y=None, output_file="test_data.npy", labels_file=None):
    """
    Save data to NumPy file.
    
    Args:
        X: Feature array
        y: Cluster labels array (optional)
        output_file: Path to save the features file
        labels_file: Path to save the labels file (optional)
    """

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    
    # Save features
    np.save(output_file, X)
    print(f"Features saved to NumPy: {output_file}")
    
    # Save labels if provided
    if y is not None:

        if labels_file is None:
            # Create labels filename based on features filename
            base, ext = os.path.splitext(output_file)
            labels_file = f"{base}_labels{ext}"
        
        np.save(labels_file, y)
        print(f"Labels saved to NumPy: {labels_file}")

def generate_all_test_files():
    """
    Generate all types of test files.
    """

    # Define paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    files_dir = os.path.join(base_dir, 'files')
    
    # Create and clean directory for test files
    ensure_directory_exists(files_dir)
    clean_directory(files_dir)
    
    print("Starting test files generation...")
    
    # Generate standard blobs data for CSV
    print("\nGenerating CSV files...")
    X, y = generate_blobs_data(n_samples=200, n_features=5, n_clusters=3)
    save_to_csv(X, y, os.path.join(files_dir, "blobs_standard.csv"))
    
    # Blobs without labels
    save_to_csv(X, None, os.path.join(files_dir, "blobs_unlabeled.csv"))
    
    # Moons data for CSV
    X, y = generate_moons_data(n_samples=200, noise=0.1)
    save_to_csv(X, y, os.path.join(files_dir, "moons.csv"))
    
    # Circles data for CSV
    X, y = generate_circles_data(n_samples=200, noise=0.05)
    save_to_csv(X, y, os.path.join(files_dir, "circles.csv"))
    
    # Generate Excel files
    print("\nGenerating Excel files...")
    X, y = generate_blobs_data(n_samples=200, n_features=5, n_clusters=3)
    save_to_excel(X, y, os.path.join(files_dir, "blobs_standard.xlsx"), "ClusterData")
    
    # Random data for Excel
    X = generate_random_data(n_samples=200, n_features=5)
    save_to_excel(X, None, os.path.join(files_dir, "random_data.xlsx"), "RandomData")
    
    # Generate NumPy files
    print("\nGenerating NumPy files...")
    X, y = generate_blobs_data(n_samples=200, n_features=5, n_clusters=3)
    save_to_numpy(X, y, os.path.join(files_dir, "blobs_standard.npy"))
    
    # High-dimensional data
    X, y = generate_blobs_data(n_samples=200, n_features=20, n_clusters=5)
    save_to_numpy(X, y, os.path.join(files_dir, "blobs_highdim.npy"))
    
    print("\nTest file generation completed")
    print(f"Total files created: {len(os.listdir(files_dir))}")
    
    return files_dir

if __name__ == "__main__":
    generate_all_test_files()