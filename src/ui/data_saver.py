"""
Module for saving data and clustering results.
"""

import numpy as np
import pandas as pd
from PyQt5.QtWidgets import QFileDialog, QMessageBox

def save_results(self):
    """
    Save clustering results to a file.
    """
    
    if self.data is None or self.labels is None:
        QMessageBox.warning(self, "Warning", "No data to save. Perform clustering first.")
        return
        
    try:
        options = QFileDialog.Options()
        file_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save Clustering Results",
            "",
            "CSV files (*.csv);;Excel files (*.xlsx);;NumPy files (*.npy)",
            options=options
        )
        
        if not file_path:
            return
            
        # Create DataFrame with original data and cluster labels
        if self.original_columns is None:
            self.original_columns = [f"Feature_{i}" for i in range(self.data.shape[1])]
            
        results_df = pd.DataFrame(self.data, columns=self.original_columns)
        results_df['cluster'] = self.labels
        
        # Add coordinates in reduced dimensionality if available
        if self.reduced_data is not None and self.reduced_data.shape[1] == 2:
            results_df['x_reduced'] = self.reduced_data[:, 0]
            results_df['y_reduced'] = self.reduced_data[:, 1]
        
        # Save according to selected format
        if file_path.endswith('.csv') or selected_filter == "CSV files (*.csv)":

            if not file_path.endswith('.csv'):
                file_path += '.csv'

            results_df.to_csv(file_path, index=False)

        elif file_path.endswith('.xlsx') or selected_filter == "Excel files (*.xlsx)":

            if not file_path.endswith('.xlsx'):
                file_path += '.xlsx'

            results_df.to_excel(file_path, index=False)

        elif file_path.endswith('.npy') or selected_filter == "NumPy files (*.npy)":

            if not file_path.endswith('.npy'):
                file_path += '.npy'

            # For NumPy, save only the array with cluster labels
            np.save(file_path, self.labels)

        else:

            # Default save as CSV
            if not file_path.endswith('.csv'):
                file_path += '.csv'

            results_df.to_csv(file_path, index=False)
            
        QMessageBox.information(self, "Success", f"Clustering results successfully saved to\n{file_path}")
   
    except Exception as e:
        QMessageBox.critical(self, "Error", f"Error saving results: {str(e)}")