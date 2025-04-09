"""
Module for saving data and clustering results.

Provides interface functionality for exporting clustering results.
"""

import numpy as np
import pandas as pd
from PyQt5.QtWidgets import QFileDialog, QMessageBox

def save_results(self):
    """
    Save clustering results to user-specified file.
    
    Parameters:
        self: Parent application with:
            - data: Original dataset
            - labels: Cluster assignments
            - reduced_data: Dimensionality-reduced data (optional)
            - original_columns: Column names
            - translator: Localization handler
            - results_text: Text display widget
            
    Returns:
        None: Results are saved to file; UI is updated with status
        
    Raises:
        Exception: Errors during file saving are caught and displayed
    """
    tr = self.translator
    
    if self.data is None or self.labels is None:
        QMessageBox.warning(self, tr('msg_warning'), tr('msg_no_data'))
        return
        
    try:
        options = QFileDialog.Options()
        file_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            tr('file_save_title'),
            "",
            tr('file_types'),
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
            file_type = "CSV"

        elif file_path.endswith('.xlsx') or selected_filter == "Excel files (*.xlsx)":

            if not file_path.endswith('.xlsx'):
                file_path += '.xlsx'

            results_df.to_excel(file_path, index=False)
            file_type = "Excel"

        elif file_path.endswith('.npy') or selected_filter == "NumPy files (*.npy)":

            if not file_path.endswith('.npy'):
                file_path += '.npy'

            # For NumPy, save only the array with cluster labels
            np.save(file_path, self.labels)
            file_type = "NumPy (only cluster labels)"

        else:

            # Default save as CSV
            if not file_path.endswith('.csv'):
                file_path += '.csv'

            results_df.to_csv(file_path, index=False)
            file_type = "CSV (default)"
        
        # Save the path for localization purposes
        self.last_save_path = file_path
        
        # Create result message
        info_text = f"{tr('msg_results_saved')}\n\n"
        info_text += f"{tr('file_save_title')}: {file_path}\n"
        info_text += f"{tr('data_preview')}: {file_type}\n"
        
        if file_type != "NumPy (only cluster labels)":
            info_text += f"{tr('data_features')}: {len(results_df.columns)}\n"
            info_text += f"{tr('data_samples')}: {len(results_df)}\n"
            
            # Add column names
            info_text += f"\n{tr('data_features')}: "
            column_names = results_df.columns.tolist()
            info_text += ", ".join(column_names[:10])
            
            if len(column_names) > 10:
                info_text += f" {tr('data_features')} {len(column_names) - 10}..."

        else:
            info_text += f"{tr('plot_cluster')}: {len(self.labels)}\n"
        
        # Display in text area
        self.results_text.setText(info_text)
   
    except Exception as e:
        error_msg = f"{tr('msg_error')}: {str(e)}"
        QMessageBox.critical(self, tr('msg_error'), error_msg)
        self.results_text.setText(error_msg)