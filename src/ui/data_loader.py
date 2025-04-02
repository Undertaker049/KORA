"""
Module for loading data from various sources.
"""

import pandas as pd
from PyQt5.QtWidgets import QFileDialog, QMessageBox

def load_data_from_file(self):
    """
    Load data from a file.
    """
    options = QFileDialog.Options()
    file_path, _ = QFileDialog.getOpenFileName(
        self,
        "Load Data",
        "",
        "CSV files (*.csv);;Excel files (*.xlsx *.xls);;NumPy files (*.npy);;All files (*)",
        options=options
    )
    
    if not file_path:
        return
        
    try:
        
        if file_path.endswith('.csv'):
            data_df = self.data_loader.load_csv(file_path)
            self.original_columns = data_df.columns.tolist()
            self.data = data_df.values
            file_type = "CSV"

        elif file_path.endswith(('.xlsx', '.xls')):
            data_df = self.data_loader.load_excel(file_path)
            self.original_columns = data_df.columns.tolist()
            self.data = data_df.values
            file_type = "Excel"

        elif file_path.endswith('.npy'):
            self.data = self.data_loader.load_numpy(file_path)
            self.original_columns = [f"Feature_{i}" for i in range(self.data.shape[1])]
            file_type = "NumPy"
        
        else:
            self.data = self.data_loader.load_csv(file_path)
            file_type = "Generic"

            if isinstance(self.data, pd.DataFrame):
                self.original_columns = self.data.columns.tolist()
                self.data = self.data.values

            else:
                self.original_columns = [f"Feature_{i}" for i in range(self.data.shape[1])]
            
        self.update_data_info()
        
        # Create results text
        n_samples, n_features = self.data.shape
        info_text = "Data Successfully Loaded\n\n"
        info_text += f"File: {file_path}\n"
        info_text += f"Type: {file_type}\n"
        info_text += f"Samples: {n_samples}\n"
        info_text += f"Features: {n_features}\n"
        
        if hasattr(self, 'original_columns') and self.original_columns:
            info_text += f"\nFeature names: {', '.join(self.original_columns[:10])}"
            
            if len(self.original_columns) > 10:
                info_text += f" and {len(self.original_columns) - 10} more..."
        
        # Display in text area
        self.results_text.setText(info_text)

    except Exception as e:
        error_msg = f"Error loading data: {str(e)}"
        QMessageBox.critical(self, "Error", error_msg)
        self.results_text.setText(error_msg)