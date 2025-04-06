"""
Module for loading data.

Provides interface functionality for loading data from different file formats.
"""

import pandas as pd
from PyQt5.QtWidgets import QFileDialog, QMessageBox

def load_data_from_file(self):
    """
    Load data from a user-selected file via file dialog.
    
    Parameters:
        self: The parent application instance containing:
            - translator: For localizing UI messages
            - data_loader: Component with methods for loading different file formats
            - results_text: Text widget to display information about loaded data
            
    Returns:
        None: Data is stored as instance attributes:
            - data: NumPy array containing the loaded data
            - original_columns: List of column names from the original dataset
            
    Raises:
        Exception: If there is an error loading or parsing the file,
                  the exception is caught and displayed to the user
    """
    tr = self.translator
    
    options = QFileDialog.Options()
    file_path, _ = QFileDialog.getOpenFileName(
        self,
        tr('file_open_title'),
        "",
        tr('file_types'),
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
        info_text = f"{tr('msg_data_loaded')}\n\n"
        info_text += f"{tr('file_open_title')}: {file_path}\n"
        info_text += f"{tr('data_preview')}: {file_type}\n"
        info_text += f"{tr('data_samples')}: {n_samples}\n"
        info_text += f"{tr('data_features')}: {n_features}\n"
        
        if hasattr(self, 'original_columns') and self.original_columns:
            feature_names = f"\n{tr('data_features')}: {', '.join(self.original_columns[:10])}"
            
            if len(self.original_columns) > 10:
                feature_names += f" {tr('data_features')} {len(self.original_columns) - 10}..."
                
            info_text += feature_names
        
        # Display in text area
        self.results_text.setText(info_text)

    except Exception as e:
        error_msg = f"{tr('msg_error')}: {str(e)}"
        QMessageBox.critical(self, tr('msg_error'), error_msg)
        self.results_text.setText(error_msg)