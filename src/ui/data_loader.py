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

        elif file_path.endswith(('.xlsx', '.xls')):
            data_df = self.data_loader.load_excel(file_path)
            self.original_columns = data_df.columns.tolist()
            self.data = data_df.values

        elif file_path.endswith('.npy'):
            self.data = self.data_loader.load_numpy(file_path)
            self.original_columns = [f"Feature_{i}" for i in range(self.data.shape[1])]
        
        else:
            self.data = self.data_loader.load_csv(file_path)

            if isinstance(self.data, pd.DataFrame):
                self.original_columns = self.data.columns.tolist()
                self.data = self.data.values

            else:
                self.original_columns = [f"Feature_{i}" for i in range(self.data.shape[1])]
            
        self.update_data_info()
        QMessageBox.information(self, "Success", "Data successfully loaded.")

    except Exception as e:
        QMessageBox.critical(self, "Error", f"Error loading data: {str(e)}")
        
def generate_test_data(self):
    """
    Generate test data.
    """
    try:
        n_samples = self.n_samples_spin.value()
        n_features = self.n_features_spin.value()
        n_clusters = self.n_clusters_gen_spin.value()
        
        self.data, self.true_labels = self.data_loader.generate_sample_data(
            n_samples=n_samples,
            n_features=n_features,
            n_clusters=n_clusters,
            random_state=42
        )
        
        # Create column names for generated data
        self.original_columns = [f"Feature_{i}" for i in range(self.data.shape[1])]
        
        self.update_data_info()
        QMessageBox.information(self, "Success", "Test data successfully generated.")
   
    except Exception as e:
        QMessageBox.critical(self, "Error", f"Error generating data: {str(e)}")