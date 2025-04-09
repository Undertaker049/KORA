"""
Module for performing clustering operations.

Provides functionality for executing and evaluating clustering
operations on preprocessed data.
"""

import numpy as np
from PyQt5.QtWidgets import QMessageBox

def perform_clustering(self):
    """
    Perform clustering on preprocessed data and visualize the results.
    
    Executes the clustering algorithm (K-means by default) with 
    the parameters specified in the UI.
    
    Parameters:
        self: The parent application instance containing:
            - processed_data: The preprocessed dataset to cluster
            - n_clusters_spin: UI element for number of clusters selection
            - max_iter_spin: UI element for maximum iterations selection
            - clustering_class: The clustering algorithm class to use
            - translator: Translator object for UI messages
            - results_text: Text area to display results
            - tabs: Tab widget for switching visualization focus
            
    Returns:
        None: Results are displayed in the UI and stored as instance attributes
        
    Raises:
        Exception: If clustering fails or evaluation encounters errors
    """
    tr = self.translator
    
    if self.processed_data is None:
        QMessageBox.warning(self, tr('msg_warning'), tr('msg_no_processed_data'))
        return
        
    try:

        # Get clustering parameters
        n_clusters = self.n_clusters_spin.value()
        max_iter = self.max_iter_spin.value()
        
        # Create model
        self.kmeans = self.clustering_class(
            n_clusters=n_clusters,
            max_iter=max_iter,
            random_state=42
        )
        
        # Perform clustering
        self.labels = self.kmeans.fit_predict(self.processed_data)
        
        # Evaluate results
        evaluation = self.kmeans.evaluate(self.processed_data)
        
        # Calculate and display elbow method plot
        try:

            # Get range of k values (from 1 to n_clusters+5, but no more than 15)
            k_max = min(n_clusters + 5, 15)
            k_range = list(range(1, k_max))
            
            # Find optimal k and create plot
            temp_kmeans = self.clustering_class()
            k_range, inertia_values = temp_kmeans.optimal_k_elbow(
                self.processed_data,
                k_range=k_range
            )
            
            # Save data for visualization
            self.elbow_k_range = k_range
            self.elbow_curve = inertia_values
            
        except Exception as e:
            print(f"Error calculating elbow method: {str(e)}")

        # Calculate silhouette coefficients if more than one cluster
        if n_clusters > 1:

            try:
                from sklearn.metrics import silhouette_samples
                self.silhouette_values = silhouette_samples(self.processed_data, self.labels)

            except Exception as e:
                print(f"Error calculating silhouette coefficients: {str(e)}")
                self.silhouette_values = None

        # Calculate feature importance (if data available)
        try:

            if hasattr(self, 'processed_data') and self.processed_data is not None:

                # Use built-in feature importance metric or variance
                from sklearn.feature_selection import mutual_info_classif
                
                # Calculate mutual information between features and cluster labels
                self.feature_importance = mutual_info_classif(
                    self.processed_data, 
                    self.labels,
                    discrete_features=False,
                    random_state=42
                )
                
                # If we have original column names, save them
                if hasattr(self, 'data') and hasattr(self.data, 'columns'):
                    self.original_columns = self.data.columns.tolist()

            else:
                self.feature_importance = None

        except Exception as e:
            print(f"Error calculating feature importance: {str(e)}")
            self.feature_importance = None

        # Visualize results
        self.visualize_results()
        
        # Results information
        info_text = f"{tr('clustering_results')}:\n\n"
        info_text += f"{tr('clustering_k')}: {n_clusters}\n"
        info_text += f"{tr('metric_inertia')}: {evaluation['inertia']:.4f}\n"
        
        if 'silhouette_score' in evaluation:
            info_text += f"{tr('metric_silhouette')}: {evaluation['silhouette_score']:.4f}\n"
            
        if 'calinski_harabasz_score' in evaluation:
            info_text += f"{tr('metric_calinski_harabasz')}: {evaluation['calinski_harabasz_score']:.4f}\n"
            
        if 'davies_bouldin_score' in evaluation:
            info_text += f"{tr('metric_davies_bouldin')}: {evaluation['davies_bouldin_score']:.4f}\n"
        
        # Add cluster size information
        if self.labels is not None:
            info_text += f"\n{tr('plot_cluster')}:\n"
            unique_labels, counts = np.unique(self.labels, return_counts=True)
            
            for label, count in zip(unique_labels, counts):
                info_text += f"{tr('plot_cluster')} {label}: {count} {tr('data_samples')}\n"
        
        # Display results in the text field instead of message box
        self.results_text.setText(info_text)
        
        # Switch to the clusters tab to show visualization
        self.tabs.setCurrentIndex(0)
    
    except Exception as e:
        QMessageBox.critical(self, tr('msg_error'), f"{tr('msg_error')}: {str(e)}")
        self.results_text.setText(f"{tr('msg_error')}: {str(e)}")

def update_data_info(self):
    """
    Update the displayed information about the loaded dataset.
    
    Parameters:
        self: The parent application instance containing:
            - data: The loaded dataset (or None if no data is loaded)
            - data_info_label: UI label element to update
            - translator: Translator object for UI messages
            
    Returns:
        None: The UI is updated directly
    """
    tr = self.translator
    
    if self.data is not None:
        n_samples, n_features = self.data.shape
        self.data_info_label.setText(f"{tr('tab_data')}: {n_samples} {tr('data_samples')}, {n_features} {tr('data_features')}")
    
    else:
        self.data_info_label.setText(tr('msg_no_data_loaded'))

def preprocess_data(self):
    """
    Preprocess the loaded dataset according to user-selected options.
    
    Applies preprocessing techniques to the loaded dataset
    based on the options selected in the UI.
    
    Parameters:
        self: The parent application instance containing:
            - data: The original dataset to preprocess
            - scale_check, scale_method_combo: UI elements for scaling settings
            - missing_check, missing_method_combo: UI elements for missing value handling
            - dim_reduce_check, dim_reduce_method_combo: UI elements for dimensionality reduction
            - n_components_spin: UI element for number of components selection
            - preprocessor: Object with preprocessing methods
            - visualizer: Object with visualization methods including dimensionality reduction
            - translator: Translator object for UI messages
            - results_text: Text area to display results
            
    Returns:
        None: Processed data is stored as instance attributes (processed_data, reduced_data)
        
    Raises:
        Exception: If preprocessing fails due to invalid data or parameters
    """
    tr = self.translator
    
    if self.data is None:
        QMessageBox.warning(self, tr('msg_warning'), tr('msg_load_data_first'))
        return
        
    try:

        # Get preprocessing parameters
        scale = self.scale_check.isChecked()
        scaling_method = self.scale_method_combo.currentText()
        handle_missing = self.missing_check.isChecked()
        missing_strategy = self.missing_method_combo.currentText()
        reduce_dims = self.dim_reduce_check.isChecked()
        n_components = self.n_components_spin.value()
        dim_reduce_method = self.dim_reduce_method_combo.currentText()
        
        # Preprocess data
        self.processed_data = self.preprocessor.preprocess_pipeline(
            self.data,
            scale=scale,
            scaling_method=scaling_method,
            handle_missing=handle_missing,
            missing_strategy=missing_strategy,
            reduce_dims=False
        )
        
        # Reduce dimensionality if needed
        if reduce_dims:
            self.reduced_data = self.visualizer.reduce_dimensions(
                self.processed_data,
                method=dim_reduce_method,
                n_components=n_components
            )
            
        else:
            self.reduced_data = self.processed_data
        
        # Update results text    
        self.results_text.setText(tr('msg_preprocessing_done'))
    
    except Exception as e:
        QMessageBox.critical(self, tr('msg_error'), f"{tr('msg_error')}: {str(e)}")
        self.results_text.setText(f"{tr('msg_error')}: {str(e)}")