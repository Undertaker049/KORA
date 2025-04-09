"""
UI Localization module.

Provides functions to update all UI elements
when the application language is changed.
"""

from PyQt5.QtWidgets import QDialog, QGroupBox, QFormLayout, QLabel
from PyQt5.QtCore import Qt


def update_ui_language(self):
    """
    Update all UI elements with the current language.
    
    Parameters:
        self: Parent application with translator and UI components
        
    Notes:
        Visualization updates are only performed when data is available
    """
    tr = self.translator.translate
    
    # Update window title
    self.setWindowTitle(tr('app_title'))
    
    # Update menus
    update_menu_language(self)
    
    # Update tabs
    update_tab_language(self)
    
    # Update other UI elements
    update_ui_elements_language(self)
    
    # Check if visualization data exists
    has_visualization_data = (
        hasattr(self, 'reduced_data') and self.reduced_data is not None and
        hasattr(self, 'labels') and self.labels is not None and
        len(getattr(self, 'reduced_data', [])) > 0 and
        len(getattr(self, 'labels', [])) > 0
    )
    
    # Update visualizations only if data exists
    if has_visualization_data:

        try:
            update_visualization_language(self)

        except Exception as e:
            print(f"Error updating visualization language: {str(e)}")


def update_menu_language(self):
    """
    Update menu text elements with current language.
    
    Parameters:
        self: Parent application with menu components and translator
    """
    tr = self.translator.translate
    
    if hasattr(self, 'menu_file'):
        self.menu_file.setTitle(tr('menu_file'))
        self.action_open.setText(tr('menu_open'))
        self.action_exit.setText(tr('menu_exit'))
        
    if hasattr(self, 'menu_data'):
        self.menu_data.setTitle(tr('menu_data'))
        self.action_preprocess.setText(tr('menu_preprocess'))
        
    if hasattr(self, 'menu_analysis'):
        self.menu_analysis.setTitle(tr('menu_analysis'))
        self.action_perform_clustering.setText(tr('menu_perform_clustering'))
        
    if hasattr(self, 'menu_results'):
        self.menu_results.setTitle(tr('menu_results'))
        self.action_save_results.setText(tr('menu_save_results'))
        self.action_visualize.setText(tr('menu_visualize'))
        
    if hasattr(self, 'menu_settings'):
        self.menu_settings.setTitle(tr('menu_settings'))
        self.menu_language.setTitle(tr('menu_language'))
        
        # Update language menu items
        for lang, action in self.language_actions.items():
            action.setChecked(lang == self.current_language)
            lang_key = f'menu_language_{lang}'
            action.setText(tr(lang_key))
        
    if hasattr(self, 'menu_help'):
        self.menu_help.setTitle(tr('menu_help'))
        self.action_about.setText(tr('menu_about'))


def update_tab_language(self):
    """
    Update tab labels with current language.
    
    Parameters:
        self: Parent application with tabs and translator
    """
    tr = self.translator.translate
    
    if hasattr(self, 'tabs'):
        if hasattr(self, 'tab_data'):
            self.tabs.setTabText(self.tabs.indexOf(self.tab_data), tr('tab_data'))
            
        if hasattr(self, 'tab_clustering'):
            self.tabs.setTabText(self.tabs.indexOf(self.tab_clustering), tr('tab_clustering'))
            
        if hasattr(self, 'tab_visualization'):
            self.tabs.setTabText(self.tabs.indexOf(self.tab_visualization), tr('tab_visualization'))


def update_ui_elements_language(self):
    """
    Update all remaining UI elements with current language.
    
    Updates group boxes, buttons, labels and other UI elements not
    handled by menu and tab update functions.
    
    Parameters:
        self: Parent application with UI components and translator
    """
    tr = self.translator.translate
    
    # Update group box titles
    for widget in self.findChildren(QGroupBox):
        
        # Determine title based on current content
        if widget.title() in ['Data', 'Данные']:
            widget.setTitle(tr('tab_data'))

        elif widget.title() in ['Preprocessing', 'Предобработка данных']:
            widget.setTitle(tr('menu_preprocess'))

        elif widget.title() in ['Clustering', 'Кластеризация']:
            widget.setTitle(tr('tab_clustering'))

        elif widget.title() in ['Information', 'Информация о данных']:
            widget.setTitle(tr('data_info'))
    
    # Update buttons
    if hasattr(self, 'load_data_btn'):
        self.load_data_btn.setText(tr('menu_open'))
        
    if hasattr(self, 'preprocess_btn'):
        self.preprocess_btn.setText(tr('preprocess_button'))
        
    if hasattr(self, 'cluster_btn'):
        self.cluster_btn.setText(tr('clustering_button'))
        
    if hasattr(self, 'save_results_btn'):
        self.save_results_btn.setText(tr('button_save'))
    
    # Update labels
    if hasattr(self, 'data_info_label'):
        self.data_info_label.setText(tr('data_preview'))
        
    if hasattr(self, 'results_label'):
        self.results_label.setText(tr('clustering_results') + ":")
    
    # Update tab texts
    if hasattr(self, 'tabs'):
        self.tabs.setTabText(0, tr('plot_cluster'))
        self.tabs.setTabText(1, tr('plot_elbow_title'))
        self.tabs.setTabText(2, tr('plot_silhouette_title'))
        self.tabs.setTabText(3, tr('data_features'))
    
    # Update form layout labels
    if hasattr(self, 'scale_check'):

        # Find parent layout
        parent_layout = self.scale_check.parent().layout()

        if parent_layout:

            # Get element index
            for i in range(parent_layout.rowCount()):

                # If QFormLayout, get labelForField
                if isinstance(parent_layout, QFormLayout):
                    label = parent_layout.itemAt(i, QFormLayout.LabelRole)

                    if label and label.widget():

                        # Update text based on context
                        text = label.widget().text()

                        if 'scale' in text.lower() or 'масштаб' in text.lower():
                            label.widget().setText(tr('preprocess_scale') + ":")

                        elif 'missing' in text.lower() or 'пропущен' in text.lower():
                            label.widget().setText(tr('preprocess_missing') + ":")

                        elif 'method' in text.lower() or 'метод' in text.lower():
                            label.widget().setText(tr('viz_method') + ":")

                        elif 'features' in text.lower() or 'признак' in text.lower():
                            label.widget().setText(tr('data_features') + ":")

                        elif 'clusters' in text.lower() or 'кластер' in text.lower():
                            label.widget().setText(tr('clustering_k') + ":")

                        elif 'iterations' in text.lower() or 'итера' in text.lower():
                            label.widget().setText(tr('clustering_max_iter') + ":")
                    
    # Update all texts in visualization tabs
    for tab in [self.clusters_tab, self.elbow_tab, self.silhouette_tab, self.features_tab]:

        for label in tab.findChildren(QLabel):

            # Update text based on context
            if 'component 1' in label.text().lower() or 'компонент 1' in label.text().lower():
                label.setText(tr('plot_component1'))

            elif 'component 2' in label.text().lower() or 'компонент 2' in label.text().lower():
                label.setText(tr('plot_component2'))

            elif 'cluster' in label.text().lower() or 'кластер' in label.text().lower():
                label.setText(tr('plot_cluster'))

            elif 'elbow' in label.text().lower() or 'локт' in label.text().lower():
                label.setText(tr('plot_elbow_title'))

            elif 'silhouette' in label.text().lower() or 'силуэт' in label.text().lower():
                label.setText(tr('plot_silhouette_title'))


def change_language(self, language):
    """
    Change application language and update interface.
    
    Parameters:
        self: Parent application with translator and UI components
        language: Language code to switch to
    """
    from PyQt5.QtWidgets import QApplication
    
    if language not in self.available_languages:
        return
        
    if language == self.current_language:
        return
        
    # Set wait cursor to indicate processing
    QApplication.setOverrideCursor(Qt.WaitCursor)
    
    try:

        # Set new language
        self.current_language = language
        from src.localization import set_language
        set_language(language)
        
        # Update translator after language change
        self.translator.language = language
        
        # Check if visualization update is needed
        has_visualization_data = (
            hasattr(self, 'reduced_data') and self.reduced_data is not None and
            hasattr(self, 'labels') and self.labels is not None
        )
        
        # Clear plots before updating if they exist
        if has_visualization_data:

            try:

                if hasattr(self, 'clusters_canvas') and self.clusters_canvas:
                    self.clusters_canvas.axes.clear()
                    
                if hasattr(self, 'elbow_canvas') and self.elbow_canvas:
                    self.elbow_canvas.axes.clear()
                    
                if hasattr(self, 'silhouette_canvas') and self.silhouette_canvas:
                    self.silhouette_canvas.axes.clear()
                    
                if hasattr(self, 'features_canvas') and self.features_canvas:
                    self.features_canvas.axes.clear()

            except Exception as e:
                print(f"Error clearing plots: {str(e)}")
        
        update_ui_language(self)
        
        # Update plots with new labels if visualization data exists
        if has_visualization_data:

            try:

                # Try to update visualizations
                update_visualization_language(self)

            except Exception as e:
                print(f"Error updating visualization language: {str(e)}")
                import traceback
                traceback.print_exc()
            
            # Redraw all plots with error handling
            try:

                for canvas_name in ['clusters_canvas', 'elbow_canvas', 'silhouette_canvas', 'features_canvas']:
                    
                    if hasattr(self, canvas_name):
                        canvas = getattr(self, canvas_name)
                        
                        if canvas and hasattr(canvas, 'figure') and canvas.figure:
                            canvas.figure.tight_layout()
                            canvas.draw()

            except Exception as e:
                print(f"Error redrawing plots: {str(e)}")
            
        # Update all child dialogs
        for dialog in self.findChildren(QDialog):

            if hasattr(dialog, 'update_language'):

                try:
                    dialog.update_language()

                except Exception as e:
                    print(f"Error updating dialog language: {str(e)}")
        
    finally:

        # Restore normal cursor
        QApplication.restoreOverrideCursor()


def update_visualization_language(self):
    """
    Update language on all plots.
    
    Delegates to implementation in _update_visualization_language module.
    
    Parameters:
        self: Parent application with visualization components
        
    Notes:
        Performs validation before attempting update
        Handles errors to prevent localization process interruption
    """
    # Check if necessary data exists for plot updates
    if not all(hasattr(self, attr) for attr in ['reduced_data', 'labels']):
        # If no data, nothing to update
        return
        
    if getattr(self, 'reduced_data', None) is None or getattr(self, 'labels', None) is None:
        # If data not loaded, nothing to update
        return
    
    # Check for plot existence
    if not hasattr(self, 'clusters_canvas') or self.clusters_canvas is None:
        # If plots not created yet, nothing to update
        return
    
    try:
        # Import and call function from separate module
        from ._update_visualization_language import update_visualization_language as update_vis
        update_vis(self)
    except Exception as e:
        print(f"Error updating visualization: {str(e)}")
        # Don't let visualization errors interrupt localization process

__all__ = [
    'update_ui_language',
    'update_menu_language',
    'update_tab_language',
    'update_ui_elements_language',
    'change_language',
    'update_visualization_language'
]