"""
Translator module for managing application translations.
"""

import os
import json
from typing import Dict, Optional

# Path to the translations directory
TRANSLATIONS_DIR = os.path.join(os.path.dirname(__file__), 'translations')

class Translator:
    """
    Class for handling translations between languages.
    
    Attributes:
        language (str): Current language code ('en' or 'ru')
        translations (dict): Dictionary with translations for the current language
        available_languages (list): List of available language codes
    """
    
    _instance = None
    
    def __new__(cls, language: str = 'en'):
        """
        Singleton pattern to ensure only one Translator instance exists.
        
        Args:
            language: Language code ('en' or 'ru')
            
        Returns:
            Translator instance
        """
        if cls._instance is None:
            cls._instance = super(Translator, cls).__new__(cls)
            cls._instance.language = language
            cls._instance.translations = {}
            cls._instance.available_languages = cls._get_available_languages()
            cls._instance._load_translations()
        
        return cls._instance
    
    @staticmethod
    def _get_available_languages() -> list:
        """
        Get the list of available languages from the translations directory.
        
        Returns:
            List of available language codes
        """
        available_languages = []
        
        if not os.path.exists(TRANSLATIONS_DIR):
            return ['en']  # Default to English if no translations directory
            
        for filename in os.listdir(TRANSLATIONS_DIR):
            if filename.endswith('.json'):
                lang_code = filename.split('.')[0]
                available_languages.append(lang_code)
                
        return available_languages if available_languages else ['en']
    
    def _load_translations(self) -> None:
        """
        Load translations for the current language from a JSON file.
        If the language file doesn't exist, fall back to English.
        """
        # Default to empty dictionary
        self.translations = {}
        
        # Define the file path for the current language
        lang_file = os.path.join(TRANSLATIONS_DIR, f"{self.language}.json")
        
        # If the language file doesn't exist, try to fall back to English
        if not os.path.exists(lang_file) and self.language != 'en':
            self.language = 'en'
            lang_file = os.path.join(TRANSLATIONS_DIR, "en.json")
            
        # Load translations from the file
        try:
            with open(lang_file, 'r', encoding='utf-8') as f:
                self.translations = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading translations: {str(e)}")
            # Initialize with empty dictionary if loading fails
            self.translations = {}
        
    def set_language(self, language: str) -> None:
        """
        Set the current language and reload translations.
        
        Args:
            language: Language code ('en', 'ru', etc.)
        """
        if language not in self.available_languages:
            print(f"Warning: Unsupported language: {language}. Using English instead.")
            language = 'en'
            
        if language != self.language:
            self.language = language
            self._load_translations()
    
    def translate(self, key: str) -> str:
        """
        Translate a key to the current language.
        
        Args:
            key: Translation key
            
        Returns:
            Translated string in the current language
        """
        return self.translations.get(key, key)
        
    def __call__(self, key: str) -> str:
        """
        Call method to use the translator as a function.
        
        Args:
            key: Translation key
            
        Returns:
            Translated string
        """
        return self.translate(key)


# Global translator instance
_translator = None

def get_translator(language: str = None) -> Translator:
    """
    Get the global translator instance.
    
    Creates a new translator if one doesn't exist yet, or returns the existing one.
    If a language is specified, it will be used to set the translator's language.
    
    Args:
        language: Optional language code to set on the translator
    
    Returns:
        Translator instance configured with the appropriate language
    """
    global _translator
    
    if _translator is None:
        # Create new translator with default language (English)
        # or the specified language if provided
        _translator = Translator(language if language is not None else 'en')
        
    elif language is not None:
        # If translator exists but language is specified, update it
        _translator.set_language(language)
        
    return _translator
    
def set_language(language: str) -> None:
    """
    Set the global translator language.
    
    Args:
        language: Language code ('en', 'ru', etc.)
    """
    get_translator().set_language(language) 