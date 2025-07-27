"""
Settings module for Segmagic project.
Handles loading and merging of TOML configuration files with path adjustments.
"""

import toml
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union
import os


class Settings:
    """
    A class to handle loading and merging TOML configuration files with automatic path adjustments.
    
    This class loads a default configuration file and an optional project-specific configuration,
    merges them, and performs necessary path adjustments to ensure all paths are absolute and valid.
    """
    
    def __init__(
            self, 
            default_config: Union[str, Path] = "default.toml", 
            project_config: Optional[Union[str, Path]] = None
        ) -> None:
        """
        Initialize the Settings class.
        
        Args:
            default_config: Path to the default configuration file
            project_config: Path to the project-specific configuration file (optional)
        """
        self.default_config_path = Path(default_config)
        self.project_config_path = Path(project_config) if project_config else None
        self.config = {}
        self._load_configurations()
        self._adjust_paths()
    
    def _load_configurations(self) -> None:
        """Load and merge configuration files."""
        # Load default configuration
        if not self.default_config_path.exists():
            raise FileNotFoundError(f"Default configuration file not found: {self.default_config_path}")
        
        # Use json.loads(json.dumps()) to ensure deep copy and avoid TOML-specific types
        default_data = json.loads(json.dumps(toml.load(self.default_config_path)))
        
        # Start with default configuration and convert "None" strings to None
        self.config = self._convert_none_strings(default_data.copy())
        
        # Load and merge project configuration if provided
        if self.project_config_path and self.project_config_path.exists():
            project_data = json.loads(json.dumps(toml.load(self.project_config_path)))
            project_data = self._convert_none_strings(project_data)
            self.config = self._deep_merge(self.config, project_data)
        elif self.project_config_path:
            print(f"Warning: Project configuration file not found: {self.project_config_path}")
    
    def _convert_none_strings(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively convert string "None" values to Python None.
        
        Args:
            data: The configuration dictionary to process
            
        Returns:
            The processed configuration dictionary with "None" strings converted to None
        """
        if isinstance(data, dict):
            return {key: self._convert_none_strings(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._convert_none_strings(item) for item in data]
        elif isinstance(data, str) and data.lower() == "none":
            return None
        else:
            return data
    
    def _deep_merge(self, default: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively merge two dictionaries, with override values taking precedence.
        
        Args:
            default: The default configuration dictionary
            override: The override configuration dictionary
            
        Returns:
            The merged configuration dictionary
        """
        merged = default.copy()
        
        for key, value in override.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._deep_merge(merged[key], value)
            else:
                merged[key] = value
        
        return merged
    
    def _adjust_paths(self) -> None:
        """Adjust relative paths to absolute paths based on the configuration."""
        # Get the base path for relative path resolution
        base_path = self.get("data.base_path", "")
        
        if base_path:
            # Convert base_path to absolute if it's relative
            base_path = Path(base_path)
            if not base_path.is_absolute():
                # Make it relative to the configuration file location
                config_dir = self.default_config_path.parent
                base_path = (config_dir / base_path).resolve()
            
            # Update the base_path in config
            self._set_nested_value("data.base_path", str(base_path))
            
            # Adjust model folder path
            model_folder = self.get("data.model_folder", "models")
            if not Path(model_folder).is_absolute():
                model_path = base_path / model_folder
                self._set_nested_value("data.model_folder", str(model_path))
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.
        
        Args:
            key: The configuration key in dot notation (e.g., 'data.base_path')
            default: Default value if key is not found
            
        Returns:
            The configuration value or default
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def _set_nested_value(self, key: str, value: Any) -> None:
        """
        Set a nested configuration value using dot notation.
        
        Args:
            key: The configuration key in dot notation
            value: The value to set
        """
        keys = key.split('.')
        config = self.config
        
        # Navigate to the parent dictionary
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the final value
        config[keys[-1]] = value
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value using dot notation.
        
        Args:
            key: The configuration key in dot notation
            value: The value to set
        """
        self._set_nested_value(key, value)
    
    def get_data_paths(self) -> Dict[str, str]:
        """
        Get commonly used data paths.
        
        Returns:
            Dictionary containing common paths like base_path, model_path, etc.
        """
        base_path = self.get("data.base_path", "")
        model_folder = self.get("data.model_folder", "models")
        
        paths = {
            "base_path": base_path,
            "model_path": model_folder,
        }
        
        # Add additional common paths if base_path exists
        if base_path:
            base = Path(base_path)
            paths.update({
                "training_path": str(base / "Training"),
                "testing_path": str(base / "Testing"),
                "data_path": str(base / "Data"),
                "cache_path": str(base / "cache"),
            })
        
        return paths
    
    def create_directories(self, paths: Optional[Dict[str, str]] = None) -> None:
        """
        Create necessary directories based on configuration.
        
        Args:
            paths: Optional dictionary of paths to create. If None, uses get_data_paths()
        """
        if paths is None:
            paths = self.get_data_paths()
        
        for path_name, path_value in paths.items():
            if path_value:
                path_obj = Path(path_value)
                path_obj.mkdir(parents=True, exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Get the full configuration as a dictionary.
        
        Returns:
            The complete configuration dictionary
        """
        return self.config.copy()
    
    def save_config(self, output_path: Union[str, Path]) -> None:
        """
        Save the current configuration to a TOML file.
        
        Args:
            output_path: Path where to save the configuration
        """
        output_path = Path(output_path)
        # Convert None values back to "None" strings for TOML compatibility
        toml_compatible_config = self._convert_none_to_strings(self.config)
        with open(output_path, 'w') as f:
            toml.dump(toml_compatible_config, f)
    
    def _convert_none_to_strings(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively convert Python None values to string "None" for TOML compatibility.
        
        Args:
            data: The configuration dictionary to process
            
        Returns:
            The processed configuration dictionary with None values converted to "None" strings
        """
        if isinstance(data, dict):
            return {key: self._convert_none_to_strings(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._convert_none_to_strings(item) for item in data]
        elif data is None:
            return "None"
        else:
            return data
    
    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access to configuration."""
        return self.get(key)
    
    def __setitem__(self, key: str, value: Any) -> None:
        """Allow dictionary-style setting of configuration."""
        self.set(key, value)
    
    def __repr__(self) -> str:
        """String representation of the Settings object."""
        return f"Settings(default='{self.default_config_path}', project='{self.project_config_path}')"


# Convenience function for quick setup
def load_settings(
        default_config: str = "default.toml", 
        project_config: Optional[str] = None,
        create_dirs: bool = True
    ) -> Settings:
    """
    Convenience function to load settings with optional directory creation.
    
    Args:
        default_config: Path to default configuration file
        project_config: Path to project configuration file (optional)
        create_dirs: Whether to automatically create necessary directories
        
    Returns:
        Configured Settings instance
    """
    settings = Settings(default_config, project_config)
    
    if create_dirs:
        settings.create_directories()
    
    return settings


if __name__ == "__main__":
    # Example usage
    settings = load_settings("default.toml", "example.toml")
    print("Configuration loaded successfully!")
    print(f"Base path: {settings.get('data.base_path')}")
    print(f"Model folder: {settings.get('data.model_folder')}")
    print(f"Project name: {settings.get('general.name')}")
    
    # Show all data paths
    print("\nData paths:")
    for name, path in settings.get_data_paths().items():
        print(f"  {name}: {path}")
