"""
Example usage of the Settings class for Segmagic project.
This script demonstrates various ways to use the Settings class.
"""

from settings import Settings, load_settings
from pathlib import Path

def any_none_values(data):
    """Recursively check if there are any None values in the configuration."""
    if isinstance(data, dict):
        return any(any_none_values(value) for value in data.values())
    elif isinstance(data, list):
        return any(any_none_values(item) for item in data)
    else:
        return data is None

def main():
    print("=== Settings Class Example ===\n")
    
    # Method 1: Direct instantiation
    print("1. Loading settings with direct instantiation:")
    settings = Settings("default.toml", "example.toml")
    print(f"   Project name: {settings.get('general.name')}")
    print(f"   Base path: {settings.get('data.base_path')}")
    print(f"   Model folder: {settings.get('data.model_folder')}")
    print()
    
    # Method 2: Using convenience function
    print("2. Loading settings with convenience function:")
    settings2 = load_settings("default.toml", "example.toml", create_dirs=False)
    print(f"   Settings loaded: {settings2}")
    print()
    
    # Method 3: Dictionary-style access
    print("3. Dictionary-style access:")
    print(f"   Batch size: {settings['training.dataloader.batch_size']}")
    print(f"   Loss name: {settings['training.loss_name']}")
    print()
    
    # Method 4: Getting data paths
    print("4. Data paths:")
    paths = settings.get_data_paths()
    for name, path in paths.items():
        print(f"   {name}: {path}")
    print()
    
    # Method 5: Modifying configuration
    print("5. Modifying configuration:")
    original_batch_size = settings.get('training.dataloader.batch_size')
    print(f"   Original batch size: {original_batch_size}")
    
    settings.set('training.dataloader.batch_size', 32)
    new_batch_size = settings.get('training.dataloader.batch_size')
    print(f"   New batch size: {new_batch_size}")
    print()
    
    # Method 6: Saving modified configuration
    print("6. Saving configuration:")
    output_path = "modified_config.toml"
    settings.save_config(output_path)
    print(f"   Configuration saved to: {output_path}")
    
    # Clean up the example file
    if Path(output_path).exists():
        Path(output_path).unlink()
        print(f"   Cleaned up: {output_path}")
    print()
    
    # Method 7: Error handling and None conversion
    print("7. Error handling and None conversion:")
    non_existent_value = settings.get('non.existent.key', 'default_value')
    print(f"   Non-existent key with default: {non_existent_value}")
    
    # Test None conversion (if there are any "None" strings in the config)
    print("   Testing None conversion:")
    print(f"   Any None values in config: {any_none_values(settings.to_dict())}")
    print()
    
    print("=== Example completed successfully! ===")


if __name__ == "__main__":
    main()