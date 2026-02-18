#!/usr/bin/env python3
"""
Training script for segmagic model.
Converted from train-ml.ipynb
"""

import json
from settings import Settings
from dataloader_ml import DataHandler
from segmagic_ml import Segmagic


def main():
    # Load configuration using the Settings class
    settings = Settings(
        default_config="default.toml",
        project_config="xenium.toml"
    )

    print(f"Project: {settings.get('general.name')}")
    print(f"Base path: {settings.get('data.base_path')}")
    print(f"Model folder: {settings.get('data.model_folder')}")

    # Create necessary directories
    settings.create_directories()

    # Load and split data
    print("\n### Loading and splitting data ###")
    data = DataHandler(settings)
    data.train_test_split(test_ratio=0.1, valid_ratio=0.2)
    distribution = data.run_data_loader()

    with open(settings.get('data.model_folder') + "/distribution.json", 'w') as f:
        json.dump(distribution, f, indent=4)

    # Train model
    print("\n### Training model ###")
    print("You will find the best model and its metrics in 'base_path'/model")
    seg = Segmagic(settings)
    seg.train_model(data)

    # Test model
    print("\n### Testing model ###")
    print("Test results are stored under 'base_path'/Testing")
    seg.test_images(data)


if __name__ == "__main__":
    main()
