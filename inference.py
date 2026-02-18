#!/usr/bin/env python3
"""
Inference script for segmagic model.
Takes a config TOML, an image file, and one or more model weights (.pth),
then outputs an annotation GeoJSON.

Usage:
    python inference.py --config xenium.toml --image input.tif --models models/best_model.pth --output output.geojson
    python inference.py --config xenium.toml --image input.tif --models model_0.pth model_1.pth model_2.pth -o output.geojson
"""

from pathlib import Path
from typing import List, Optional

import numpy as np
import tifffile as tiff
import torch
import typer

from settings import Settings
from normalize import load_and_normalize_tiff
from segmagic_ml import Segmagic

app = typer.Typer(help="Run segmagic inference on an image and output a GeoJSON annotation.")


def load_models(model_paths: List[Path]) -> list:
    """Load one or more model files and return them in eval mode on GPU."""
    models = []
    for path in model_paths:
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        model = torch.load(path, weights_only=False)
        model.eval()
        model.cuda()
        model.deep_supervision = False
        models.append(model)
        print(f"Loaded model: {path}")
    return models


@app.command()
def main(
    config: Path = typer.Option(..., "--config", "-c", help="Path to the project TOML config (merged with default.toml)."),
    image: Path = typer.Option(..., "--image", "-i", help="Path to the input image (TIFF)."),
    models: List[Path] = typer.Option(..., "--models", "-m", help="Path(s) to model weight files (.pth). Repeat for ensemble."),
    output: Path = typer.Option(..., "--output", "-o", help="Path for the output GeoJSON file."),
    default_config: Path = typer.Option("default.toml", "--default-config", help="Path to the default TOML config."),
    threshold: float = typer.Option(0.5, "--threshold", "-t", help="Prediction threshold."),
    save_mask: Optional[Path] = typer.Option(None, "--save-mask", help="Optional path to save the predicted mask as TIFF."),
    save_uncertainty: Optional[Path] = typer.Option(None, "--save-uncertainty", help="Optional path to save the uncertainty map as TIFF."),
):
    """Run segmagic inference on a single image."""

    # Load settings (merge project config with defaults)
    settings = Settings(
        default_config=str(default_config),
        project_config=str(config),
    )

    labels = settings.get("dataset.labels")
    norm_method = settings.get("dataset.normalization_method", "loq")

    # Load and normalize image
    print(f"Loading image: {image}")
    img = load_and_normalize_tiff(str(image), norm_method)
    print(f"Image shape: {img.shape} (C, H, W)")

    # Build a Segmagic instance and inject pre-loaded models
    seg = Segmagic(settings)
    loaded_models = load_models(models)

    if len(loaded_models) == 1:
        seg.model = loaded_models[0]
        seg.ensemble = False
        print("Using single model for prediction.")
    else:
        seg.models = loaded_models
        seg.ensemble = True
        print(f"Using ensemble of {len(loaded_models)} models for prediction.")

    # Run prediction (reuses Segmagic.predict_image)
    predicted_mask, uncertainty = seg.predict_image(
        img, labels, threshold=threshold, show=False
    )

    # Save GeoJSON (reuses Segmagic.save_geojson)
    output_str = str(output)
    if not output_str.endswith(".geojson"):
        output_str += ".geojson"
    # save_geojson appends .geojson, so strip it for the call
    seg.save_geojson(predicted_mask, labels, output_str.removesuffix(".geojson"))
    print(f"Saved GeoJSON: {output_str}")

    # Optionally save mask and uncertainty
    if save_mask:
        tiff.imwrite(str(save_mask), predicted_mask)
        print(f"Saved mask: {save_mask}")

    if save_uncertainty:
        tiff.imwrite(str(save_uncertainty), np.uint8(uncertainty * 255))
        print(f"Saved uncertainty: {save_uncertainty}")

    print("Inference complete.")


if __name__ == "__main__":
    app()
