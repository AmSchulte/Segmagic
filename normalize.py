import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
from sklearn.mixture import GaussianMixture

class BaseNormalizer(ABC):
    """Base class for all normalizers."""
    
    @abstractmethod
    def fit(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Compute normalization parameters from the input image.
        
        Args:
            image: Input image array
            
        Returns:
            Dictionary containing normalization parameters
        """
        pass
    
    @abstractmethod
    def transform(self, image: np.ndarray, settings: Dict[str, Any]) -> np.ndarray:
        """
        Apply normalization to the image using precomputed settings.
        
        Args:
            image: Input image array
            settings: Dictionary containing normalization parameters from fit()
            
        Returns:
            Normalized image array
        """
        pass


class QuantileNormalizer(BaseNormalizer):
    """Normalizer using quantile-based normalization (q5-q95)."""
    
    def fit(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Compute q5 and q95 quantiles for normalization.
        
        Args:
            image: Input image array with shape (C, H, W) or (H, W)
            
        Returns:
            Dictionary with 'q5' and 'q95' keys
        """
        if image.ndim == 2:
            # Single channel image
            q5 = np.quantile(image, 0.05)
            q95 = np.quantile(image, 0.95)
        else:
            # Multi-channel image, compute per channel
            q5 = np.quantile(image, 0.05, axis=(1, 2), keepdims=True)
            q95 = np.quantile(image, 0.95, axis=(1, 2), keepdims=True)
        
        return {
            'q5': q5,
            'q95': q95
        }
    
    def transform(self, image: np.ndarray, settings: Dict[str, Any]) -> np.ndarray:
        """
        Apply quantile normalization to scale image to [0, 1] range.
        
        Args:
            image: Input image array
            settings: Dictionary containing 'q5' and 'q95' values
            
        Returns:
            Normalized image as float32
        """
        q5 = settings['q5']
        q95 = settings['q95']
        
        # Avoid division by zero
        denominator = q95 - q5
        denominator = np.where(denominator == 0, 1, denominator)
        
        normalized = (image - q5) / denominator
        normalized = (normalized - 0.5) * 2
        return np.float32(normalized)

class LogNormalizer(BaseNormalizer):
    """Log-based normalization mapping data to roughly [-1, 1], without clipping."""

    def fit(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Store a small epsilon to stabilize log1p, and compute global scale.
        
        Args:
            image: Input array (C, H, W) or (H, W)
        
        Returns:
            Dictionary with 'scale' and 'epsilon'
        """
        epsilon = 1e-6  # Prevent log(0)
        if image.ndim == 2:
            scale = np.mean(np.log1p(image + epsilon))
        else:
            scale = np.mean(np.log1p(image + epsilon), axis=(1, 2), keepdims=True)

        return {
            'scale': scale,
            'epsilon': epsilon
        }

    def transform(self, image: np.ndarray, settings: Dict[str, Any]) -> np.ndarray:
        """
        Apply centered log1p normalization.

        Args:
            image: Input array
            settings: Dict with 'scale', 'epsilon'

        Returns:
            Normalized image in [-1, 1] (approx), float32
        """
        scale = settings['scale']
        epsilon = settings['epsilon']
        
        log_img = np.log1p(image + epsilon)
        centered = log_img - scale  # center around 0
        normalized = centered / (np.abs(centered).max() + 1e-8)  # scale to ≈ [-1, 1]
        
        return np.float32(normalized)

class LogQNormalizer(BaseNormalizer):
    """Log-based normalization with post-log quantile stretching to ~[-1, 1]."""

    def fit(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Fit log scale and determine 5/95% quantiles after log1p.
        
        Args:
            image: Input array (C, H, W) or (H, W)
        
        Returns:
            Dict with 'q5', 'q95', 'epsilon'
        """
        epsilon = 1e-6
        log_img = np.log1p(image + epsilon)

        if image.ndim == 2:
            q5 = np.quantile(log_img, 0.05)
            q95 = np.quantile(log_img, 0.95)
        else:
            q5 = np.quantile(log_img, 0.05, axis=(1, 2), keepdims=True)
            q95 = np.quantile(log_img, 0.95, axis=(1, 2), keepdims=True)

        return {
            'q5': q5,
            'q95': q95,
            'epsilon': epsilon
        }

    def transform(self, image: np.ndarray, settings: Dict[str, Any]) -> np.ndarray:
        """
        Apply log1p, then stretch so that q5 → -1 and q95 → +1.
        
        Args:
            image: Input array
            settings: Dict from fit()
        
        Returns:
            Normalized image, float32, approx in [-1, 1]
        """
        q5 = settings['q5']
        q95 = settings['q95']
        epsilon = settings['epsilon']
        
        log_img = np.log1p(image + epsilon)
        scale = q95 - q5
        scale = np.where(scale == 0, 1, scale)
        
        # Map q5 → -1, q95 → +1
        normalized = 2 * (log_img - q5) / scale - 1
        return np.float32(normalized)

class ZScoreNormalizer(BaseNormalizer):
    """Normalizer using z-score normalization (mean=0, std=1)."""
    
    def fit(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Compute mean and standard deviation for z-score normalization.
        
        Args:
            image: Input image array with shape (C, H, W) or (H, W)
            
        Returns:
            Dictionary with 'mean' and 'std' keys
        """
        if image.ndim == 2:
            # Single channel image
            mean = np.mean(image)
            std = np.std(image)
        else:
            # Multi-channel image, compute per channel
            mean = np.mean(image, axis=(1, 2), keepdims=True)
            std = np.std(image, axis=(1, 2), keepdims=True)
        
        return {
            'mean': mean,
            'std': std
        }
    
    def transform(self, image: np.ndarray, settings: Dict[str, Any]) -> np.ndarray:
        """
        Apply z-score normalization.
        
        Args:
            image: Input image array
            settings: Dictionary containing 'mean' and 'std' values
            
        Returns:
            Normalized image as float32
        """
        mean = settings['mean']
        std = settings['std']
        
        # Avoid division by zero
        std = np.where(std == 0, 1, std)
        
        normalized = (image - mean) / std
        return np.float32(normalized)


class MinMaxNormalizer(BaseNormalizer):
    """Normalizer using min-max normalization to [0, 1] range."""
    
    def fit(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Compute min and max values for min-max normalization.
        
        Args:
            image: Input image array with shape (C, H, W) or (H, W)
            
        Returns:
            Dictionary with 'min' and 'max' keys
        """
        if image.ndim == 2:
            # Single channel image
            min_val = np.min(image)
            max_val = np.max(image)
        else:
            # Multi-channel image, compute per channel
            min_val = np.min(image, axis=(1, 2), keepdims=True)
            max_val = np.max(image, axis=(1, 2), keepdims=True)
        
        return {
            'min': min_val,
            'max': max_val
        }
    
    def transform(self, image: np.ndarray, settings: Dict[str, Any]) -> np.ndarray:
        """
        Apply min-max normalization to scale image to [0, 1] range.
        
        Args:
            image: Input image array
            settings: Dictionary containing 'min' and 'max' values
            
        Returns:
            Normalized image as float32
        """
        min_val = settings['min']
        max_val = settings['max']
        
        # Avoid division by zero
        denominator = max_val - min_val
        denominator = np.where(denominator == 0, 1, denominator)
        
        normalized = (image - min_val) / denominator
        return np.float32(normalized)


class PercentileNormalizer(BaseNormalizer):
    """Normalizer using 100 percentiles (0-100) for distribution-based normalization."""
    
    def fit(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Compute 100 percentiles (0-100) for each channel.
        
        Args:
            image: Input image array with shape (C, H, W) or (H, W)
            
        Returns:
            Dictionary with 'percentiles' key containing the 101 percentile values (0-100)
        """
        percentile_values = np.linspace(0, 100, 11)  # 0, 1, 2, ..., 100
        
        if image.ndim == 2:
            # Single channel image
            percentiles = np.percentile(image, percentile_values)
        else:
            # Multi-channel image, compute per channel
            # Reshape to (C, H*W) for easier percentile computation
            reshaped = image.reshape(image.shape[0], -1)
            percentiles = np.percentile(reshaped, percentile_values, axis=1).T  # Shape: (C, 101)
        
        return {
            'percentiles': percentiles,
            'percentile_values': percentile_values
        }
    
    def transform(self, image: np.ndarray, settings: Dict[str, Any]) -> np.ndarray:
        """
        Apply percentile-based normalization.
        
        Values are mapped to their percentile position in the original distribution.
        For example, if a value is at the 25th percentile, it gets mapped to 0.25.
        
        Args:
            image: Input image array
            settings: Dictionary containing 'percentiles' array
            
        Returns:
            Normalized image as float32 with values between 0 and 1
        """
        percentiles = settings['percentiles']
        
        if image.ndim == 2:
            # Single channel image
            normalized = self._interpolate_percentiles(image, percentiles)
        else:
            # Multi-channel image
            normalized = np.zeros_like(image, dtype=np.float32)
            for c in range(image.shape[0]):
                normalized[c] = self._interpolate_percentiles(image[c], percentiles[c])
        
        # Scale to [-1, 1] range
        return np.float32(normalized*2-1)  
    
    def _interpolate_percentiles(self, channel_data: np.ndarray, channel_percentiles: np.ndarray) -> np.ndarray:
        """
        Interpolate values to their percentile positions.
        
        Args:
            channel_data: Single channel image data
            channel_percentiles: Percentile values for this channel (101 values for 0-100%)
            
        Returns:
            Normalized channel data
        """
        # Create percentile positions (0.0, 0.1, 0.2, ..., 1.0)
        percentile_positions = np.linspace(0, 1, len(channel_percentiles))
        
        # Use numpy's interp function to map values to percentile positions
        # Values below 0th percentile get 0.0, values above 100th percentile get 1.0
        normalized = np.interp(channel_data.flatten(), channel_percentiles, percentile_positions)
        
        return normalized.reshape(channel_data.shape)


class SmoothPercentileNormalizer(BaseNormalizer):
    """Normalizer using smooth interpolation curve: -1 at 5th percentile, 0 at 75th percentile, 1 at 100th percentile."""
    
    def fit(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Compute key percentiles for smooth normalization curve.
        
        Args:
            image: Input image array with shape (C, H, W) or (H, W)
            
        Returns:
            Dictionary with 'p5', 'p75', and 'p100' keys
        """
        if image.ndim == 2:
            # Single channel image
            p5 = np.percentile(image, 5)
            p75 = np.percentile(image, 75)
            p100 = np.percentile(image, 100)  # max value
        else:
            # Multi-channel image, compute per channel
            p5 = np.percentile(image, 5, axis=(1, 2), keepdims=True)
            p75 = np.percentile(image, 75, axis=(1, 2), keepdims=True)
            p100 = np.percentile(image, 100, axis=(1, 2), keepdims=True)
        
        return {
            'p5': p5,
            'p75': p75,
            'p100': p100
        }
    
    def transform(self, image: np.ndarray, settings: Dict[str, Any]) -> np.ndarray:
        """
        Apply smooth percentile normalization with custom curve.
        
        The mapping is:
        - p5 -> -1
        - p75 -> 0  
        - p100 -> 1
        
        Uses piecewise linear interpolation for smooth transitions.
        
        Args:
            image: Input image array
            settings: Dictionary containing 'p5', 'p75', 'p100' values
            
        Returns:
            Normalized image as float32
        """
        p5 = settings['p5']
        p75 = settings['p75']
        p100 = settings['p100']
        
        # Create normalized image
        normalized = np.zeros_like(image, dtype=np.float32)
        
        # Handle single channel vs multi-channel
        if image.ndim == 2:
            normalized = self._apply_smooth_curve(image, p5, p75, p100)
        else:
            for c in range(image.shape[0]):
                # Extract percentile values for this channel
                if isinstance(p5, np.ndarray) and p5.ndim > 0:
                    p5_c = p5[c] if p5.shape[0] > c else p5.flatten()[0]
                    p75_c = p75[c] if p75.shape[0] > c else p75.flatten()[0]
                    p100_c = p100[c] if p100.shape[0] > c else p100.flatten()[0]
                else:
                    p5_c = p5
                    p75_c = p75
                    p100_c = p100
                
                normalized[c] = self._apply_smooth_curve(image[c], p5_c, p75_c, p100_c)
        
        return np.float32(normalized)
    
    def _apply_smooth_curve(self, channel_data: np.ndarray, p5: float, p75: float, p100: float) -> np.ndarray:
        """
        Apply the smooth interpolation curve to a single channel.
        
        Args:
            channel_data: Single channel image data
            p5, p75, p100: Percentile values for this channel
            
        Returns:
            Normalized channel data
        """
        # Handle scalar values for percentiles (remove any extra dimensions)
        if isinstance(p5, np.ndarray):
            p5 = float(p5.item())
        if isinstance(p75, np.ndarray):
            p75 = float(p75.item())
        if isinstance(p100, np.ndarray):
            p100 = float(p100.item())
            
        # Avoid division by zero
        if p75 == p5:
            p75 = p5 + 1e-8
        if p100 == p75:
            p100 = p75 + 1e-8
            
        # Piecewise linear interpolation
        # For values <= p5: map to -1
        # For values between p5 and p75: map linearly from -1 to 0
        # For values between p75 and p100: map linearly from 0 to 1
        # For values > p100: clamp to 1
        
        normalized = np.zeros_like(channel_data, dtype=np.float32)
        
        # Below p5: set to -1
        mask_below = channel_data <= p5
        normalized[mask_below] = -1.0
        
        # Between p5 and p75: linear interpolation from -1 to 0
        mask_mid = (channel_data > p5) & (channel_data <= p75)
        if np.any(mask_mid):
            # Extract values and compute interpolation
            values_mid = channel_data[mask_mid]
            t = (values_mid - p5) / (p75 - p5)
            normalized[mask_mid] = -1.0 + t  # maps from -1 to 0
        
        # Between p75 and p100: linear interpolation from 0 to 1
        mask_high = (channel_data > p75) & (channel_data <= p100)
        if np.any(mask_high):
            # Extract values and compute interpolation
            values_high = channel_data[mask_high]
            t = (values_high - p75) / (p100 - p75)
            normalized[mask_high] = t  # maps from 0 to 1
        
        # Above p100: clamp to 1
        mask_above = channel_data > p100
        normalized[mask_above] = 1.0
        
        return normalized


class NormalizerFactory:
    """Factory class for creating normalizers."""
    
    _normalizers = {
        'q5_q95': QuantileNormalizer,
        'quantile': QuantileNormalizer,
        'z_norm': ZScoreNormalizer,
        'zscore': ZScoreNormalizer,
        'minmax': MinMaxNormalizer,
        'min_max': MinMaxNormalizer,
        'percentile': PercentileNormalizer,
        'percentile_normalizer': PercentileNormalizer,
        'smooth_percentile': SmoothPercentileNormalizer,
        'smooth': SmoothPercentileNormalizer,
        "log": LogNormalizer,
        "loq": LogQNormalizer,
    }
    
    @classmethod
    def create_normalizer(cls, method: str) -> BaseNormalizer:
        """
        Create a normalizer instance based on the method string.
        
        Args:
            method: String identifier for the normalization method
            
        Returns:
            Normalizer instance
            
        Raises:
            ValueError: If method is not recognized
        """
        method = method.lower()
        if method not in cls._normalizers:
            available_methods = list(cls._normalizers.keys())
            raise ValueError(f"Unknown normalization method '{method}'. "
                           f"Available methods: {available_methods}")
        
        return cls._normalizers[method]()
    
    @classmethod
    def get_available_methods(cls) -> list:
        """Get list of available normalization methods."""
        return list(cls._normalizers.keys())


def normalize_image_factory(image: np.ndarray, method: str = 'q5_q95', 
                          settings: Dict[str, Any] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Convenience function that combines fit and transform operations.
    
    Args:
        image: Input image array
        method: Normalization method string
        settings: Optional pre-computed settings. If None, will compute from image.
        
    Returns:
        Tuple of (normalized_image, settings_dict)
    """
    normalizer = NormalizerFactory.create_normalizer(method)
    
    if settings is None:
        settings = normalizer.fit(image)
    
    normalized_image = normalizer.transform(image, settings)
    
    return normalized_image, settings


def fit_normalizer(image: np.ndarray, method: str = 'q5_q95') -> Dict[str, Any]:
    """
    Fit a normalizer to an image and return the settings.
    
    Args:
        image: Input image array
        method: Normalization method string
        
    Returns:
        Dictionary containing normalization parameters
    """
    normalizer = NormalizerFactory.create_normalizer(method)
    return normalizer.fit(image)


def transform_image(image: np.ndarray, settings: Dict[str, Any], 
                   method: str = 'q5_q95') -> np.ndarray:
    """
    Transform an image using pre-computed normalization settings.
    
    Args:
        image: Input image array
        settings: Dictionary containing normalization parameters
        method: Normalization method string
        
    Returns:
        Normalized image array
    """
    normalizer = NormalizerFactory.create_normalizer(method)
    return normalizer.transform(image, settings)
