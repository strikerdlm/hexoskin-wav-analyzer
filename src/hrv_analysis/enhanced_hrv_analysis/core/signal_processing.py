"""
Enhanced Signal Processing for HRV Analysis

This module provides robust signal processing capabilities including:
- RR interval conversion and validation
- Artifact detection and correction
- Signal filtering and interpolation
- Outlier detection with multiple methods
"""

import numpy as np
import pandas as pd
from scipy import signal, interpolate
from scipy.stats import zscore
from typing import Optional, Tuple, Dict, Any, List
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ArtifactMethod(Enum):
    """Available artifact detection methods."""
    MALIK = "malik"
    KARLSSON = "karlsson" 
    KAMATH = "kamath"
    ACAR = "acar"
    IQR = "iqr"
    ZSCORE = "zscore"

class InterpolationMethod(Enum):
    """Available interpolation methods."""
    LINEAR = "linear"
    CUBIC = "cubic"
    SPLINE = "spline"
    QUADRATIC = "quadratic"

@dataclass 
class SignalQuality:
    """Signal quality assessment metrics."""
    total_beats: int
    valid_beats: int
    artifacts_detected: int
    artifact_ratio: float
    mean_rr: float
    std_rr: float
    rmssd: float
    nn50_count: int
    pnn50: float
    quality_score: float

class SignalProcessor:
    """Enhanced signal processor with comprehensive RR interval handling."""
    
    def __init__(self, 
                 min_rr: float = 300,
                 max_rr: float = 2000,
                 artifact_method: ArtifactMethod = ArtifactMethod.MALIK,
                 interp_method: InterpolationMethod = InterpolationMethod.CUBIC):
        """
        Initialize signal processor.
        
        Args:
            min_rr: Minimum physiologically plausible RR interval (ms)
            max_rr: Maximum physiologically plausible RR interval (ms) 
            artifact_method: Method for artifact detection
            interp_method: Method for interpolation
        """
        self.min_rr = min_rr
        self.max_rr = max_rr
        self.artifact_method = artifact_method
        self.interp_method = interp_method
        self.signal_quality = None
        
    def compute_rr_intervals(self, 
                           hr_data: pd.Series,
                           validate: bool = True,
                           clean_artifacts: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Convert heart rate to RR intervals with comprehensive processing.
        
        Args:
            hr_data: Heart rate data in BPM
            validate: Whether to validate physiological ranges
            clean_artifacts: Whether to detect and clean artifacts
            
        Returns:
            Tuple of (RR intervals in ms, processing metadata)
        """
        try:
            # Initial conversion
            rr_intervals, conversion_info = self._convert_hr_to_rr(hr_data, validate)
            
            if len(rr_intervals) == 0:
                logger.warning("No valid RR intervals could be computed")
                return np.array([]), conversion_info
                
            # Artifact detection and cleaning
            if clean_artifacts:
                rr_intervals, artifact_info = self._detect_and_clean_artifacts(rr_intervals)
                conversion_info.update(artifact_info)
                
            # Calculate signal quality metrics
            self.signal_quality = self._assess_signal_quality(rr_intervals)
            conversion_info['signal_quality'] = self.signal_quality.__dict__
            
            logger.info(f"RR processing complete: {len(rr_intervals)} intervals, "
                       f"quality score: {self.signal_quality.quality_score:.2f}")
                       
            return rr_intervals, conversion_info
            
        except Exception as e:
            logger.error(f"Error in RR interval computation: {e}")
            return np.array([]), {"error": str(e)}
            
    def _convert_hr_to_rr(self, 
                         hr_data: pd.Series, 
                         validate: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Convert heart rate to RR intervals."""
        try:
            # Clean and convert to numeric
            hr_clean = pd.to_numeric(hr_data, errors='coerce').dropna()
            
            if len(hr_clean) == 0:
                return np.array([]), {"error": "No valid heart rate data"}
                
            # Convert BPM to RR intervals (ms)
            rr_intervals = 60000.0 / hr_clean
            
            # Handle infinite values
            rr_intervals = rr_intervals.replace([np.inf, -np.inf], np.nan).dropna()
            
            conversion_info = {
                "original_hr_samples": len(hr_data),
                "valid_hr_samples": len(hr_clean),
                "initial_rr_count": len(rr_intervals),
                "hr_mean": float(hr_clean.mean()),
                "hr_std": float(hr_clean.std()),
                "hr_range": (float(hr_clean.min()), float(hr_clean.max()))
            }
            
            # Physiological validation
            if validate:
                valid_mask = (rr_intervals >= self.min_rr) & (rr_intervals <= self.max_rr)
                invalid_count = len(rr_intervals) - valid_mask.sum()
                
                if invalid_count > 0:
                    logger.warning(f"Filtered {invalid_count} physiologically implausible RR intervals")
                    
                rr_intervals = rr_intervals[valid_mask].values
                conversion_info["filtered_implausible"] = invalid_count
            else:
                rr_intervals = rr_intervals.values
                
            return rr_intervals, conversion_info
            
        except Exception as e:
            logger.error(f"Error converting HR to RR: {e}")
            return np.array([]), {"error": str(e)}
            
    def _detect_and_clean_artifacts(self, rr_intervals: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Detect and clean artifacts using selected method."""
        if len(rr_intervals) < 10:
            return rr_intervals, {"artifacts_detected": 0}
            
        try:
            if self.artifact_method == ArtifactMethod.MALIK:
                return self._malik_artifact_detection(rr_intervals)
            elif self.artifact_method == ArtifactMethod.KARLSSON:
                return self._karlsson_artifact_detection(rr_intervals)
            elif self.artifact_method == ArtifactMethod.KAMATH:
                return self._kamath_artifact_detection(rr_intervals)
            elif self.artifact_method == ArtifactMethod.IQR:
                return self._iqr_artifact_detection(rr_intervals)
            elif self.artifact_method == ArtifactMethod.ZSCORE:
                return self._zscore_artifact_detection(rr_intervals)
            else:
                return self._malik_artifact_detection(rr_intervals)  # Default
                
        except Exception as e:
            logger.error(f"Error in artifact detection: {e}")
            return rr_intervals, {"error": str(e)}
            
    def _malik_artifact_detection(self, rr_intervals: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Malik et al. (1996) artifact detection method.
        
        Removes RR intervals where:
        - RR_i differs by >20% from RR_(i-1) AND >20% from RR_(i+1)
        """
        if len(rr_intervals) < 3:
            return rr_intervals, {"artifacts_detected": 0}
            
        artifacts = np.zeros(len(rr_intervals), dtype=bool)
        
        for i in range(1, len(rr_intervals) - 1):
            prev_diff = abs(rr_intervals[i] - rr_intervals[i-1]) / rr_intervals[i-1]
            next_diff = abs(rr_intervals[i] - rr_intervals[i+1]) / rr_intervals[i+1]
            
            if prev_diff > 0.20 and next_diff > 0.20:
                artifacts[i] = True
                
        clean_rr = rr_intervals[~artifacts]
        artifact_count = artifacts.sum()
        
        logger.debug(f"Malik method detected {artifact_count} artifacts")
        
        return clean_rr, {
            "artifacts_detected": int(artifact_count),
            "artifact_method": "malik",
            "artifact_ratio": float(artifact_count / len(rr_intervals))
        }
        
    def _karlsson_artifact_detection(self, rr_intervals: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Karlsson et al. (2012) artifact detection method.
        
        More sophisticated method using local statistical properties.
        """
        if len(rr_intervals) < 5:
            return rr_intervals, {"artifacts_detected": 0}
            
        artifacts = np.zeros(len(rr_intervals), dtype=bool)
        window_size = min(5, len(rr_intervals))
        
        for i in range(window_size//2, len(rr_intervals) - window_size//2):
            # Local window statistics
            window = rr_intervals[i-window_size//2:i+window_size//2+1]
            local_median = np.median(window)
            local_mad = np.median(np.abs(window - local_median))
            
            # Threshold based on local variability
            threshold = max(0.15 * local_median, 3 * local_mad)
            
            if abs(rr_intervals[i] - local_median) > threshold:
                artifacts[i] = True
                
        clean_rr = rr_intervals[~artifacts]
        artifact_count = artifacts.sum()
        
        return clean_rr, {
            "artifacts_detected": int(artifact_count),
            "artifact_method": "karlsson",
            "artifact_ratio": float(artifact_count / len(rr_intervals))
        }
        
    def _kamath_artifact_detection(self, rr_intervals: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Kamath et al. (1993) artifact detection method.
        
        Based on physiological limits for RR interval changes.
        """
        if len(rr_intervals) < 2:
            return rr_intervals, {"artifacts_detected": 0}
            
        artifacts = np.zeros(len(rr_intervals), dtype=bool)
        
        # Maximum physiological change limits
        max_change_ms = 200  # Max change in ms
        max_change_percent = 0.25  # Max 25% change
        
        for i in range(1, len(rr_intervals)):
            change_ms = abs(rr_intervals[i] - rr_intervals[i-1])
            change_percent = change_ms / rr_intervals[i-1]
            
            if change_ms > max_change_ms and change_percent > max_change_percent:
                artifacts[i] = True
                
        clean_rr = rr_intervals[~artifacts]
        artifact_count = artifacts.sum()
        
        return clean_rr, {
            "artifacts_detected": int(artifact_count),
            "artifact_method": "kamath",
            "artifact_ratio": float(artifact_count / len(rr_intervals))
        }
        
    def _iqr_artifact_detection(self, rr_intervals: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """IQR-based outlier detection."""
        Q1 = np.percentile(rr_intervals, 25)
        Q3 = np.percentile(rr_intervals, 75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        artifacts = (rr_intervals < lower_bound) | (rr_intervals > upper_bound)
        clean_rr = rr_intervals[~artifacts]
        artifact_count = artifacts.sum()
        
        return clean_rr, {
            "artifacts_detected": int(artifact_count),
            "artifact_method": "iqr",
            "artifact_ratio": float(artifact_count / len(rr_intervals))
        }
        
    def _zscore_artifact_detection(self, rr_intervals: np.ndarray, threshold: float = 3.0) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Z-score based outlier detection."""
        z_scores = np.abs(zscore(rr_intervals))
        artifacts = z_scores > threshold
        
        clean_rr = rr_intervals[~artifacts]
        artifact_count = artifacts.sum()
        
        return clean_rr, {
            "artifacts_detected": int(artifact_count),
            "artifact_method": "zscore",
            "artifact_ratio": float(artifact_count / len(rr_intervals)),
            "zscore_threshold": threshold
        }
        
    def interpolate_rr_series(self, 
                             rr_intervals: np.ndarray,
                             target_fs: float = 4.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Interpolate RR intervals to regular sampling rate.
        
        Args:
            rr_intervals: RR intervals in ms
            target_fs: Target sampling frequency in Hz
            
        Returns:
            Tuple of (time_vector, interpolated_rr_intervals)
        """
        try:
            if len(rr_intervals) < 3:
                logger.warning("Insufficient data for interpolation")
                return np.array([]), np.array([])
                
            # Create time vector from cumulative RR intervals
            time_original = np.cumsum(rr_intervals) / 1000.0  # Convert to seconds
            time_original = np.insert(time_original[:-1], 0, 0)  # Start at 0
            
            # Create regular time grid
            time_regular = np.arange(0, time_original[-1], 1/target_fs)
            
            if len(time_regular) < 2:
                logger.warning("Insufficient time coverage for interpolation")
                return np.array([]), np.array([])
                
            # Select interpolation method
            if self.interp_method == InterpolationMethod.LINEAR:
                kind = 'linear'
            elif self.interp_method == InterpolationMethod.CUBIC:
                kind = 'cubic' if len(rr_intervals) >= 4 else 'linear'
            elif self.interp_method == InterpolationMethod.QUADRATIC:
                kind = 'quadratic' if len(rr_intervals) >= 3 else 'linear'
            else:
                kind = 'linear'
                
            # Perform interpolation
            f_interp = interpolate.interp1d(
                time_original, 
                rr_intervals[:-1], 
                kind=kind, 
                bounds_error=False, 
                fill_value='extrapolate'
            )
            
            rr_interpolated = f_interp(time_regular)
            
            logger.debug(f"Interpolated {len(rr_intervals)} RR intervals to {len(time_regular)} samples at {target_fs} Hz")
            
            return time_regular, rr_interpolated
            
        except Exception as e:
            logger.error(f"Error in RR interpolation: {e}")
            return np.array([]), np.array([])
            
    def apply_filter(self, 
                    rr_intervals: np.ndarray,
                    filter_type: str = 'savgol',
                    **filter_params) -> np.ndarray:
        """
        Apply filtering to RR interval series.
        
        Args:
            rr_intervals: RR intervals to filter
            filter_type: Type of filter ('savgol', 'butter', 'median')
            **filter_params: Filter-specific parameters
            
        Returns:
            Filtered RR intervals
        """
        try:
            if len(rr_intervals) < 5:
                return rr_intervals
                
            if filter_type == 'savgol':
                window_length = filter_params.get('window_length', 5)
                polyorder = filter_params.get('polyorder', 2)
                window_length = min(window_length, len(rr_intervals))
                if window_length % 2 == 0:
                    window_length -= 1
                polyorder = min(polyorder, window_length - 1)
                
                return signal.savgol_filter(rr_intervals, window_length, polyorder)
                
            elif filter_type == 'median':
                kernel_size = filter_params.get('kernel_size', 3)
                kernel_size = min(kernel_size, len(rr_intervals))
                return signal.medfilt(rr_intervals, kernel_size)
                
            elif filter_type == 'butter':
                # For Butterworth filter, need to interpolate first
                time_regular, rr_interpolated = self.interpolate_rr_series(rr_intervals)
                if len(rr_interpolated) == 0:
                    return rr_intervals
                    
                cutoff = filter_params.get('cutoff', 0.5)  # Hz
                order = filter_params.get('order', 4)
                fs = filter_params.get('fs', 4.0)
                
                nyquist = fs / 2
                normal_cutoff = cutoff / nyquist
                b, a = signal.butter(order, normal_cutoff, btype='low')
                
                return signal.filtfilt(b, a, rr_interpolated)
                
            else:
                logger.warning(f"Unknown filter type: {filter_type}")
                return rr_intervals
                
        except Exception as e:
            logger.error(f"Error applying filter: {e}")
            return rr_intervals
            
    def _assess_signal_quality(self, rr_intervals: np.ndarray) -> SignalQuality:
        """Assess overall signal quality."""
        try:
            if len(rr_intervals) == 0:
                return SignalQuality(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
                
            # Basic statistics
            mean_rr = np.mean(rr_intervals)
            std_rr = np.std(rr_intervals)
            
            # Time domain HRV metrics for quality assessment
            if len(rr_intervals) > 1:
                rr_diff = np.diff(rr_intervals)
                rmssd = np.sqrt(np.mean(rr_diff**2))
                nn50_count = np.sum(np.abs(rr_diff) > 50)
                pnn50 = (nn50_count / len(rr_diff)) * 100 if len(rr_diff) > 0 else 0
            else:
                rmssd = 0
                nn50_count = 0
                pnn50 = 0
                
            # Calculate quality score (0-1)
            quality_factors = []
            
            # Factor 1: Data completeness (length)
            min_beats_for_quality = 100
            length_factor = min(1.0, len(rr_intervals) / min_beats_for_quality)
            quality_factors.append(length_factor)
            
            # Factor 2: Physiological variability
            cv_rr = std_rr / mean_rr if mean_rr > 0 else 0
            variability_factor = min(1.0, cv_rr / 0.1)  # Normalize to 10% CV
            quality_factors.append(variability_factor)
            
            # Factor 3: RMSSD reasonableness (should be >10ms for healthy subjects)
            rmssd_factor = min(1.0, rmssd / 30)  # Normalize to 30ms
            quality_factors.append(rmssd_factor)
            
            quality_score = np.mean(quality_factors)
            
            return SignalQuality(
                total_beats=len(rr_intervals),
                valid_beats=len(rr_intervals),
                artifacts_detected=0,  # Updated by artifact detection
                artifact_ratio=0.0,    # Updated by artifact detection
                mean_rr=float(mean_rr),
                std_rr=float(std_rr),
                rmssd=float(rmssd),
                nn50_count=int(nn50_count),
                pnn50=float(pnn50),
                quality_score=float(quality_score)
            )
            
        except Exception as e:
            logger.error(f"Error assessing signal quality: {e}")
            return SignalQuality(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
            
    def get_processing_report(self) -> Dict[str, Any]:
        """Get comprehensive signal processing report."""
        if self.signal_quality is None:
            return {"error": "No signal processing performed yet"}
            
        return {
            "signal_quality": self.signal_quality.__dict__,
            "processing_config": {
                "min_rr": self.min_rr,
                "max_rr": self.max_rr,
                "artifact_method": self.artifact_method.value,
                "interpolation_method": self.interp_method.value
            },
            "quality_interpretation": self._interpret_quality_score()
        }
        
    def _interpret_quality_score(self) -> str:
        """Interpret the quality score."""
        if self.signal_quality is None:
            return "No quality assessment available"
            
        score = self.signal_quality.quality_score
        
        if score >= 0.8:
            return "Excellent signal quality"
        elif score >= 0.6:
            return "Good signal quality"
        elif score >= 0.4:
            return "Fair signal quality - consider preprocessing"
        elif score >= 0.2:
            return "Poor signal quality - preprocessing recommended"
        else:
            return "Very poor signal quality - may not be suitable for HRV analysis" 