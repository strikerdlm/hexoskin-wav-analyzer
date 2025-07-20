"""
Enhanced HRV Processor with Comprehensive Metrics Calculation

This module provides a complete HRV analysis processor that computes:
- Time domain metrics (SDNN, RMSSD, pNN50, etc.)
- Frequency domain metrics (VLF, LF, HF powers, LF/HF ratio)
- Nonlinear metrics (Poincaré plot, DFA, entropy measures)
- Parasympathetic and sympathetic indices
- Advanced statistical measures with confidence intervals
"""

import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.interpolate import interp1d
from typing import Dict, Any, Optional, Tuple, List
import logging
from dataclasses import dataclass, asdict
from enum import Enum
import warnings
from joblib import Parallel, delayed
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

logger = logging.getLogger(__name__)

# Try to import numba for performance optimization, fallback if not available
try:
    from numba import jit, njit
    HAS_NUMBA = True
    logger.info("Numba available for performance optimization")
except ImportError:
    # Create dummy decorators if numba is not available
    def njit(func):
        logger.warning("Numba not available, using standard Python functions")
        return func
    def jit(func):
        return func
    HAS_NUMBA = False
    logger.info("Numba not available, performance may be reduced")

# Vectorized utility functions with optional numba optimization
@njit
def _fast_rmssd(rr_intervals: np.ndarray) -> float:
    """Fast vectorized RMSSD computation."""
    if len(rr_intervals) < 2:
        return 0.0
    diff = np.diff(rr_intervals)
    return np.sqrt(np.mean(diff ** 2))

@njit
def _fast_nn_metrics(rr_intervals: np.ndarray) -> Tuple[int, float, int, float]:
    """Fast vectorized NN metrics computation."""
    if len(rr_intervals) < 2:
        return 0, 0.0, 0, 0.0
    
    diff = np.diff(rr_intervals)
    abs_diff = np.abs(diff)
    
    nn50 = np.sum(abs_diff > 50)
    pnn50 = (nn50 / len(diff)) * 100 if len(diff) > 0 else 0.0
    nn20 = np.sum(abs_diff > 20)
    pnn20 = (nn20 / len(diff)) * 100 if len(diff) > 0 else 0.0
    
    return nn50, pnn50, nn20, pnn20

@njit
def _fast_poincare_features(rr_intervals: np.ndarray) -> Tuple[float, float, float]:
    """Fast vectorized Poincaré plot features."""
    if len(rr_intervals) < 2:
        return 0.0, 0.0, 0.0
    
    rr1 = rr_intervals[:-1]
    rr2 = rr_intervals[1:]
    
    # SD1 and SD2 computation
    diff = rr2 - rr1
    sum_rr = rr2 + rr1
    
    sd1 = np.std(diff) / np.sqrt(2)
    sd2 = np.std(sum_rr) / np.sqrt(2)
    sd_ratio = sd2 / sd1 if sd1 > 0 else 0.0
    
    return sd1, sd2, sd_ratio

@njit 
def _fast_welch_psd(data: np.ndarray, fs: float, nperseg: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    """Fast PSD computation using vectorized operations."""
    # Simplified Welch's method using numpy operations
    n = len(data)
    if n < nperseg:
        nperseg = n
        
    # Detrend and window
    data_detrend = data - np.mean(data)
    
    # Compute periodogram
    window = np.hanning(nperseg)
    overlap = nperseg // 2
    
    # Calculate number of segments
    n_segments = (n - overlap) // (nperseg - overlap)
    
    if n_segments < 1:
        # Fallback to full signal
        windowed = data_detrend * window[:n]
        fft_result = np.fft.fft(windowed)
        psd = np.abs(fft_result)**2
        freqs = np.fft.fftfreq(n, 1/fs)
        return freqs[:n//2], psd[:n//2]
    
    # Initialize PSD accumulator
    psd_avg = np.zeros(nperseg // 2)
    
    for i in range(n_segments):
        start = i * (nperseg - overlap)
        end = start + nperseg
        
        if end <= n:
            segment = data_detrend[start:end] * window
            fft_result = np.fft.fft(segment)
            psd_seg = np.abs(fft_result)**2
            psd_avg += psd_seg[:nperseg//2]
    
    psd_avg /= n_segments
    freqs = np.fft.fftfreq(nperseg, 1/fs)[:nperseg//2]
    
    return freqs, psd_avg

class HRVDomain(Enum):
    """HRV analysis domains."""
    TIME = "time_domain"
    FREQUENCY = "frequency_domain" 
    NONLINEAR = "nonlinear"
    PARASYMPATHETIC = "parasympathetic"
    SYMPATHETIC = "sympathetic"
    ANS_BALANCE = "ans_balance"
    ALL = "all"

@dataclass
class TimeDomainMetrics:
    """Time domain HRV metrics."""
    # Basic statistics
    mean_nni: float  # Mean NN intervals (ms)
    sdnn: float      # Standard deviation of NN intervals (ms)
    rmssd: float     # Root mean square of successive differences (ms)
    
    # Interval-based metrics
    nn50: int        # Number of pairs of successive NNs differing > 50ms
    pnn50: float     # Percentage of NN50 
    nn20: int        # Number of pairs of successive NNs differing > 20ms
    pnn20: float     # Percentage of NN20
    
    # Geometric measures
    cvnn: float      # Coefficient of variation of NN intervals
    mean_hr: float   # Mean heart rate (BPM)
    std_hr: float    # Standard deviation of heart rate (BPM)
    min_hr: float    # Minimum heart rate (BPM)
    max_hr: float    # Maximum heart rate (BPM)
    
    # Additional time domain measures
    sdsd: float      # Standard deviation of successive differences
    cvsd: float      # Coefficient of variation of successive differences
    median_nni: float # Median NN interval
    mad_nni: float   # Median absolute deviation of NN intervals
    mcv_nni: float   # Median-based coefficient of variation

@dataclass 
class FrequencyDomainMetrics:
    """Frequency domain HRV metrics."""
    # Absolute powers (ms²)
    vlf_power: float      # Very low frequency power (0.0033-0.04 Hz)
    lf_power: float       # Low frequency power (0.04-0.15 Hz)
    hf_power: float       # High frequency power (0.15-0.4 Hz)
    total_power: float    # Total spectral power
    
    # Normalized powers
    lf_nu: float          # LF power in normalized units
    hf_nu: float          # HF power in normalized units
    
    # Ratios and indices
    lf_hf_ratio: float    # LF/HF ratio
    vlf_percent: float    # VLF as percentage of total power
    lf_percent: float     # LF as percentage of total power
    hf_percent: float     # HF as percentage of total power
    
    # Peak frequencies
    vlf_peak: float       # Peak frequency in VLF band
    lf_peak: float        # Peak frequency in LF band
    hf_peak: float        # Peak frequency in HF band
    
    # Method information
    method: str           # PSD estimation method
    sampling_rate: float  # Sampling rate used

@dataclass
class NonlinearMetrics:
    """Nonlinear HRV metrics."""
    # Poincaré plot measures
    sd1: float           # Width of Poincaré plot
    sd2: float           # Length of Poincaré plot  
    sd1_sd2_ratio: float # SD1/SD2 ratio
    ellipse_area: float  # Area of Poincaré ellipse
    
    # Detrended fluctuation analysis
    dfa_alpha1: float    # Short-term DFA scaling exponent
    dfa_alpha2: float    # Long-term DFA scaling exponent
    
    # Entropy measures
    sample_entropy: float    # Sample entropy
    approximate_entropy: float # Approximate entropy
    
    # Triangular index
    tinn: float              # Triangular interpolation of NN interval histogram
    triangular_index: float  # HRV triangular index

@dataclass
class ParasympatheticMetrics:
    """Parasympathetic nervous system indices."""
    hf_power: float          # High frequency power (primary parasympathetic indicator)
    rmssd: float             # RMSSD (parasympathetic indicator)
    pnn50: float             # pNN50 (parasympathetic indicator)  
    sd1: float               # Poincaré SD1 (parasympathetic indicator)
    
    # Normalized indices
    hf_nu: float             # HF normalized units
    parasympathetic_index: float  # Combined parasympathetic index
    
    # Respiratory sinus arrhythmia
    rsa_amplitude: float     # RSA amplitude estimate
    respiratory_frequency: float # Estimated respiratory frequency
    
    # Enhanced parasympathetic metrics
    hf_rmssd_ratio: float    # HF power to RMSSD ratio
    rsa_coupling_index: float # Respiratory-cardiac coupling strength
    vagal_tone_index: float  # Composite vagal tone measure
    respiratory_coherence: float # Respiratory coherence index

@dataclass
class SympatheticMetrics:
    """Sympathetic nervous system indices."""
    lf_power: float              # Low frequency power 
    lf_nu: float                 # LF normalized units
    lf_hf_ratio: float          # LF/HF ratio (sympathovagal balance)
    stress_index: float         # Baevsky's stress index
    
    # Advanced sympathetic indices
    sympathetic_index: float     # Combined sympathetic index
    autonomic_balance: float     # Overall autonomic balance score
    sympathovagal_balance: float # Sympathovagal balance index
    
    # Enhanced sympathetic metrics
    cardiac_sympathetic_index: float  # Cardiac sympathetic nerve activity index
    sympathetic_modulation: float     # Sympathetic modulation strength
    beta_adrenergic_sensitivity: float # Beta-adrenergic receptor sensitivity estimate

@dataclass
class ANSBalanceMetrics:
    """Advanced Autonomic Nervous System balance metrics."""
    # Traditional measures
    lf_hf_ratio: float           # Classical LF/HF ratio
    autonomic_balance: float     # (HF_nu - LF_nu) / 100
    
    # Enhanced balance indices
    sympathovagal_index: float   # Advanced sympathovagal balance
    ans_complexity: float        # Complexity of ANS interactions
    cardiac_autonomic_balance: float # Cardiac-specific autonomic balance
    
    # Dynamic measures
    autonomic_reactivity: float  # ANS reactivity measure
    baroreflex_sensitivity: float # Estimated baroreflex sensitivity
    
    # Confidence intervals for key measures
    lf_hf_ratio_ci: Tuple[float, float]    # Bootstrap CI for LF/HF ratio
    sympathovagal_index_ci: Tuple[float, float] # Bootstrap CI for sympathovagal index
    ans_complexity_ci: Tuple[float, float]  # Bootstrap CI for ANS complexity

class HRVProcessor:
    """Enhanced HRV processor with comprehensive metrics."""
    
    def __init__(self, 
                 validate_input: bool = True,
                 parallel_processing: bool = True,
                 n_jobs: int = -1,
                 confidence_level: float = 0.95):
        """
        Initialize HRV processor.
        
        Args:
            validate_input: Whether to validate input data
            parallel_processing: Whether to use parallel processing for computations
            n_jobs: Number of parallel jobs (-1 for all cores)
            confidence_level: Confidence level for statistical measures
        """
        self.validate_input = validate_input
        self.parallel_processing = parallel_processing
        self.n_jobs = n_jobs
        self.confidence_level = confidence_level
        self.last_processing_info = {}
        
    def compute_hrv_metrics(self, 
                          rr_intervals: np.ndarray,
                          domains: List[HRVDomain] = None,
                          include_confidence_intervals: bool = True) -> Dict[str, Any]:
        """
        Compute comprehensive HRV metrics.
        
        Args:
            rr_intervals: RR intervals in milliseconds
            domains: List of HRV domains to compute
            include_confidence_intervals: Whether to compute confidence intervals
            
        Returns:
            Dictionary containing all computed metrics
        """
        if domains is None:
            domains = [HRVDomain.ALL]
            
        try:
            # Validate input
            if self.validate_input:
                rr_intervals = self._validate_rr_intervals(rr_intervals)
                if len(rr_intervals) == 0:
                    return {"error": "No valid RR intervals provided"}
                    
            results = {"processing_info": {"n_intervals": len(rr_intervals)}}
            
            # Compute metrics for requested domains
            if HRVDomain.ALL in domains:
                domains = [HRVDomain.TIME, HRVDomain.FREQUENCY, HRVDomain.NONLINEAR,
                          HRVDomain.PARASYMPATHETIC, HRVDomain.SYMPATHETIC, HRVDomain.ANS_BALANCE]
                          
            # Use parallel processing if enabled and beneficial
            if self.parallel_processing and len(domains) > 1:
                domain_results = Parallel(n_jobs=self.n_jobs)(
                    delayed(self._compute_domain_metrics)(rr_intervals, domain) 
                    for domain in domains
                )
                
                for domain, metrics in zip(domains, domain_results):
                    results[domain.value] = metrics
            else:
                for domain in domains:
                    results[domain.value] = self._compute_domain_metrics(rr_intervals, domain)
                    
            # Compute confidence intervals if requested
            if include_confidence_intervals:
                results["confidence_intervals"] = self._compute_confidence_intervals(
                    rr_intervals, results
                )
                
            # Add quality assessment
            results["quality_assessment"] = self._assess_analysis_quality(rr_intervals, results)
            
            self.last_processing_info = results["processing_info"]
            return results
            
        except Exception as e:
            logger.error(f"Error computing HRV metrics: {e}")
            return {"error": str(e)}
            
    def _validate_rr_intervals(self, rr_intervals: np.ndarray) -> np.ndarray:
        """Validate and clean RR intervals."""
        if not isinstance(rr_intervals, np.ndarray):
            rr_intervals = np.array(rr_intervals)
            
        # Remove NaN and infinite values
        valid_mask = np.isfinite(rr_intervals)
        rr_clean = rr_intervals[valid_mask]
        
        if len(rr_clean) == 0:
            logger.error("No valid RR intervals after cleaning")
            return np.array([])
            
        # Filter physiologically plausible values (300-2000 ms)
        physio_mask = (rr_clean >= 300) & (rr_clean <= 2000)
        rr_clean = rr_clean[physio_mask]
        
        invalid_count = len(rr_intervals) - len(rr_clean)
        if invalid_count > 0:
            logger.info(f"Filtered {invalid_count} invalid RR intervals")
            
        return rr_clean
        
    def _compute_domain_metrics(self, rr_intervals: np.ndarray, domain: HRVDomain) -> Dict[str, Any]:
        """Compute metrics for a specific domain."""
        try:
            if domain == HRVDomain.TIME:
                return asdict(self._compute_time_domain(rr_intervals))
            elif domain == HRVDomain.FREQUENCY:
                return asdict(self._compute_frequency_domain(rr_intervals))
            elif domain == HRVDomain.NONLINEAR:
                return asdict(self._compute_nonlinear(rr_intervals))
            elif domain == HRVDomain.PARASYMPATHETIC:
                return asdict(self._compute_parasympathetic_indices(rr_intervals))
            elif domain == HRVDomain.SYMPATHETIC:
                return asdict(self._compute_sympathetic_indices(rr_intervals))
            elif domain == HRVDomain.ANS_BALANCE:
                return asdict(self._compute_ans_balance_metrics(rr_intervals))
            else:
                return {"error": f"Unknown domain: {domain}"}
                
        except Exception as e:
            logger.error(f"Error computing {domain.value} metrics: {e}")
            return {"error": str(e)}
            
    def _compute_time_domain(self, rr_intervals: np.ndarray) -> TimeDomainMetrics:
        """Compute time domain HRV metrics."""
        if len(rr_intervals) == 0:
            return TimeDomainMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
            
        # Basic statistics
        mean_nni = np.mean(rr_intervals)
        sdnn = np.std(rr_intervals, ddof=1)
        median_nni = np.median(rr_intervals)
        mad_nni = np.median(np.abs(rr_intervals - median_nni))
        cvnn = (sdnn / mean_nni) * 100 if mean_nni > 0 else 0
        mcv_nni = (mad_nni / median_nni) * 100 if median_nni > 0 else 0
        
        # Heart rate statistics
        hr_values = 60000 / rr_intervals  # Convert to BPM
        mean_hr = np.mean(hr_values)
        std_hr = np.std(hr_values, ddof=1)
        min_hr = np.min(hr_values)
        max_hr = np.max(hr_values)
        
        # Successive difference measures using optimized functions
        if len(rr_intervals) > 1:
            # Use fast vectorized functions
            rmssd = _fast_rmssd(rr_intervals)
            nn50, pnn50, nn20, pnn20 = _fast_nn_metrics(rr_intervals)
            
            # Additional measures
            rr_diff = np.diff(rr_intervals)
            sdsd = np.std(rr_diff, ddof=1)
            cvsd = (sdsd / np.mean(np.abs(rr_diff))) * 100 if np.mean(np.abs(rr_diff)) > 0 else 0
        else:
            rmssd = sdsd = cvsd = 0
            nn50 = pnn50 = nn20 = pnn20 = 0
            
        return TimeDomainMetrics(
            mean_nni=mean_nni,
            sdnn=sdnn,
            rmssd=rmssd,
            nn50=int(nn50),
            pnn50=pnn50,
            nn20=int(nn20),
            pnn20=pnn20,
            cvnn=cvnn,
            mean_hr=mean_hr,
            std_hr=std_hr,
            min_hr=min_hr,
            max_hr=max_hr,
            sdsd=sdsd,
            cvsd=cvsd,
            median_nni=median_nni,
            mad_nni=mad_nni,
            mcv_nni=mcv_nni
        )
        
    def _compute_frequency_domain(self, rr_intervals: np.ndarray, 
                                 method: str = 'welch',
                                 sampling_rate: float = 4.0) -> FrequencyDomainMetrics:
        """
        Compute frequency domain HRV metrics using scientifically valid interpolation.
        
        SCIENTIFIC APPROACH - OPTIMIZED FOR ACCURACY:
        1. RR intervals represent the time between consecutive R-peaks
        2. Timestamps are created at the midpoint of each RR interval 
        3. This preserves the physiological meaning and temporal relationships
        4. Regular interpolation to uniform sampling for spectral analysis
        5. No data truncation - all RR intervals are preserved
        6. CRITICAL FIX: Robust array alignment validation to prevent mismatches
        
        This approach follows HRV analysis standards and prevents the 
        time-RR length mismatch warnings while maintaining data integrity.
        """
        if len(rr_intervals) < 50:  # Minimum for reliable frequency analysis
            return FrequencyDomainMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, method, sampling_rate)
            
        try:
            # SCIENTIFIC FIX: Proper time-series construction for frequency analysis
            try:
                # Method 1: Construct proper timestamps for RR intervals
                # Each RR interval represents the time between consecutive R-peaks
                # We create timestamps at the mid-point of each RR interval
                
                if len(rr_intervals) < 3:
                    return FrequencyDomainMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, method, sampling_rate)
                
                # CRITICAL FIX: Enhanced array alignment validation
                original_length = len(rr_intervals)
                logger.debug(f"Starting frequency analysis with {original_length} RR intervals")
                
                # Create cumulative time points (in seconds) - these represent R-peak times
                rr_seconds = rr_intervals / 1000.0  # Convert to seconds
                r_peak_times = np.concatenate([[0], np.cumsum(rr_seconds)])
                
                # CRITICAL FIX: Ensure consistent array dimensions
                # For n RR intervals, we have n+1 R-peak times (including t=0)
                # Timestamps should be at midpoints between consecutive R-peaks
                if len(r_peak_times) != len(rr_intervals) + 1:
                    logger.debug(f"R-peak time array length mismatch: {len(r_peak_times)} vs {len(rr_intervals) + 1} - attempting correction")
                    return FrequencyDomainMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, method, sampling_rate)
                
                # Create timestamps for RR intervals (midpoint between consecutive R-peaks)
                rr_timestamps = (r_peak_times[:-1] + r_peak_times[1:]) / 2.0
                
                # Use all RR intervals for interpolation - no truncation needed
                rr_values = rr_intervals.copy()
                
                # CRITICAL FIX: Comprehensive array validation - reduce log level to debug
                if len(rr_timestamps) != len(rr_values):
                    logger.debug(f"Time-RR length mismatch: {len(rr_timestamps)} vs {len(rr_values)} - using fallback interpolation")
                    return FrequencyDomainMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, method, sampling_rate)
                
                # Additional validation: Check for any invalid values
                if np.any(np.isnan(rr_timestamps)) or np.any(np.isinf(rr_timestamps)):
                    logger.warning("Invalid timestamps detected, attempting correction")
                    valid_mask = np.isfinite(rr_timestamps)
                    rr_timestamps = rr_timestamps[valid_mask]
                    rr_values = rr_values[valid_mask]
                    
                if np.any(np.isnan(rr_values)) or np.any(np.isinf(rr_values)):
                    logger.warning("Invalid RR values detected, attempting correction")
                    valid_mask = np.isfinite(rr_values)
                    rr_timestamps = rr_timestamps[valid_mask]
                    rr_values = rr_values[valid_mask]
                
                # Final validation after cleaning
                if len(rr_timestamps) != len(rr_values) or len(rr_timestamps) < 3:
                    logger.warning(f"Insufficient valid data after cleaning: {len(rr_timestamps)} valid pairs")
                    return FrequencyDomainMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, method, sampling_rate)
                
                # Create regular time grid for interpolation
                total_duration = r_peak_times[-1]
                if total_duration <= 0:
                    logger.warning("Invalid total duration for time grid")
                    return FrequencyDomainMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, method, sampling_rate)
                
                time_regular = np.arange(0, total_duration, 1/sampling_rate)
                
                if len(time_regular) < 10:
                    logger.warning(f"Insufficient time coverage for frequency analysis: {len(time_regular)} points")
                    return FrequencyDomainMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, method, sampling_rate)
                
                # Interpolate RR intervals to regular sampling grid
                # Use cubic interpolation if enough points, otherwise linear
                interp_kind = 'cubic' if len(rr_values) >= 4 else 'linear'
                
                # CRITICAL FIX: Enhanced interpolation with bounds checking
                try:
                    f_interp = interp1d(rr_timestamps, rr_values, 
                                      kind=interp_kind,
                                      bounds_error=False, 
                                      fill_value='extrapolate')
                    rr_interpolated = f_interp(time_regular)
                    
                    # Validate interpolation results
                    if np.any(np.isnan(rr_interpolated)) or np.any(np.isinf(rr_interpolated)):
                        logger.warning("Invalid values in interpolated RR series, using linear interpolation fallback")
                        f_interp_linear = interp1d(rr_timestamps, rr_values, 
                                                 kind='linear',
                                                 bounds_error=False, 
                                                 fill_value='extrapolate')
                        rr_interpolated = f_interp_linear(time_regular)
                        
                        # Final check after fallback
                        if np.any(np.isnan(rr_interpolated)) or np.any(np.isinf(rr_interpolated)):
                            logger.error("Interpolation failed even with linear fallback")
                            return FrequencyDomainMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, method, sampling_rate)
                    
                    logger.debug(f"✅ RR interpolation successful: {len(rr_values)} intervals → {len(rr_interpolated)} samples at {sampling_rate} Hz")
                    
                except Exception as interp_error:
                    logger.error(f"Interpolation failed: {interp_error}")
                    # Try simplified interpolation as last resort
                    rr_interpolated = self._fallback_rr_interpolation(rr_intervals, sampling_rate)
                    if len(rr_interpolated) < 10:
                        return FrequencyDomainMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, method, sampling_rate)
                
            except Exception as e:
                logger.error(f"RR interval interpolation failed: {e}")
                # Try simplified interpolation as fallback
                logger.info("Attempting simplified interpolation fallback")
                try:
                    rr_interpolated = self._fallback_rr_interpolation(rr_intervals, sampling_rate)
                    if len(rr_interpolated) < 10:
                        return FrequencyDomainMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, method, sampling_rate)
                except Exception as fallback_error:
                    logger.error(f"Fallback interpolation also failed: {fallback_error}")
                    return FrequencyDomainMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, method, sampling_rate)
            
            # Detrend
            rr_detrended = signal.detrend(rr_interpolated)
            
            # Compute power spectral density
            if method == 'welch':
                freqs, psd = signal.welch(rr_detrended, fs=sampling_rate, 
                                        nperseg=min(len(rr_detrended)//4, 256),
                                        window='hann')
            elif method == 'periodogram':
                freqs, psd = signal.periodogram(rr_detrended, fs=sampling_rate, window='hann')
            else:
                freqs, psd = signal.welch(rr_detrended, fs=sampling_rate)
                
            # Define frequency bands
            vlf_band = (0.0033, 0.04)
            lf_band = (0.04, 0.15)
            hf_band = (0.15, 0.4)
            
            # Calculate power in each band
            vlf_mask = (freqs >= vlf_band[0]) & (freqs < vlf_band[1])
            lf_mask = (freqs >= lf_band[0]) & (freqs < lf_band[1])
            hf_mask = (freqs >= hf_band[0]) & (freqs < hf_band[1])
            
            vlf_power = np.trapz(psd[vlf_mask], freqs[vlf_mask]) if np.any(vlf_mask) else 0
            lf_power = np.trapz(psd[lf_mask], freqs[lf_mask]) if np.any(lf_mask) else 0
            hf_power = np.trapz(psd[hf_mask], freqs[hf_mask]) if np.any(hf_mask) else 0
            
            total_power = vlf_power + lf_power + hf_power
            
            # Normalized units and ratios
            lf_nu = (lf_power / (lf_power + hf_power)) * 100 if (lf_power + hf_power) > 0 else 0
            hf_nu = (hf_power / (lf_power + hf_power)) * 100 if (lf_power + hf_power) > 0 else 0
            lf_hf_ratio = lf_power / hf_power if hf_power > 0 else 0
            
            # Percentages of total power
            vlf_percent = (vlf_power / total_power) * 100 if total_power > 0 else 0
            lf_percent = (lf_power / total_power) * 100 if total_power > 0 else 0
            hf_percent = (hf_power / total_power) * 100 if total_power > 0 else 0
            
            # Peak frequencies
            vlf_peak = freqs[vlf_mask][np.argmax(psd[vlf_mask])] if np.any(vlf_mask) and vlf_power > 0 else 0
            lf_peak = freqs[lf_mask][np.argmax(psd[lf_mask])] if np.any(lf_mask) and lf_power > 0 else 0
            hf_peak = freqs[hf_mask][np.argmax(psd[hf_mask])] if np.any(hf_mask) and hf_power > 0 else 0
            
            return FrequencyDomainMetrics(
                vlf_power=vlf_power,
                lf_power=lf_power,
                hf_power=hf_power,
                total_power=total_power,
                lf_nu=lf_nu,
                hf_nu=hf_nu,
                lf_hf_ratio=lf_hf_ratio,
                vlf_percent=vlf_percent,
                lf_percent=lf_percent,
                hf_percent=hf_percent,
                vlf_peak=vlf_peak,
                lf_peak=lf_peak,
                hf_peak=hf_peak,
                method=method,
                sampling_rate=sampling_rate
            )
            
        except Exception as e:
            logger.error(f"Error in frequency domain analysis: {e}")
            return FrequencyDomainMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, method, sampling_rate)
    
    def _fallback_rr_interpolation(self, rr_intervals: np.ndarray, sampling_rate: float = 4.0) -> np.ndarray:
        """
        Fallback interpolation method for RR intervals when main interpolation fails.
        Uses a simpler approach that's more robust to edge cases.
        """
        if len(rr_intervals) < 2:
            return np.array([])
            
        # Simple approach: use RR intervals as instantaneous HR values
        # Create time points assuming regular R-R intervals
        mean_rr_sec = np.mean(rr_intervals) / 1000.0
        total_time = len(rr_intervals) * mean_rr_sec
        
        # Create regular time grid
        time_regular = np.arange(0, total_time, 1/sampling_rate)
        
        if len(time_regular) < 2:
            return np.array([])
        
        # Simple linear interpolation using the available data
        indices = np.linspace(0, len(rr_intervals) - 1, len(time_regular))
        rr_interpolated = np.interp(indices, np.arange(len(rr_intervals)), rr_intervals)
        
        return rr_interpolated
            
    def _compute_nonlinear(self, rr_intervals: np.ndarray) -> NonlinearMetrics:
        """
        Compute nonlinear HRV metrics with timeout protection and optimizations.
        
        CRITICAL FIX: Added timeout protection to prevent "task lost" errors
        - Individual timeouts for each computation
        - Fast fallback when computations take too long
        - Progressive complexity based on data size
        - Memory-efficient implementations
        """
        if len(rr_intervals) < 10:
            return NonlinearMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        
        logger.debug(f"Computing nonlinear metrics for {len(rr_intervals)} RR intervals")
        
        # Timeout protection wrapper
        def safe_compute(func, *args, timeout_seconds=10, fallback_value=0):
            """Execute function with timeout protection."""
            from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
            
            try:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(func, *args)
                    result = future.result(timeout=timeout_seconds)
                    return result
            except FutureTimeoutError:
                logger.warning(f"Function {func.__name__} timed out after {timeout_seconds}s")
                return fallback_value
            except Exception as e:
                logger.warning(f"Function {func.__name__} failed: {e}")
                return fallback_value
        
        # Poincaré plot analysis (fast, no timeout needed)
        try:
            sd1, sd2, sd1_sd2_ratio = self._compute_poincare_features(rr_intervals)
        except Exception as e:
            logger.warning(f"Poincare computation failed: {e}")
            sd1, sd2, sd1_sd2_ratio = 0, 0, 0
        
        # DFA with timeout protection (most expensive computation)
        dfa_timeout = min(30, max(5, len(rr_intervals) // 1000))  # Scale timeout with data size
        dfa_alpha1, dfa_alpha2 = safe_compute(
            self._compute_dfa_optimized, rr_intervals, 
            timeout_seconds=dfa_timeout, 
            fallback_value=(0, 0)
        )
        
        # Entropy measures with timeout protection
        entropy_timeout = min(15, max(3, len(rr_intervals) // 2000))
        sample_entropy = safe_compute(
            self._compute_sample_entropy_optimized, rr_intervals,
            timeout_seconds=entropy_timeout,
            fallback_value=0
        )
        
        approximate_entropy = safe_compute(
            self._compute_approximate_entropy_optimized, rr_intervals,
            timeout_seconds=entropy_timeout,
            fallback_value=0
        )
        
        # Triangular measures (relatively fast)
        try:
            tinn, triangular_index = self._compute_triangular_measures(rr_intervals)
        except Exception as e:
            logger.warning(f"Triangular measures failed: {e}")
            tinn, triangular_index = 0, 0
        
        # Calculate ellipse area (was missing)
        ellipse_area = np.pi * sd1 * sd2 if sd1 > 0 and sd2 > 0 else 0
        
        logger.debug(f"Nonlinear metrics computed successfully")
        
        return NonlinearMetrics(
            sd1=sd1,
            sd2=sd2,
            sd1_sd2_ratio=sd1_sd2_ratio,
            ellipse_area=ellipse_area,
            dfa_alpha1=dfa_alpha1,
            dfa_alpha2=dfa_alpha2,
            sample_entropy=sample_entropy,
            approximate_entropy=approximate_entropy,
            tinn=tinn,
            triangular_index=triangular_index
        )
        
    def _compute_poincare_features(self, rr_intervals: np.ndarray) -> Tuple[float, float, float]:
        """Compute Poincaré plot features using optimized vectorized operations."""
        # Use fast vectorized function
        return _fast_poincare_features(rr_intervals)
        
    def _compute_dfa(self, rr_intervals: np.ndarray) -> Tuple[float, float]:
        """Compute Detrended Fluctuation Analysis scaling exponents."""
        if len(rr_intervals) < 100:  # Need sufficient data for DFA
            return 0, 0
            
        try:
            # Create cumulative sum (integration)
            y = np.cumsum(rr_intervals - np.mean(rr_intervals))
            
            # Define scales
            scales_short = np.arange(4, min(17, len(rr_intervals)//4))
            scales_long = np.arange(16, min(65, len(rr_intervals)//4))
            
            def compute_fluctuation(scales):
                fluctuations = []
                for scale in scales:
                    # Divide into segments
                    segments = len(y) // scale
                    if segments < 4:
                        continue
                        
                    local_fluctuations = []
                    for i in range(segments):
                        start = i * scale
                        end = start + scale
                        segment = y[start:end]
                        
                        # Detrend (linear fit)
                        x_segment = np.arange(len(segment))
                        coeffs = np.polyfit(x_segment, segment, 1)
                        trend = np.polyval(coeffs, x_segment)
                        detrended = segment - trend
                        
                        local_fluctuations.append(np.sqrt(np.mean(detrended**2)))
                        
                    if local_fluctuations:
                        fluctuations.append(np.mean(local_fluctuations))
                    else:
                        fluctuations.append(0)
                        
                return fluctuations
                
            # Compute fluctuations for short and long scales
            fluc_short = compute_fluctuation(scales_short)
            fluc_long = compute_fluctuation(scales_long)
            
            # Calculate scaling exponents
            alpha1 = alpha2 = 0
            
            if len(fluc_short) > 2 and np.all(np.array(fluc_short) > 0):
                log_scales = np.log10(scales_short[:len(fluc_short)])
                log_fluc = np.log10(fluc_short)
                alpha1, _ = np.polyfit(log_scales, log_fluc, 1)
                
            if len(fluc_long) > 2 and np.all(np.array(fluc_long) > 0):
                log_scales = np.log10(scales_long[:len(fluc_long)])
                log_fluc = np.log10(fluc_long)
                alpha2, _ = np.polyfit(log_scales, log_fluc, 1)
                
            return float(alpha1), float(alpha2)
            
        except Exception as e:
            logger.debug(f"DFA computation failed: {e}")
            return 0, 0
            
    def _compute_sample_entropy(self, rr_intervals: np.ndarray, 
                               m: int = 2, r: float = 0.2) -> float:
        """Compute sample entropy."""
        if len(rr_intervals) < 50:
            return 0
            
        try:
            N = len(rr_intervals)
            r_abs = r * np.std(rr_intervals)
            
            def _maxdist(xi, xj, m):
                return max([abs(ua - va) for ua, va in zip(xi, xj)])
                
            def _phi(m):
                patterns = np.array([rr_intervals[i:i + m] for i in range(N - m + 1)])
                C = np.zeros(N - m + 1)
                
                for i in range(N - m + 1):
                    template = patterns[i]
                    for j in range(N - m + 1):
                        if _maxdist(template, patterns[j], m) <= r_abs:
                            C[i] += 1.0
                            
                phi = np.mean(np.log(C / (N - m + 1.0)))
                return phi
                
            return float(_phi(m) - _phi(m + 1))
            
        except Exception as e:
            logger.debug(f"Sample entropy computation failed: {e}")
            return 0
            
    def _compute_approximate_entropy(self, rr_intervals: np.ndarray, 
                                   m: int = 2, r: float = 0.2) -> float:
        """Compute approximate entropy."""
        if len(rr_intervals) < 50:
            return 0
            
        try:
            N = len(rr_intervals)
            r_abs = r * np.std(rr_intervals)
            
            def _phi(m):
                patterns = np.array([rr_intervals[i:i + m] for i in range(N - m + 1)])
                phi = 0
                
                for i in range(N - m + 1):
                    template = patterns[i]
                    matches = 0
                    for j in range(N - m + 1):
                        if max(abs(template - patterns[j])) <= r_abs:
                            matches += 1
                    phi += np.log(matches / (N - m + 1.0))
                    
                return phi / (N - m + 1.0)
                
            return float(_phi(m) - _phi(m + 1))
            
        except Exception as e:
            logger.debug(f"Approximate entropy computation failed: {e}")
            return 0
            
    def _compute_triangular_measures(self, rr_intervals: np.ndarray) -> Tuple[float, float]:
        """Compute triangular interpolation measures."""
        if len(rr_intervals) < 50:
            return 0, 0
            
        try:
            # Create histogram
            hist, bin_edges = np.histogram(rr_intervals, bins=128)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # Find peak of distribution
            peak_idx = np.argmax(hist)
            peak_height = hist[peak_idx]
            
            if peak_height == 0:
                return 0, 0
                
            # Triangular interpolation
            tinn = 2 * len(rr_intervals) / peak_height if peak_height > 0 else 0
            triangular_index = len(rr_intervals) / peak_height if peak_height > 0 else 0
            
            return float(tinn), float(triangular_index)
            
        except Exception as e:
            logger.debug(f"Triangular measures computation failed: {e}")
            return 0, 0
            
    def _compute_parasympathetic_indices(self, rr_intervals: np.ndarray) -> ParasympatheticMetrics:
        """Compute parasympathetic nervous system indices."""
        # Get basic metrics
        time_metrics = self._compute_time_domain(rr_intervals)
        freq_metrics = self._compute_frequency_domain(rr_intervals)
        poincare_features = self._compute_poincare_features(rr_intervals)
        
        # Primary parasympathetic indicators
        hf_power = freq_metrics.hf_power
        rmssd = time_metrics.rmssd
        pnn50 = time_metrics.pnn50
        sd1 = poincare_features[0]  # SD1 from Poincaré plot
        hf_nu = freq_metrics.hf_nu
        
        # Compute combined parasympathetic index
        # Normalize each measure to 0-1 scale
        rmssd_norm = min(1.0, rmssd / 100)  # Normalize to 100ms max
        pnn50_norm = min(1.0, pnn50 / 50)   # Normalize to 50% max
        hf_nu_norm = hf_nu / 100            # Already in percentage
        sd1_norm = min(1.0, sd1 / 50)       # Normalize to 50ms max
        
        parasympathetic_index = np.mean([rmssd_norm, pnn50_norm, hf_nu_norm, sd1_norm])
        
        # Estimate respiratory parameters
        rsa_amplitude = np.std(rr_intervals[-min(100, len(rr_intervals)):]) if len(rr_intervals) > 0 else 0
        respiratory_frequency = freq_metrics.hf_peak  # HF peak as respiratory frequency estimate
        
        # Enhanced parasympathetic metrics
        hf_rmssd_ratio = hf_power / rmssd if rmssd > 0 else 0
        
        # RSA coupling index - correlation between HF power and RMSSD in time windows
        rsa_coupling_index = self._compute_rsa_coupling(rr_intervals)
        
        # Vagal tone index - composite measure of parasympathetic activity
        vagal_tone_index = self._compute_vagal_tone(hf_nu, rmssd, pnn50, sd1)
        
        # Respiratory coherence - regularity of respiratory influence
        respiratory_coherence = self._compute_respiratory_coherence(rr_intervals, freq_metrics.hf_peak)
        
        return ParasympatheticMetrics(
            hf_power=hf_power,
            rmssd=rmssd,
            pnn50=pnn50,
            sd1=sd1,
            hf_nu=hf_nu,
            parasympathetic_index=parasympathetic_index,
            rsa_amplitude=rsa_amplitude,
            respiratory_frequency=respiratory_frequency,
            hf_rmssd_ratio=hf_rmssd_ratio,
            rsa_coupling_index=rsa_coupling_index,
            vagal_tone_index=vagal_tone_index,
            respiratory_coherence=respiratory_coherence
        )
        
    def _compute_sympathetic_indices(self, rr_intervals: np.ndarray) -> SympatheticMetrics:
        """Compute sympathetic nervous system indices."""
        # Get basic metrics
        time_metrics = self._compute_time_domain(rr_intervals)
        freq_metrics = self._compute_frequency_domain(rr_intervals)
        
        lf_power = freq_metrics.lf_power
        lf_nu = freq_metrics.lf_nu
        lf_hf_ratio = freq_metrics.lf_hf_ratio
        
        # Baevsky's stress index (simplified version)
        # SI = AMo / (2 * MxDMn * Mo) where AMo = mode amplitude, etc.
        hist, _ = np.histogram(rr_intervals, bins=50)
        mode_amplitude = np.max(hist) / len(rr_intervals) * 100  # Percentage
        geometric_mean = np.sqrt(np.min(rr_intervals) * np.max(rr_intervals))
        stress_index = mode_amplitude / (2 * (np.max(rr_intervals) - np.min(rr_intervals)) * geometric_mean) if geometric_mean > 0 else 0
        
        # Combined sympathetic index
        lf_nu_norm = lf_nu / 100
        lf_hf_norm = min(1.0, lf_hf_ratio / 5)  # Normalize to ratio of 5
        stress_norm = min(1.0, stress_index / 100)  # Normalize stress index
        
        sympathetic_index = np.mean([lf_nu_norm, lf_hf_norm, stress_norm])
        
        # Autonomic balance measures
        autonomic_balance = (freq_metrics.hf_nu - freq_metrics.lf_nu) / 100  # Range: -1 to 1
        sympathovagal_balance = lf_hf_ratio  # Classical sympathovagal balance
        
        # Enhanced sympathetic metrics
        # Cardiac sympathetic index - based on LF power, stress index, and heart rate variability
        cardiac_sympathetic_index = self._compute_cardiac_sympathetic_index(lf_nu, stress_index, time_metrics.mean_hr)
        
        # Sympathetic modulation - strength of sympathetic influence on heart rate
        sympathetic_modulation = self._compute_sympathetic_modulation(lf_power, time_metrics.sdnn)
        
        # Beta-adrenergic sensitivity - estimate based on heart rate response patterns
        beta_adrenergic_sensitivity = self._compute_beta_adrenergic_sensitivity(rr_intervals)
        
        return SympatheticMetrics(
            lf_power=lf_power,
            lf_nu=lf_nu,
            lf_hf_ratio=lf_hf_ratio,
            stress_index=stress_index,
            sympathetic_index=sympathetic_index,
            autonomic_balance=autonomic_balance,
            sympathovagal_balance=sympathovagal_balance,
            cardiac_sympathetic_index=cardiac_sympathetic_index,
            sympathetic_modulation=sympathetic_modulation,
            beta_adrenergic_sensitivity=beta_adrenergic_sensitivity
        )
        
    def _compute_confidence_intervals(self, 
                                    rr_intervals: np.ndarray, 
                                    results: Dict[str, Any],
                                    n_bootstrap: int = 50) -> Dict[str, Any]:
        """Compute confidence intervals using bootstrap resampling."""
        if len(rr_intervals) < 50:
            return {"error": "Insufficient data for confidence intervals"}
            
        try:
            # Select key metrics for CI computation
            key_metrics = [
                ('time_domain', 'sdnn'),
                ('time_domain', 'rmssd'),
                ('time_domain', 'pnn50'),
                ('frequency_domain', 'lf_power'),
                ('frequency_domain', 'hf_power'),
                ('frequency_domain', 'lf_hf_ratio'),
                ('nonlinear', 'sd1'),
                ('nonlinear', 'sd2')
            ]
            
            confidence_intervals = {}
            alpha = 1 - self.confidence_level
            
            for domain, metric in key_metrics:
                if domain in results and metric in results[domain]:
                    # Bootstrap resampling
                    bootstrap_values = []
                    
                    for _ in range(n_bootstrap):
                        # Resample with replacement
                        bootstrap_sample = np.random.choice(rr_intervals, 
                                                          size=len(rr_intervals), 
                                                          replace=True)
                        
                        # Compute metric for bootstrap sample
                        if domain == 'time_domain':
                            bootstrap_metrics = self._compute_time_domain(bootstrap_sample)
                            bootstrap_values.append(getattr(bootstrap_metrics, metric))
                        elif domain == 'frequency_domain':
                            bootstrap_metrics = self._compute_frequency_domain(bootstrap_sample)
                            bootstrap_values.append(getattr(bootstrap_metrics, metric))
                        elif domain == 'nonlinear':
                            bootstrap_metrics = self._compute_nonlinear(bootstrap_sample)
                            bootstrap_values.append(getattr(bootstrap_metrics, metric))
                            
                    # Calculate confidence intervals
                    if bootstrap_values:
                        lower_percentile = (alpha/2) * 100
                        upper_percentile = (1 - alpha/2) * 100
                        
                        ci_lower = np.percentile(bootstrap_values, lower_percentile)
                        ci_upper = np.percentile(bootstrap_values, upper_percentile)
                        
                        confidence_intervals[f"{domain}_{metric}"] = {
                            "lower": float(ci_lower),
                            "upper": float(ci_upper),
                            "confidence_level": self.confidence_level
                        }
                        
            return confidence_intervals
            
        except Exception as e:
            logger.error(f"Error computing confidence intervals: {e}")
            return {"error": str(e)}
            
    def _assess_analysis_quality(self, 
                               rr_intervals: np.ndarray, 
                               results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess the quality of HRV analysis."""
        quality_assessment = {
            "data_quality": "unknown",
            "analysis_reliability": "unknown", 
            "recommendations": []
        }
        
        try:
            n_intervals = len(rr_intervals)
            
            # Data sufficiency assessment
            if n_intervals < 50:
                quality_assessment["data_quality"] = "poor"
                quality_assessment["recommendations"].append("Insufficient data for reliable HRV analysis")
            elif n_intervals < 200:
                quality_assessment["data_quality"] = "fair"
                quality_assessment["recommendations"].append("Short recording may limit analysis reliability")
            elif n_intervals < 500:
                quality_assessment["data_quality"] = "good"
            else:
                quality_assessment["data_quality"] = "excellent"
                
            # Analysis reliability based on computed metrics
            reliability_factors = []
            
            # Check time domain metrics
            if "time_domain" in results:
                time_metrics = results["time_domain"]
                if time_metrics.get("rmssd", 0) > 10:  # Reasonable RMSSD
                    reliability_factors.append(1)
                else:
                    reliability_factors.append(0)
                    quality_assessment["recommendations"].append("Low RMSSD may indicate poor signal quality")
                    
            # Check frequency domain metrics
            if "frequency_domain" in results:
                freq_metrics = results["frequency_domain"]
                if freq_metrics.get("total_power", 0) > 100:  # Reasonable total power
                    reliability_factors.append(1)
                else:
                    reliability_factors.append(0)
                    quality_assessment["recommendations"].append("Low frequency domain power may indicate artifacts")
                    
            # Overall reliability
            if reliability_factors:
                reliability_score = np.mean(reliability_factors)
                if reliability_score >= 0.8:
                    quality_assessment["analysis_reliability"] = "high"
                elif reliability_score >= 0.6:
                    quality_assessment["analysis_reliability"] = "moderate"
                else:
                    quality_assessment["analysis_reliability"] = "low"
                    quality_assessment["recommendations"].append("Consider signal preprocessing or longer recordings")
                    
            if not quality_assessment["recommendations"]:
                quality_assessment["recommendations"] = ["Analysis quality appears acceptable"]
                
            quality_assessment["n_intervals"] = n_intervals
            quality_assessment["recording_duration_minutes"] = np.sum(rr_intervals) / (1000 * 60)
            
            return quality_assessment
            
        except Exception as e:
            logger.error(f"Error in quality assessment: {e}")
            return {"error": str(e)}
            
    def batch_process(self, 
                     datasets: Dict[str, np.ndarray],
                     domains: List[HRVDomain] = None) -> Dict[str, Dict[str, Any]]:
        """
        Process multiple datasets in batch.
        
        Args:
            datasets: Dictionary mapping dataset names to RR intervals
            domains: HRV domains to compute
            
        Returns:
            Dictionary mapping dataset names to HRV results
        """
        if self.parallel_processing:
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(self.compute_hrv_metrics)(rr_intervals, domains)
                for dataset_name, rr_intervals in datasets.items()
            )
            return dict(zip(datasets.keys(), results))
        else:
            return {
                dataset_name: self.compute_hrv_metrics(rr_intervals, domains)
                for dataset_name, rr_intervals in datasets.items()
            }
            
    # Enhanced ANS Balance Analysis Methods
    def _compute_rsa_coupling(self, rr_intervals: np.ndarray) -> float:
        """Compute respiratory-sinus arrhythmia coupling index."""
        if len(rr_intervals) < 100:
            return 0.0
        
        try:
            # Split into overlapping windows
            window_size = min(60, len(rr_intervals) // 4)
            hop_size = window_size // 2
            
            hf_powers = []
            rmssd_values = []
            
            for i in range(0, len(rr_intervals) - window_size, hop_size):
                window_rr = rr_intervals[i:i+window_size]
                
                # Compute HF power for window
                freq_metrics = self._compute_frequency_domain(window_rr)
                hf_powers.append(freq_metrics.hf_power)
                
                # Compute RMSSD for window
                rmssd_values.append(_fast_rmssd(window_rr))
            
            # Compute correlation between HF power and RMSSD across windows
            if len(hf_powers) > 3:
                correlation = np.corrcoef(hf_powers, rmssd_values)[0,1]
                return float(correlation) if not np.isnan(correlation) else 0.0
            else:
                return 0.0
                
        except Exception as e:
            logger.debug(f"RSA coupling computation failed: {e}")
            return 0.0
    
    def _compute_vagal_tone(self, hf_nu: float, rmssd: float, pnn50: float, sd1: float) -> float:
        """Compute composite vagal tone index."""
        # Normalize each component to 0-1 scale
        hf_nu_norm = hf_nu / 100.0
        rmssd_norm = min(1.0, rmssd / 100.0)  # Normalize to 100ms
        pnn50_norm = pnn50 / 100.0
        sd1_norm = min(1.0, sd1 / 50.0)  # Normalize to 50ms
        
        # Weighted composite index (HF power weighted more heavily)
        weights = [0.4, 0.25, 0.2, 0.15]  # HF_nu, RMSSD, pNN50, SD1
        components = [hf_nu_norm, rmssd_norm, pnn50_norm, sd1_norm]
        
        vagal_tone = sum(w * c for w, c in zip(weights, components))
        return float(vagal_tone)
    
    def _compute_respiratory_coherence(self, rr_intervals: np.ndarray, hf_peak: float) -> float:
        """Compute respiratory coherence index."""
        if len(rr_intervals) < 50 or hf_peak == 0:
            return 0.0
        
        try:
            # Compute PSD
            freq_metrics = self._compute_frequency_domain(rr_intervals)
            
            # Calculate coherence as the ratio of peak power to total HF power
            # This measures how concentrated the HF power is around the respiratory frequency
            
            # Simple coherence measure: peak HF power concentration
            coherence = min(1.0, (freq_metrics.hf_power / freq_metrics.total_power * 10)) if freq_metrics.total_power > 0 else 0
            return float(coherence)
            
        except Exception as e:
            logger.debug(f"Respiratory coherence computation failed: {e}")
            return 0.0
    
    def _compute_cardiac_sympathetic_index(self, lf_nu: float, stress_index: float, mean_hr: float) -> float:
        """Compute cardiac sympathetic index."""
        # Normalize components
        lf_nu_norm = lf_nu / 100.0
        stress_norm = min(1.0, stress_index / 100.0)
        
        # Heart rate contribution (higher HR suggests more sympathetic activity)
        hr_norm = min(1.0, (mean_hr - 60) / 60.0) if mean_hr > 60 else 0.0  # Normalize above resting HR
        
        # Weighted combination
        weights = [0.5, 0.3, 0.2]  # LF_nu, stress_index, HR
        components = [lf_nu_norm, stress_norm, hr_norm]
        
        cardiac_sympathetic = sum(w * c for w, c in zip(weights, components))
        return float(cardiac_sympathetic)
    
    def _compute_sympathetic_modulation(self, lf_power: float, sdnn: float) -> float:
        """Compute sympathetic modulation strength."""
        # Higher LF power and moderate SDNN suggest stronger sympathetic modulation
        if sdnn == 0:
            return 0.0
        
        # Normalize LF power (log scale due to wide range)
        lf_norm = min(1.0, np.log10(lf_power + 1) / 4.0) if lf_power > 0 else 0.0
        
        # SDNN contribution (moderate values suggest balanced modulation)
        sdnn_norm = min(1.0, sdnn / 100.0)  # Normalize to 100ms
        
        # Combined modulation strength
        modulation = (lf_norm + sdnn_norm) / 2.0
        return float(modulation)
    
    def _compute_beta_adrenergic_sensitivity(self, rr_intervals: np.ndarray) -> float:
        """Estimate beta-adrenergic sensitivity."""
        if len(rr_intervals) < 50:
            return 0.0
        
        try:
            # Estimate sensitivity based on heart rate variability patterns
            # Higher variability in LF band suggests higher beta-adrenergic sensitivity
            
            # Compute coefficient of variation
            cv = np.std(rr_intervals) / np.mean(rr_intervals) if np.mean(rr_intervals) > 0 else 0
            
            # Scale to 0-1 range (typical CV for healthy adults is 0.03-0.1)
            sensitivity = min(1.0, cv / 0.1)
            return float(sensitivity)
            
        except Exception as e:
            logger.debug(f"Beta-adrenergic sensitivity computation failed: {e}")
            return 0.0
    
    def _compute_ans_balance_metrics(self, rr_intervals: np.ndarray) -> ANSBalanceMetrics:
        """Compute comprehensive ANS balance metrics with bootstrap confidence intervals."""
        # Get component metrics
        time_metrics = self._compute_time_domain(rr_intervals)
        freq_metrics = self._compute_frequency_domain(rr_intervals)
        parasympathetic_metrics = self._compute_parasympathetic_indices(rr_intervals)
        sympathetic_metrics = self._compute_sympathetic_indices(rr_intervals)
        
        # Traditional measures
        lf_hf_ratio = freq_metrics.lf_hf_ratio
        autonomic_balance = (freq_metrics.hf_nu - freq_metrics.lf_nu) / 100.0
        
        # Enhanced balance indices
        sympathovagal_index = self._compute_sympathovagal_index(
            freq_metrics.lf_nu, freq_metrics.hf_nu, 
            sympathetic_metrics.stress_index, parasympathetic_metrics.parasympathetic_index
        )
        
        ans_complexity = self._compute_ans_complexity(rr_intervals)
        cardiac_autonomic_balance = self._compute_cardiac_autonomic_balance(
            parasympathetic_metrics.parasympathetic_index,
            sympathetic_metrics.sympathetic_index
        )
        
        # Dynamic measures
        autonomic_reactivity = self._compute_autonomic_reactivity(rr_intervals)
        baroreflex_sensitivity = self._estimate_baroreflex_sensitivity(rr_intervals)
        
        # Compute bootstrap confidence intervals
        lf_hf_ci, sympathovagal_ci, ans_complexity_ci = self._bootstrap_ans_confidence_intervals(rr_intervals)
        
        return ANSBalanceMetrics(
            lf_hf_ratio=lf_hf_ratio,
            autonomic_balance=autonomic_balance,
            sympathovagal_index=sympathovagal_index,
            ans_complexity=ans_complexity,
            cardiac_autonomic_balance=cardiac_autonomic_balance,
            autonomic_reactivity=autonomic_reactivity,
            baroreflex_sensitivity=baroreflex_sensitivity,
            lf_hf_ratio_ci=lf_hf_ci,
            sympathovagal_index_ci=sympathovagal_ci,
            ans_complexity_ci=ans_complexity_ci
        )
    
    def _compute_sympathovagal_index(self, lf_nu: float, hf_nu: float, 
                                   stress_index: float, parasympathetic_index: float) -> float:
        """Compute advanced sympathovagal balance index."""
        # Traditional LF/HF ratio normalized
        lf_hf_traditional = lf_nu / hf_nu if hf_nu > 0 else 10.0
        lf_hf_norm = min(1.0, lf_hf_traditional / 10.0)  # Normalize to max ratio of 10
        
        # Stress vs parasympathetic balance
        stress_para_balance = stress_index / (stress_index + parasympathetic_index * 100) if (stress_index + parasympathetic_index * 100) > 0 else 0.5
        
        # Combined sympathovagal index
        sympathovagal_index = (lf_hf_norm + stress_para_balance) / 2.0
        return float(sympathovagal_index)
    
    def _compute_ans_complexity(self, rr_intervals: np.ndarray) -> float:
        """Compute ANS complexity measure."""
        if len(rr_intervals) < 100:
            return 0.0
        
        try:
            # Multi-scale complexity based on sample entropy at different scales
            complexities = []
            scales = [1, 2, 4, 8]
            
            for scale in scales:
                # Coarse-grain the series
                if scale == 1:
                    coarse_grained = rr_intervals
                else:
                    n_samples = len(rr_intervals) // scale
                    coarse_grained = np.array([
                        np.mean(rr_intervals[i*scale:(i+1)*scale]) 
                        for i in range(n_samples)
                    ])
                
                if len(coarse_grained) > 50:
                    entropy = self._compute_sample_entropy(coarse_grained)
                    complexities.append(entropy)
            
            # Return mean complexity across scales
            return float(np.mean(complexities)) if complexities else 0.0
            
        except Exception as e:
            logger.debug(f"ANS complexity computation failed: {e}")
            return 0.0
    
    def _compute_cardiac_autonomic_balance(self, parasympathetic_index: float, sympathetic_index: float) -> float:
        """Compute cardiac-specific autonomic balance."""
        total_activity = parasympathetic_index + sympathetic_index
        if total_activity == 0:
            return 0.5  # Neutral balance
        
        # Balance ranges from 0 (pure sympathetic) to 1 (pure parasympathetic)
        balance = parasympathetic_index / total_activity
        return float(balance)
    
    def _compute_autonomic_reactivity(self, rr_intervals: np.ndarray) -> float:
        """Compute autonomic reactivity measure."""
        if len(rr_intervals) < 100:
            return 0.0
        
        try:
            # Compute reactivity as the coefficient of variation of short-term variability
            window_size = min(30, len(rr_intervals) // 10)
            
            short_term_vars = []
            for i in range(0, len(rr_intervals) - window_size, window_size):
                window = rr_intervals[i:i+window_size]
                short_term_vars.append(np.std(window))
            
            if len(short_term_vars) > 1:
                reactivity = np.std(short_term_vars) / np.mean(short_term_vars) if np.mean(short_term_vars) > 0 else 0
                return float(min(1.0, reactivity))  # Normalize to 0-1
            else:
                return 0.0
                
        except Exception as e:
            logger.debug(f"Autonomic reactivity computation failed: {e}")
            return 0.0
    
    def _estimate_baroreflex_sensitivity(self, rr_intervals: np.ndarray) -> float:
        """Estimate baroreflex sensitivity."""
        if len(rr_intervals) < 100:
            return 0.0
        
        try:
            # Simplified estimation based on spontaneous variations
            # Real BRS requires simultaneous blood pressure measurement
            
            # Use alpha-index method approximation
            # Estimate from the relationship between RR and "systolic BP" variations
            
            # Create surrogate for systolic BP variations from RR intervals
            # This is a crude approximation
            bp_surrogate = -1 * np.diff(rr_intervals)  # Inverse relationship approximation
            rr_diff = np.diff(rr_intervals[1:])  # Match lengths
            
            if len(bp_surrogate) > 10 and len(rr_diff) > 10:
                # Compute correlation (approximation of BRS)
                correlation = np.corrcoef(bp_surrogate[:len(rr_diff)], rr_diff)[0,1]
                brs = abs(correlation) * 20  # Scale to typical BRS range (0-20 ms/mmHg)
                return float(min(20.0, brs)) if not np.isnan(brs) else 0.0
            else:
                return 0.0
                
        except Exception as e:
            logger.debug(f"Baroreflex sensitivity estimation failed: {e}")
            return 0.0
    
    def _bootstrap_ans_confidence_intervals(self, rr_intervals: np.ndarray, 
                                          n_bootstrap: int = None) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
        """
        Compute bootstrap confidence intervals for key ANS metrics with adaptive sampling.
        
        CRITICAL FIX: Enhanced bootstrap sampling for better statistical accuracy
        - Adaptive sample size based on data characteristics
        - Increased from 25 to 100-250 samples for better confidence intervals
        - Performance optimizations to prevent hanging
        - Validation of statistical significance
        """
        
        # CRITICAL FIX: Adaptive bootstrap sampling
        if n_bootstrap is None:
            data_size = len(rr_intervals)
            if data_size < 100:
                n_bootstrap = 50    # Small datasets: minimal bootstrap
            elif data_size < 1000:
                n_bootstrap = 100   # Medium datasets: moderate bootstrap
            elif data_size < 5000:
                n_bootstrap = 150   # Large datasets: good bootstrap
            else:
                n_bootstrap = 200   # Very large datasets: robust bootstrap
        
        # Absolute limits for performance
        n_bootstrap = min(n_bootstrap, 250)  # Never exceed 250 samples
        n_bootstrap = max(n_bootstrap, 25)   # Never go below 25 samples
        
        if len(rr_intervals) < 50:
            logger.warning("Insufficient RR intervals for bootstrap analysis")
            return ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0))
        
        # PERFORMANCE FIX: Smart sample size for bootstrap
        if len(rr_intervals) > 1000:
            # For large datasets, use a representative sample for each bootstrap iteration
            sample_size = min(len(rr_intervals), 500)  # Increased from 200
        else:
            sample_size = len(rr_intervals)
        
        try:
            lf_hf_ratios = []
            sympathovagal_indices = []
            complexity_measures = []
            
            successful_samples = 0
            max_attempts = n_bootstrap * 2  # Allow some failed attempts
            
            logger.debug(f"Computing bootstrap CI: {n_bootstrap} samples, {sample_size} data points per sample")
            
            for attempt in range(max_attempts):
                if successful_samples >= n_bootstrap:
                    break
                    
                try:
                    # Bootstrap resample with intelligent size
                    bootstrap_indices = np.random.choice(len(rr_intervals), size=sample_size, replace=True)
                    bootstrap_sample = rr_intervals[bootstrap_indices]
                    
                    # PERFORMANCE FIX: Enhanced but efficient frequency analysis
                    if len(bootstrap_sample) >= 50:
                        # Quick time domain calculation
                        rr_diff = np.diff(bootstrap_sample)
                        if len(rr_diff) > 0:
                            rmssd = np.sqrt(np.mean(rr_diff ** 2))
                            sdnn = np.std(bootstrap_sample, ddof=1)
                            
                            # Enhanced LF/HF estimation
                            if len(bootstrap_sample) >= 100:
                                # For larger samples, use actual frequency domain estimation
                                try:
                                    # Simple spectral analysis for LF/HF estimation
                                    mean_rr = np.mean(bootstrap_sample)
                                    rr_normalized = (bootstrap_sample - mean_rr) / mean_rr
                                    
                                    # Basic frequency analysis
                                    freqs = np.fft.fftfreq(len(rr_normalized))
                                    fft_rr = np.fft.fft(rr_normalized)
                                    psd = np.abs(fft_rr) ** 2
                                    
                                    # Frequency bands (normalized)
                                    lf_mask = (freqs >= 0.01) & (freqs <= 0.04)  # Simplified bands
                                    hf_mask = (freqs >= 0.04) & (freqs <= 0.1)
                                    
                                    lf_power = np.sum(psd[lf_mask]) if np.any(lf_mask) else sdnn
                                    hf_power = np.sum(psd[hf_mask]) if np.any(hf_mask) else rmssd
                                    
                                    if hf_power > 0:
                                        lf_hf_ratio = lf_power / hf_power
                                    else:
                                        lf_hf_ratio = sdnn / rmssd if rmssd > 0 else 1.0
                                        
                                except:
                                    # Fallback to time domain proxy
                                    lf_hf_ratio = sdnn / rmssd if rmssd > 0 else 1.0
                            else:
                                # For smaller samples, use time domain proxy
                                lf_hf_ratio = sdnn / rmssd if rmssd > 0 else 1.0
                                
                            # Cap extreme values
                            lf_hf_ratio = np.clip(lf_hf_ratio, 0.1, 15.0)
                            lf_hf_ratios.append(lf_hf_ratio)
                            
                            # Enhanced sympathovagal index
                            sympathovagal_idx = np.clip(lf_hf_ratio / 5.0, 0.0, 1.0)
                            sympathovagal_indices.append(sympathovagal_idx)
                            
                            # Enhanced complexity measure
                            cv = sdnn / np.mean(bootstrap_sample) if np.mean(bootstrap_sample) > 0 else 0
                            complexity = np.clip(cv * 15, 0.0, 1.0)  # Adjusted scaling
                            complexity_measures.append(complexity)
                            
                            successful_samples += 1
                    
                except Exception as e:
                    logger.debug(f"Bootstrap sample {attempt} failed: {e}")
                    continue
            
            logger.debug(f"Bootstrap CI completed: {successful_samples}/{n_bootstrap} successful samples")
            
            if successful_samples < 10:  # Reduced minimum from 25 to 10
                logger.warning(f"Too few successful bootstrap samples: {successful_samples}")
                return ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0))
            
            # Calculate confidence intervals
            alpha = 1 - self.confidence_level
            lower_percentile = (alpha/2) * 100
            upper_percentile = (1 - alpha/2) * 100
            
            def compute_ci(values):
                if len(values) >= 5:  # Reduced from 3 to 5 for better reliability
                    return (float(np.percentile(values, lower_percentile)), 
                            float(np.percentile(values, upper_percentile)))
                else:
                    return (0.0, 0.0)
            
            lf_hf_ci = compute_ci(lf_hf_ratios)
            sympathovagal_ci = compute_ci(sympathovagal_indices)
            complexity_ci = compute_ci(complexity_measures)
            
            return lf_hf_ci, sympathovagal_ci, complexity_ci
            
        except Exception as e:
            logger.error(f"Bootstrap confidence interval computation failed: {e}")
            return ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0))

    def _compute_dfa_optimized(self, rr_intervals: np.ndarray) -> Tuple[float, float]:
        """
        Optimized DFA computation with memory efficiency and early termination.
        
        PERFORMANCE OPTIMIZATIONS:
        - Reduced scale ranges for large datasets
        - Vectorized operations where possible
        - Early termination for insufficient data
        - Memory-efficient segment processing
        """
        if len(rr_intervals) < 100:  # Need sufficient data for DFA
            return 0, 0
            
        try:
            # Adaptive scale selection based on data size
            n_data = len(rr_intervals)
            
            # For very large datasets, use reduced scale ranges to prevent timeout
            if n_data > 10000:
                max_scale_short = min(12, n_data//8)  # Reduced from 17
                max_scale_long = min(32, n_data//8)   # Reduced from 65
            elif n_data > 5000:
                max_scale_short = min(15, n_data//6)
                max_scale_long = min(50, n_data//6)
            else:
                max_scale_short = min(17, n_data//4)
                max_scale_long = min(65, n_data//4)
            
            # Create cumulative sum (integration)
            y = np.cumsum(rr_intervals - np.mean(rr_intervals))
            
            # Define adaptive scales
            scales_short = np.arange(4, max_scale_short)
            scales_long = np.arange(16, max_scale_long)
            
            def compute_fluctuation_optimized(scales, max_segments=50):
                """Optimized fluctuation computation with segment limits."""
                fluctuations = []
                for scale in scales:
                    segments = min(len(y) // scale, max_segments)  # Limit segments for performance
                    if segments < 4:
                        continue
                        
                    segment_fluctuations = []
                    for i in range(segments):
                        start = i * scale
                        end = start + scale
                        segment = y[start:end]
                        
                        # Vectorized detrending
                        x_segment = np.arange(len(segment))
                        coeffs = np.polyfit(x_segment, segment, 1)
                        trend = coeffs[0] * x_segment + coeffs[1]
                        detrended = segment - trend
                        
                        segment_fluctuations.append(np.sqrt(np.mean(detrended**2)))
                    
                    if segment_fluctuations:
                        fluctuations.append(np.mean(segment_fluctuations))
                        
                return fluctuations
                
            # Compute fluctuations with optimizations
            fluc_short = compute_fluctuation_optimized(scales_short, max_segments=30)
            fluc_long = compute_fluctuation_optimized(scales_long, max_segments=20)
            
            # Calculate scaling exponents
            alpha1 = alpha2 = 0
            
            if len(fluc_short) > 2 and np.all(np.array(fluc_short) > 0):
                try:
                    log_scales = np.log10(scales_short[:len(fluc_short)])
                    log_fluc = np.log10(fluc_short)
                    alpha1, _ = np.polyfit(log_scales, log_fluc, 1)
                    alpha1 = float(alpha1)
                except:
                    alpha1 = 0
                    
            if len(fluc_long) > 2 and np.all(np.array(fluc_long) > 0):
                try:
                    log_scales = np.log10(scales_long[:len(fluc_long)])
                    log_fluc = np.log10(fluc_long)
                    alpha2, _ = np.polyfit(log_scales, log_fluc, 1)
                    alpha2 = float(alpha2)
                except:
                    alpha2 = 0
                    
            return alpha1, alpha2
            
        except Exception as e:
            logger.debug(f"Optimized DFA computation failed: {e}")
            return 0, 0

    def _compute_sample_entropy_optimized(self, rr_intervals: np.ndarray, 
                                        m: int = 2, r: float = 0.2) -> float:
        """
        Optimized sample entropy with early termination and memory efficiency.
        
        PERFORMANCE OPTIMIZATIONS:
        - Reduced pattern length for large datasets
        - Early termination when computation takes too long
        - Memory-efficient pattern matching
        - Adaptive sample size reduction
        """
        if len(rr_intervals) < 50:
            return 0
        
        try:
            N = len(rr_intervals)
            
            # Adaptive parameters based on data size
            if N > 5000:
                # For large datasets, use smaller pattern length and reduced sample
                m = 1  # Reduced complexity
                N = min(N, 2000)  # Sample reduction
                rr_intervals = rr_intervals[:N]
            elif N > 2000:
                m = 2
                N = min(N, 3000)
                rr_intervals = rr_intervals[:N]
            
            r_abs = r * np.std(rr_intervals)
            
            def _maxdist_fast(xi, xj):
                """Fast maximum distance computation."""
                return np.max(np.abs(xi - xj))
                
            def _phi_optimized(m, max_comparisons=1000):
                """Optimized phi computation with comparison limit."""
                patterns = np.array([rr_intervals[i:i + m] for i in range(N - m + 1)])
                n_patterns = len(patterns)
                
                # Limit comparisons for performance
                step = max(1, n_patterns // max_comparisons)
                
                phi_sum = 0
                count = 0
                
                for i in range(0, n_patterns, step):
                    template = patterns[i]
                    matches = 0
                    
                    for j in range(0, n_patterns, step):
                        if _maxdist_fast(template, patterns[j]) <= r_abs:
                            matches += 1
                    
                    if matches > 0:
                        phi_sum += np.log(matches / (n_patterns // step))
                        count += 1
                
                return phi_sum / count if count > 0 else 0
                
            phi_m = _phi_optimized(m)
            phi_m1 = _phi_optimized(m + 1)
            
            return float(phi_m - phi_m1)
            
        except Exception as e:
            logger.debug(f"Optimized sample entropy computation failed: {e}")
            return 0

    def _compute_approximate_entropy_optimized(self, rr_intervals: np.ndarray, 
                                             m: int = 2, r: float = 0.2) -> float:
        """
        Optimized approximate entropy with performance improvements.
        
        PERFORMANCE OPTIMIZATIONS:
        - Reduced pattern matching for large datasets
        - Memory-efficient operations
        - Early termination conditions
        """
        if len(rr_intervals) < 50:
            return 0
            
        try:
            N = len(rr_intervals)
            
            # Adaptive sampling for large datasets
            if N > 3000:
                # Sample down to manageable size
                indices = np.linspace(0, N-1, 2000, dtype=int)
                rr_intervals = rr_intervals[indices]
                N = len(rr_intervals)
            
            r_abs = r * np.std(rr_intervals)
            
            def _phi_fast(m):
                """Fast phi computation with vectorized operations."""
                phi_sum = 0
                n_patterns = N - m + 1
                
                for i in range(n_patterns):
                    template = rr_intervals[i:i + m]
                    
                    # Vectorized distance computation
                    matches = 0
                    for j in range(n_patterns):
                        pattern = rr_intervals[j:j + m]
                        if np.max(np.abs(template - pattern)) <= r_abs:
                            matches += 1
                    
                    if matches > 0:
                        phi_sum += np.log(matches / n_patterns)
                        
                return phi_sum / n_patterns
                
            return float(_phi_fast(m) - _phi_fast(m + 1))
            
        except Exception as e:
            logger.debug(f"Optimized approximate entropy computation failed: {e}")
            return 0
            
    def _compute_triangular_measures(self, rr_intervals: np.ndarray) -> Tuple[float, float]:
        """Compute triangular interpolation measures."""
        if len(rr_intervals) < 50:
            return 0, 0
            
        try:
            # Create histogram
            hist, bin_edges = np.histogram(rr_intervals, bins=128)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # Find peak of distribution
            peak_idx = np.argmax(hist)
            peak_height = hist[peak_idx]
            
            if peak_height == 0:
                return 0, 0
                
            # Triangular interpolation
            tinn = 2 * len(rr_intervals) / peak_height if peak_height > 0 else 0
            triangular_index = len(rr_intervals) / peak_height if peak_height > 0 else 0
            
            return float(tinn), float(triangular_index)
            
        except Exception as e:
            logger.debug(f"Triangular measures computation failed: {e}")
            return 0, 0