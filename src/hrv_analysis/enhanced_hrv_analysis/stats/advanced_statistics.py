"""
Advanced Statistical Analysis for HRV Data

This module provides sophisticated statistical methods for HRV analysis including:
- Generalized Additive Models (GAMs) for capturing nonlinear trends
- Mixed-effects models for repeated measures with subject-specific random effects  
- Power analysis and effect size calculations
- Bootstrap procedures and permutation tests
- Time series decomposition and trend analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from typing import Dict, Any, Optional, List, Tuple, Union
import logging
from dataclasses import dataclass
import warnings

# Statistical modeling imports (with fallbacks)
try:
    import statsmodels.api as sm
    from statsmodels.gam.api import GLMGam, BSplines
    from statsmodels.formula.api import mixedlm
    import patsy
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    
try:
    import pingouin as pg
    HAS_PINGOUIN = True
except ImportError:
    HAS_PINGOUIN = False

logger = logging.getLogger(__name__)

@dataclass
class StatisticalResult:
    """Container for statistical analysis results."""
    test_name: str
    statistic: float
    p_value: float
    effect_size: float
    confidence_interval: Tuple[float, float]
    degrees_of_freedom: Optional[int]
    interpretation: str
    raw_results: Dict[str, Any]

@dataclass
class GAMResult:
    """Container for GAM analysis results."""
    model_summary: str
    aic: float
    bic: float
    r_squared: float
    predicted_values: np.ndarray
    residuals: np.ndarray
    smooth_terms: Dict[str, np.ndarray]
    significance_tests: Dict[str, float]

@dataclass
class MixedEffectsResult:
    """Container for mixed-effects model results."""
    fixed_effects: pd.DataFrame
    random_effects: pd.DataFrame
    aic: float
    bic: float
    log_likelihood: float
    residuals: np.ndarray
    predicted_values: np.ndarray
    variance_components: Dict[str, float]

class AdvancedStats:
    """Advanced statistical analysis for HRV data."""
    
    def __init__(self, alpha: float = 0.05, n_bootstrap: int = 1000):
        """
        Initialize advanced statistics analyzer.
        
        Args:
            alpha: Significance level for statistical tests
            n_bootstrap: Number of bootstrap iterations
        """
        self.alpha = alpha
        self.n_bootstrap = n_bootstrap
        self.scaler = StandardScaler()
        
        if not HAS_STATSMODELS:
            logger.warning("statsmodels not available. GAM and mixed-effects analyses will be limited.")
        if not HAS_PINGOUIN:
            logger.warning("pingouin not available. Some effect size calculations may be limited.")
            
    def fit_gam_temporal_trend(self,
                             hrv_data: pd.DataFrame,
                             outcome_var: str,
                             time_var: str = 'Sol',
                             subject_var: str = 'subject',
                             spline_df: int = 4) -> GAMResult:
        """
        Fit Generalized Additive Model to capture nonlinear temporal trends.
        
        Args:
            hrv_data: DataFrame with HRV metrics
            outcome_var: Name of outcome variable (e.g., 'rmssd', 'lf_hf_ratio')  
            time_var: Name of time variable
            subject_var: Name of subject identifier
            spline_df: Degrees of freedom for spline smoothing
            
        Returns:
            GAM analysis results
        """
        if not HAS_STATSMODELS:
            logger.error("statsmodels required for GAM analysis")
            return GAMResult("Error: statsmodels not available", 0, 0, 0, 
                           np.array([]), np.array([]), {}, {})
                           
        try:
            # Prepare data
            data_clean = hrv_data[[outcome_var, time_var, subject_var]].dropna()
            
            if len(data_clean) < 20:
                logger.warning("Insufficient data for reliable GAM analysis")
                
            # Create design matrix
            y = data_clean[outcome_var].values
            X = data_clean[[time_var]].values
            
            # Set up B-spline basis
            spline_basis = BSplines(X, df=spline_df, degree=3)
            
            # Fit GAM
            model = GLMGam(y, smoother=spline_basis, family=sm.families.Gaussian())
            gam_results = model.fit()
            
            # Extract results
            predicted_values = gam_results.fittedvalues
            residuals = gam_results.resid_response
            
            # Compute smooth terms
            smooth_terms = {}
            smooth_terms[time_var] = gam_results.fittedvalues
            
            # Significance tests
            significance_tests = {
                'smooth_term_pvalue': float(gam_results.pvalues[0]) if len(gam_results.pvalues) > 0 else 1.0,
                'deviance_explained': float(gam_results.deviance / gam_results.null_deviance) if gam_results.null_deviance != 0 else 0
            }
            
            # Model summary
            model_summary = f"""
GAM Analysis for {outcome_var}:
AIC: {gam_results.aic:.2f}
BIC: {gam_results.bic:.2f}
Deviance Explained: {significance_tests['deviance_explained']:.1%}
Smooth term p-value: {significance_tests['smooth_term_pvalue']:.4f}
"""
            
            return GAMResult(
                model_summary=model_summary,
                aic=float(gam_results.aic),
                bic=float(gam_results.bic),
                r_squared=significance_tests['deviance_explained'],
                predicted_values=predicted_values,
                residuals=residuals,
                smooth_terms=smooth_terms,
                significance_tests=significance_tests
            )
            
        except Exception as e:
            logger.error(f"Error in GAM analysis: {e}")
            return GAMResult(f"Error: {e}", 0, 0, 0, np.array([]), np.array([]), {}, {})
            
    def fit_mixed_effects_model(self,
                              hrv_data: pd.DataFrame,
                              outcome_var: str,
                              fixed_effects: List[str],
                              random_effects: str = 'subject',
                              group_var: str = 'subject') -> MixedEffectsResult:
        """
        Fit mixed-effects model for repeated measures analysis.
        
        Args:
            hrv_data: DataFrame with HRV metrics
            outcome_var: Outcome variable name
            fixed_effects: List of fixed effect variables
            random_effects: Random effects specification
            group_var: Grouping variable for random effects
            
        Returns:
            Mixed-effects model results
        """
        if not HAS_STATSMODELS:
            logger.error("statsmodels required for mixed-effects analysis")
            return MixedEffectsResult(pd.DataFrame(), pd.DataFrame(), 0, 0, 0, 
                                    np.array([]), np.array([]), {})
                                    
        try:
            # Prepare data
            required_vars = [outcome_var, group_var] + fixed_effects
            data_clean = hrv_data[required_vars].dropna()
            
            if len(data_clean) < 20:
                logger.warning("Insufficient data for reliable mixed-effects analysis")
                
            # Create formula
            fixed_formula = f"{outcome_var} ~ " + " + ".join(fixed_effects)
            
            # Fit mixed-effects model
            model = mixedlm(fixed_formula, data_clean, groups=data_clean[group_var])
            me_results = model.fit()
            
            # Extract fixed effects
            fixed_effects_df = pd.DataFrame({
                'coefficient': me_results.params,
                'std_error': me_results.bse,
                't_value': me_results.tvalues,
                'p_value': me_results.pvalues,
                'conf_lower': me_results.conf_int()[0],
                'conf_upper': me_results.conf_int()[1]
            })
            
            # Extract random effects (if available)
            try:
                random_effects_df = pd.DataFrame(me_results.random_effects).T
            except:
                random_effects_df = pd.DataFrame()
                
            # Variance components
            variance_components = {
                'within_subject_variance': float(me_results.scale),
                'between_subject_variance': float(me_results.cov_re.iloc[0, 0]) if hasattr(me_results, 'cov_re') else 0
            }
            
            # ICC calculation
            total_var = variance_components['within_subject_variance'] + variance_components['between_subject_variance']
            icc = variance_components['between_subject_variance'] / total_var if total_var > 0 else 0
            variance_components['icc'] = icc
            
            return MixedEffectsResult(
                fixed_effects=fixed_effects_df,
                random_effects=random_effects_df,
                aic=float(me_results.aic),
                bic=float(me_results.bic),
                log_likelihood=float(me_results.llf),
                residuals=me_results.resid,
                predicted_values=me_results.fittedvalues,
                variance_components=variance_components
            )
            
        except Exception as e:
            logger.error(f"Error in mixed-effects analysis: {e}")
            return MixedEffectsResult(pd.DataFrame(), pd.DataFrame(), 0, 0, 0,
                                    np.array([]), np.array([]), {})
                                    
    def bootstrap_confidence_interval(self,
                                    data: np.ndarray,
                                    statistic_func: callable,
                                    confidence_level: float = 0.95) -> Tuple[float, float, np.ndarray]:
        """
        Compute bootstrap confidence intervals for a given statistic.
        
        Args:
            data: Input data array
            statistic_func: Function to compute statistic (e.g., np.mean, np.std)
            confidence_level: Confidence level (default 0.95)
            
        Returns:
            Tuple of (lower_bound, upper_bound, bootstrap_samples)
        """
        try:
            if len(data) < 10:
                logger.warning("Small sample size may lead to unreliable bootstrap CIs")
                
            bootstrap_stats = []
            
            for _ in range(self.n_bootstrap):
                # Bootstrap sample with replacement
                bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
                bootstrap_stat = statistic_func(bootstrap_sample)
                bootstrap_stats.append(bootstrap_stat)
                
            bootstrap_stats = np.array(bootstrap_stats)
            
            # Calculate confidence intervals
            alpha = 1 - confidence_level
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100
            
            ci_lower = np.percentile(bootstrap_stats, lower_percentile)
            ci_upper = np.percentile(bootstrap_stats, upper_percentile)
            
            return ci_lower, ci_upper, bootstrap_stats
            
        except Exception as e:
            logger.error(f"Error in bootstrap CI calculation: {e}")
            return 0, 0, np.array([])
            
    def permutation_test(self,
                        group1: np.ndarray,
                        group2: np.ndarray,
                        statistic_func: callable = None,
                        n_permutations: int = 10000) -> StatisticalResult:
        """
        Perform permutation test for group differences.
        
        Args:
            group1: First group data
            group2: Second group data  
            statistic_func: Function to compute test statistic (default: mean difference)
            n_permutations: Number of permutations
            
        Returns:
            Statistical test results
        """
        try:
            if statistic_func is None:
                statistic_func = lambda x, y: np.mean(x) - np.mean(y)
                
            # Observed test statistic
            observed_stat = statistic_func(group1, group2)
            
            # Combined data
            combined_data = np.concatenate([group1, group2])
            n1, n2 = len(group1), len(group2)
            
            # Permutation distribution
            permutation_stats = []
            for _ in range(n_permutations):
                # Permute combined data
                permuted_data = np.random.permutation(combined_data)
                
                # Split into groups
                perm_group1 = permuted_data[:n1]
                perm_group2 = permuted_data[n1:]
                
                # Compute statistic
                perm_stat = statistic_func(perm_group1, perm_group2)
                permutation_stats.append(perm_stat)
                
            permutation_stats = np.array(permutation_stats)
            
            # Calculate p-value
            p_value = np.mean(np.abs(permutation_stats) >= np.abs(observed_stat))
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt(((n1 - 1) * np.var(group1) + (n2 - 1) * np.var(group2)) / (n1 + n2 - 2))
            cohens_d = observed_stat / pooled_std if pooled_std > 0 else 0
            
            # Confidence interval from permutation distribution
            ci_lower = np.percentile(permutation_stats, 2.5)
            ci_upper = np.percentile(permutation_stats, 97.5)
            
            # Interpretation
            if p_value < 0.001:
                interpretation = "Highly significant difference (p < 0.001)"
            elif p_value < 0.01:
                interpretation = "Very significant difference (p < 0.01)"
            elif p_value < 0.05:
                interpretation = "Significant difference (p < 0.05)"
            else:
                interpretation = "No significant difference"
                
            return StatisticalResult(
                test_name="Permutation Test",
                statistic=observed_stat,
                p_value=p_value,
                effect_size=cohens_d,
                confidence_interval=(ci_lower, ci_upper),
                degrees_of_freedom=n1 + n2 - 2,
                interpretation=interpretation,
                raw_results={
                    'n_permutations': n_permutations,
                    'permutation_distribution': permutation_stats,
                    'group1_n': n1,
                    'group2_n': n2
                }
            )
            
        except Exception as e:
            logger.error(f"Error in permutation test: {e}")
            return StatisticalResult("Error", 0, 1, 0, (0, 0), None, str(e), {})
            
    def compute_effect_sizes(self,
                           group1: np.ndarray,
                           group2: np.ndarray) -> Dict[str, float]:
        """
        Compute multiple effect size measures.
        
        Args:
            group1: First group data
            group2: Second group data
            
        Returns:
            Dictionary of effect size measures
        """
        try:
            # Basic statistics
            m1, m2 = np.mean(group1), np.mean(group2)
            s1, s2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
            n1, n2 = len(group1), len(group2)
            
            # Cohen's d
            pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
            cohens_d = (m1 - m2) / pooled_std if pooled_std > 0 else 0
            
            # Hedges' g (corrected for small samples)
            correction_factor = 1 - (3 / (4 * (n1 + n2) - 9))
            hedges_g = cohens_d * correction_factor
            
            # Glass's delta (using group 2 as control)
            glass_delta = (m1 - m2) / s2 if s2 > 0 else 0
            
            # Common Language Effect Size
            # Probability that a randomly selected value from group1 > group2
            comparisons = []
            for val1 in group1:
                for val2 in group2:
                    comparisons.append(val1 > val2)
            cles = np.mean(comparisons) if comparisons else 0.5
            
            # Pearson's r (point-biserial correlation)
            combined = np.concatenate([group1, group2])
            group_labels = np.concatenate([np.ones(n1), np.zeros(n2)])
            pearson_r = np.corrcoef(combined, group_labels)[0, 1]
            
            effect_sizes = {
                'cohens_d': cohens_d,
                'hedges_g': hedges_g,
                'glass_delta': glass_delta,
                'common_language_effect_size': cles,
                'pearson_r': pearson_r,
                'mean_difference': m1 - m2,
                'percent_difference': ((m1 - m2) / m2) * 100 if m2 != 0 else 0
            }
            
            return effect_sizes
            
        except Exception as e:
            logger.error(f"Error computing effect sizes: {e}")
            return {}
            
    def power_analysis(self,
                      effect_size: float,
                      sample_size: int = None,
                      alpha: float = None,
                      power: float = None,
                      test_type: str = 'two_sample_ttest') -> Dict[str, Any]:
        """
        Perform statistical power analysis.
        
        Args:
            effect_size: Expected effect size (Cohen's d)
            sample_size: Sample size per group
            alpha: Type I error rate
            power: Statistical power (1 - Type II error)
            test_type: Type of statistical test
            
        Returns:
            Power analysis results
        """
        if HAS_PINGOUIN:
            try:
                # Use pingouin for power analysis
                if sample_size is None and power is not None and alpha is not None:
                    # Calculate required sample size
                    result = pg.power_ttest(d=effect_size, power=power, alpha=alpha)
                    return {
                        'analysis_type': 'sample_size',
                        'effect_size': effect_size,
                        'required_sample_size': result,
                        'power': power,
                        'alpha': alpha,
                        'interpretation': f"Need {result:.0f} subjects per group to detect effect size {effect_size:.2f}"
                    }
                elif power is None and sample_size is not None and alpha is not None:
                    # Calculate achieved power
                    result = pg.power_ttest(d=effect_size, n=sample_size, alpha=alpha)
                    return {
                        'analysis_type': 'achieved_power',
                        'effect_size': effect_size,
                        'sample_size': sample_size,
                        'achieved_power': result,
                        'alpha': alpha,
                        'interpretation': f"With n={sample_size} per group, power = {result:.2f} to detect effect size {effect_size:.2f}"
                    }
                else:
                    return {'error': 'Need to specify exactly 3 of 4 parameters (effect_size, sample_size, alpha, power)'}
                    
            except Exception as e:
                logger.error(f"Error in pingouin power analysis: {e}")
                
        # Fallback manual power calculation for t-test
        try:
            from scipy.stats import norm, t
            
            if alpha is None:
                alpha = self.alpha
                
            # Critical t-value
            if sample_size is not None:
                df = 2 * sample_size - 2
                t_critical = t.ppf(1 - alpha/2, df)
                
                # Non-centrality parameter
                ncp = effect_size * np.sqrt(sample_size / 2)
                
                # Calculate power
                power_calculated = 1 - t.cdf(t_critical, df, ncp) + t.cdf(-t_critical, df, ncp)
                
                return {
                    'analysis_type': 'achieved_power',
                    'effect_size': effect_size,
                    'sample_size': sample_size,
                    'achieved_power': power_calculated,
                    'alpha': alpha,
                    'interpretation': f"With n={sample_size} per group, power = {power_calculated:.2f}"
                }
            else:
                return {'error': 'Manual power calculation requires sample_size'}
                
        except Exception as e:
            logger.error(f"Error in manual power analysis: {e}")
            return {'error': str(e)}
            
    def time_series_decomposition(self,
                                hrv_data: pd.DataFrame,
                                value_col: str,
                                time_col: str,
                                subject_col: str = 'subject') -> Dict[str, Any]:
        """
        Perform time series decomposition of HRV metrics.
        
        Args:
            hrv_data: DataFrame with HRV time series
            value_col: Column name for values
            time_col: Column name for time variable
            subject_col: Column name for subject identifier
            
        Returns:
            Decomposition results
        """
        try:
            results = {}
            
            # Process each subject separately
            for subject in hrv_data[subject_col].unique():
                subject_data = hrv_data[hrv_data[subject_col] == subject].copy()
                subject_data = subject_data.sort_values(time_col)
                
                if len(subject_data) < 4:
                    continue
                    
                values = subject_data[value_col].values
                time_points = subject_data[time_col].values
                
                # Linear trend
                trend_coeffs = np.polyfit(time_points, values, 1)
                trend = np.polyval(trend_coeffs, time_points)
                
                # Detrended values
                detrended = values - trend
                
                # Seasonal component (if enough data points)
                seasonal = np.zeros_like(values)
                if len(values) >= 8:  # Need at least 2 cycles
                    try:
                        # Simple seasonal decomposition using moving average
                        period = 4  # Assume 4-point seasonal pattern
                        seasonal_means = []
                        
                        for i in range(period):
                            seasonal_indices = np.arange(i, len(values), period)
                            if len(seasonal_indices) > 0:
                                seasonal_means.append(np.mean(values[seasonal_indices]))
                            else:
                                seasonal_means.append(0)
                                
                        # Repeat seasonal pattern
                        for i in range(len(values)):
                            seasonal[i] = seasonal_means[i % period]
                            
                        # Center seasonal component
                        seasonal = seasonal - np.mean(seasonal)
                        
                    except:
                        seasonal = np.zeros_like(values)
                        
                # Residual component
                residual = values - trend - seasonal
                
                results[subject] = {
                    'original': values,
                    'trend': trend,
                    'seasonal': seasonal,
                    'residual': residual,
                    'trend_slope': trend_coeffs[0],
                    'trend_pvalue': stats.pearsonr(time_points, values)[1] if len(time_points) > 2 else 1.0,
                    'residual_variance': np.var(residual),
                    'trend_explained_variance': 1 - (np.var(residual) / np.var(values)) if np.var(values) > 0 else 0
                }
                
            return results
            
        except Exception as e:
            logger.error(f"Error in time series decomposition: {e}")
            return {'error': str(e)}
            
    def detect_outliers_multivariate(self,
                                   hrv_metrics: pd.DataFrame,
                                   method: str = 'isolation_forest',
                                   contamination: float = 0.1) -> Dict[str, Any]:
        """
        Detect multivariate outliers in HRV metrics.
        
        Args:
            hrv_metrics: DataFrame with HRV metrics
            method: Outlier detection method ('isolation_forest', 'elliptic_envelope', 'local_outlier_factor')
            contamination: Expected proportion of outliers
            
        Returns:
            Outlier detection results
        """
        try:
            from sklearn.ensemble import IsolationForest
            from sklearn.covariance import EllipticEnvelope  
            from sklearn.neighbors import LocalOutlierFactor
            
            # Select numeric columns only
            numeric_data = hrv_metrics.select_dtypes(include=[np.number])
            
            if numeric_data.empty:
                return {'error': 'No numeric variables found'}
                
            # Remove columns with no variation
            numeric_data = numeric_data.loc[:, numeric_data.std() > 0]
            
            if numeric_data.empty:
                return {'error': 'No variables with sufficient variation'}
                
            # Standardize data
            data_scaled = self.scaler.fit_transform(numeric_data)
            
            # Apply outlier detection method
            if method == 'isolation_forest':
                detector = IsolationForest(contamination=contamination, random_state=42)
            elif method == 'elliptic_envelope':
                detector = EllipticEnvelope(contamination=contamination, random_state=42)
            elif method == 'local_outlier_factor':
                detector = LocalOutlierFactor(contamination=contamination, novelty=False)
            else:
                return {'error': f'Unknown method: {method}'}
                
            # Fit and predict
            outlier_labels = detector.fit_predict(data_scaled)
            outlier_mask = outlier_labels == -1
            
            # Outlier scores (if available)
            if hasattr(detector, 'decision_function'):
                outlier_scores = detector.decision_function(data_scaled)
            elif hasattr(detector, 'score_samples'):
                outlier_scores = detector.score_samples(data_scaled)
            else:
                outlier_scores = np.zeros(len(data_scaled))
                
            results = {
                'outlier_mask': outlier_mask,
                'outlier_scores': outlier_scores,
                'n_outliers': np.sum(outlier_mask),
                'outlier_percentage': np.mean(outlier_mask) * 100,
                'outlier_indices': np.where(outlier_mask)[0].tolist(),
                'method': method,
                'contamination': contamination
            }
            
            # Add outlier summary statistics
            if np.any(outlier_mask):
                outlier_data = numeric_data.iloc[outlier_mask]
                normal_data = numeric_data.iloc[~outlier_mask]
                
                results['outlier_summary'] = {
                    'outlier_means': outlier_data.mean().to_dict(),
                    'normal_means': normal_data.mean().to_dict(),
                    'mean_differences': (outlier_data.mean() - normal_data.mean()).to_dict()
                }
                
            return results
            
        except ImportError:
            return {'error': 'scikit-learn required for multivariate outlier detection'}
        except Exception as e:
            logger.error(f"Error in outlier detection: {e}")
            return {'error': str(e)}
            
    # Enhanced Power Analysis and Sensitivity Simulations
    
    def post_hoc_power_analysis(self, 
                               observed_data: Dict[str, np.ndarray],
                               test_type: str = 'two_sample_ttest',
                               alpha: float = 0.05) -> Dict[str, Any]:
        """
        Perform comprehensive post-hoc power analysis on observed data.
        
        Args:
            observed_data: Dictionary with groups as keys and data arrays as values
            test_type: Type of statistical test ('two_sample_ttest', 'paired_ttest', 'anova')
            alpha: Type I error rate
            
        Returns:
            Comprehensive power analysis results
        """
        try:
            results = {
                'test_type': test_type,
                'alpha': alpha,
                'groups': list(observed_data.keys()),
                'sample_sizes': {k: len(v) for k, v in observed_data.items()},
                'group_statistics': {}
            }
            
            # Calculate group statistics
            for group_name, data in observed_data.items():
                results['group_statistics'][group_name] = {
                    'mean': float(np.mean(data)),
                    'std': float(np.std(data, ddof=1)),
                    'n': len(data),
                    'sem': float(stats.sem(data))
                }
            
            # Perform power analysis based on test type
            if test_type == 'two_sample_ttest' and len(observed_data) == 2:
                results.update(self._post_hoc_two_sample_power(observed_data, alpha))
            elif test_type == 'paired_ttest' and len(observed_data) == 2:
                results.update(self._post_hoc_paired_power(observed_data, alpha))
            elif test_type == 'anova':
                results.update(self._post_hoc_anova_power(observed_data, alpha))
            else:
                results['error'] = f"Unsupported test type '{test_type}' or incorrect number of groups"
                
            return results
            
        except Exception as e:
            logger.error(f"Error in post-hoc power analysis: {e}")
            return {'error': str(e)}
    
    def _post_hoc_two_sample_power(self, data_dict: Dict[str, np.ndarray], alpha: float) -> Dict[str, Any]:
        """Post-hoc power analysis for two-sample t-test."""
        groups = list(data_dict.keys())
        group1_data, group2_data = data_dict[groups[0]], data_dict[groups[1]]
        
        # Calculate observed effect size (Cohen's d)
        pooled_std = np.sqrt(((len(group1_data) - 1) * np.var(group1_data, ddof=1) + 
                             (len(group2_data) - 1) * np.var(group2_data, ddof=1)) / 
                            (len(group1_data) + len(group2_data) - 2))
        
        cohens_d = abs(np.mean(group1_data) - np.mean(group2_data)) / pooled_std
        
        # Calculate achieved power
        n1, n2 = len(group1_data), len(group2_data)
        
        if HAS_PINGOUIN:
            achieved_power = pg.power_ttest(d=cohens_d, n=(n1 + n2) / 2, alpha=alpha)
        else:
            # Manual calculation
            df = n1 + n2 - 2
            t_critical = stats.t.ppf(1 - alpha/2, df)
            ncp = cohens_d * np.sqrt(n1 * n2 / (n1 + n2))
            achieved_power = 1 - stats.t.cdf(t_critical, df, ncp) + stats.t.cdf(-t_critical, df, ncp)
        
        # Sample size recommendations for different power levels
        power_targets = [0.80, 0.90, 0.95]
        sample_size_recommendations = {}
        
        for power_target in power_targets:
            if HAS_PINGOUIN:
                n_required = pg.power_ttest(d=cohens_d, power=power_target, alpha=alpha)
            else:
                # Approximate calculation
                z_alpha = stats.norm.ppf(1 - alpha/2)
                z_beta = stats.norm.ppf(power_target)
                n_required = 2 * ((z_alpha + z_beta) / cohens_d) ** 2
            
            sample_size_recommendations[f'power_{power_target}'] = int(np.ceil(n_required))
        
        return {
            'cohens_d': float(cohens_d),
            'achieved_power': float(achieved_power),
            'sample_size_recommendations': sample_size_recommendations,
            'interpretation': self._interpret_power_results(cohens_d, achieved_power)
        }
    
    def _post_hoc_paired_power(self, data_dict: Dict[str, np.ndarray], alpha: float) -> Dict[str, Any]:
        """Post-hoc power analysis for paired t-test."""
        groups = list(data_dict.keys())
        pre_data, post_data = data_dict[groups[0]], data_dict[groups[1]]
        
        # Calculate differences
        differences = post_data - pre_data
        mean_diff = np.mean(differences)
        std_diff = np.std(differences, ddof=1)
        
        # Effect size for paired t-test
        cohens_d = abs(mean_diff) / std_diff
        n = len(differences)
        
        # Achieved power
        if HAS_PINGOUIN:
            achieved_power = pg.power_ttest(d=cohens_d, n=n, alpha=alpha, paired=True)
        else:
            df = n - 1
            t_critical = stats.t.ppf(1 - alpha/2, df)
            ncp = cohens_d * np.sqrt(n)
            achieved_power = 1 - stats.t.cdf(t_critical, df, ncp) + stats.t.cdf(-t_critical, df, ncp)
        
        return {
            'cohens_d': float(cohens_d),
            'achieved_power': float(achieved_power),
            'mean_difference': float(mean_diff),
            'std_difference': float(std_diff),
            'interpretation': self._interpret_power_results(cohens_d, achieved_power)
        }
    
    def _post_hoc_anova_power(self, data_dict: Dict[str, np.ndarray], alpha: float) -> Dict[str, Any]:
        """Post-hoc power analysis for ANOVA."""
        # Calculate eta-squared (effect size for ANOVA)
        all_data = np.concatenate(list(data_dict.values()))
        grand_mean = np.mean(all_data)
        
        # Between-group sum of squares
        ss_between = sum(len(group) * (np.mean(group) - grand_mean)**2 
                        for group in data_dict.values())
        
        # Total sum of squares
        ss_total = np.sum((all_data - grand_mean)**2)
        
        eta_squared = ss_between / ss_total
        f_statistic = eta_squared / (1 - eta_squared)
        
        # Convert to Cohen's f
        cohens_f = np.sqrt(eta_squared / (1 - eta_squared))
        
        # Degrees of freedom
        k = len(data_dict)  # number of groups
        n_total = len(all_data)
        df_between = k - 1
        df_within = n_total - k
        
        # Approximate power calculation
        from scipy.stats import f as f_dist
        f_critical = f_dist.ppf(1 - alpha, df_between, df_within)
        
        # Non-central F approximation for power
        ncp = n_total * eta_squared / (1 - eta_squared)
        achieved_power = 1 - f_dist.cdf(f_critical, df_between, df_within, ncp)
        
        return {
            'eta_squared': float(eta_squared),
            'cohens_f': float(cohens_f),
            'f_statistic': float(f_statistic),
            'achieved_power': float(achieved_power),
            'df_between': df_between,
            'df_within': df_within,
            'interpretation': self._interpret_anova_power(eta_squared, achieved_power)
        }
    
    def sensitivity_simulation(self,
                             effect_sizes: List[float] = None,
                             sample_sizes: List[int] = None,
                             alpha_levels: List[float] = None,
                             n_simulations: int = 1000,
                             test_type: str = 'two_sample_ttest') -> Dict[str, Any]:
        """
        Perform comprehensive sensitivity analysis through Monte Carlo simulations.
        
        Args:
            effect_sizes: List of effect sizes to test
            sample_sizes: List of sample sizes to test  
            alpha_levels: List of alpha levels to test
            n_simulations: Number of Monte Carlo simulations
            test_type: Type of statistical test
            
        Returns:
            Sensitivity analysis results
        """
        if effect_sizes is None:
            effect_sizes = [0.2, 0.5, 0.8, 1.0, 1.2]  # Small to large effects
        if sample_sizes is None:
            sample_sizes = [10, 20, 30, 50, 100, 200]
        if alpha_levels is None:
            alpha_levels = [0.01, 0.05, 0.10]
            
        try:
            results = {
                'simulation_parameters': {
                    'effect_sizes': effect_sizes,
                    'sample_sizes': sample_sizes,
                    'alpha_levels': alpha_levels,
                    'n_simulations': n_simulations,
                    'test_type': test_type
                },
                'power_surface': {},
                'type_i_error_rates': {},
                'recommendations': {}
            }
            
            # Run simulations for each combination
            for effect_size in effect_sizes:
                for sample_size in sample_sizes:
                    for alpha in alpha_levels:
                        key = f"d={effect_size:.1f}_n={sample_size}_a={alpha:.2f}"
                        
                        power, type_i_rate = self._monte_carlo_simulation(
                            effect_size, sample_size, alpha, n_simulations, test_type
                        )
                        
                        results['power_surface'][key] = {
                            'effect_size': effect_size,
                            'sample_size': sample_size,
                            'alpha': alpha,
                            'estimated_power': power,
                            'type_i_error_rate': type_i_rate
                        }
            
            # Generate recommendations
            results['recommendations'] = self._generate_power_recommendations(results['power_surface'])
            
            return results
            
        except Exception as e:
            logger.error(f"Error in sensitivity simulation: {e}")
            return {'error': str(e)}
    
    def _monte_carlo_simulation(self, effect_size: float, sample_size: int, 
                               alpha: float, n_simulations: int, test_type: str) -> Tuple[float, float]:
        """Run Monte Carlo simulation for power and Type I error estimation."""
        significant_tests = 0
        null_rejections = 0  # For Type I error calculation
        
        for _ in range(n_simulations):
            # Generate data with true effect
            group1 = np.random.normal(0, 1, sample_size)
            group2 = np.random.normal(effect_size, 1, sample_size)
            
            # Perform statistical test
            if test_type == 'two_sample_ttest':
                t_stat, p_value = stats.ttest_ind(group1, group2)
            elif test_type == 'paired_ttest':
                # For paired test, use difference scores
                differences = group2 - group1
                t_stat, p_value = stats.ttest_1samp(differences, 0)
            else:
                continue  # Skip unsupported tests
            
            # Count significant results
            if p_value < alpha:
                significant_tests += 1
            
            # Also test under null hypothesis (effect_size = 0) for Type I error
            null_group1 = np.random.normal(0, 1, sample_size)
            null_group2 = np.random.normal(0, 1, sample_size)
            
            if test_type == 'two_sample_ttest':
                _, null_p = stats.ttest_ind(null_group1, null_group2)
            elif test_type == 'paired_ttest':
                null_diff = null_group2 - null_group1
                _, null_p = stats.ttest_1samp(null_diff, 0)
                
            if null_p < alpha:
                null_rejections += 1
        
        estimated_power = significant_tests / n_simulations
        type_i_error_rate = null_rejections / n_simulations
        
        return estimated_power, type_i_error_rate
    
    def _interpret_power_results(self, effect_size: float, power: float) -> str:
        """Generate interpretation of power analysis results."""
        effect_magnitude = "negligible" if effect_size < 0.2 else \
                          "small" if effect_size < 0.5 else \
                          "medium" if effect_size < 0.8 else "large"
        
        power_adequacy = "inadequate" if power < 0.5 else \
                        "marginal" if power < 0.8 else \
                        "adequate" if power < 0.95 else "excellent"
        
        return f"Observed {effect_magnitude} effect size (d={effect_size:.2f}) with {power_adequacy} power ({power:.2f})"
    
    def _interpret_anova_power(self, eta_squared: float, power: float) -> str:
        """Generate interpretation of ANOVA power analysis."""
        effect_magnitude = "negligible" if eta_squared < 0.01 else \
                          "small" if eta_squared < 0.06 else \
                          "medium" if eta_squared < 0.14 else "large"
        
        power_adequacy = "inadequate" if power < 0.5 else \
                        "marginal" if power < 0.8 else \
                        "adequate" if power < 0.95 else "excellent"
        
        return f"Observed {effect_magnitude} effect size (η²={eta_squared:.3f}) with {power_adequacy} power ({power:.2f})"
    
    def _generate_power_recommendations(self, power_surface: Dict[str, Any]) -> Dict[str, Any]:
        """Generate recommendations based on sensitivity analysis results."""
        recommendations = {
            'optimal_designs': [],
            'minimal_requirements': {},
            'power_warnings': []
        }
        
        # Find optimal design combinations
        for key, results in power_surface.items():
            if results['estimated_power'] >= 0.8 and results['type_i_error_rate'] <= results['alpha'] * 1.1:
                recommendations['optimal_designs'].append({
                    'design': key,
                    'effect_size': results['effect_size'],
                    'sample_size': results['sample_size'],
                    'alpha': results['alpha'],
                    'power': results['estimated_power']
                })
        
        # Sort by sample size (prefer smaller samples)
        recommendations['optimal_designs'].sort(key=lambda x: x['sample_size'])
        
        # Find minimal sample size for different effect sizes
        effect_sizes = sorted(set(r['effect_size'] for r in power_surface.values()))
        for effect_size in effect_sizes:
            relevant_results = [r for r in power_surface.values() 
                              if r['effect_size'] == effect_size and r['estimated_power'] >= 0.8]
            if relevant_results:
                min_n = min(r['sample_size'] for r in relevant_results)
                recommendations['minimal_requirements'][f'effect_size_{effect_size}'] = min_n
        
        # Generate warnings
        underpowered = [r for r in power_surface.values() if r['estimated_power'] < 0.8]
        if underpowered:
            recommendations['power_warnings'].append(
                f"Found {len(underpowered)} underpowered design combinations"
            )
        
        inflated_type_i = [r for r in power_surface.values() 
                          if r['type_i_error_rate'] > r['alpha'] * 1.2]
        if inflated_type_i:
            recommendations['power_warnings'].append(
                f"Found {len(inflated_type_i)} combinations with inflated Type I error rates"
            )
        
        return recommendations
        
    # Contextual Analysis and External Data Integration
    
    def merge_external_data(self, 
                           hrv_data: pd.DataFrame,
                           external_data: Dict[str, pd.DataFrame],
                           merge_keys: Dict[str, str] = None) -> pd.DataFrame:
        """
        Merge HRV data with external contextual data sources.
        
        Args:
            hrv_data: Primary HRV dataset
            external_data: Dictionary of external datasets to merge
            merge_keys: Dictionary specifying merge keys for each dataset
            
        Returns:
            Merged dataset with all contextual information
        """
        try:
            if merge_keys is None:
                merge_keys = {}
                
            merged_data = hrv_data.copy()
            merge_info = {
                'original_records': len(hrv_data),
                'merged_datasets': [],
                'final_records': 0,
                'merge_statistics': {}
            }
            
            for dataset_name, ext_data in external_data.items():
                logger.info(f"Merging {dataset_name} data...")
                
                # Determine merge key
                merge_key = merge_keys.get(dataset_name, 'subject')
                
                # Validate merge key exists in both datasets
                if merge_key not in merged_data.columns:
                    logger.warning(f"Merge key '{merge_key}' not found in HRV data for {dataset_name}")
                    continue
                    
                if merge_key not in ext_data.columns:
                    logger.warning(f"Merge key '{merge_key}' not found in {dataset_name} data")
                    continue
                
                # Perform merge
                before_merge = len(merged_data)
                merged_data = pd.merge(
                    merged_data, 
                    ext_data, 
                    on=merge_key, 
                    how='left',
                    suffixes=('', f'_{dataset_name}')
                )
                after_merge = len(merged_data)
                
                merge_info['merged_datasets'].append(dataset_name)
                merge_info['merge_statistics'][dataset_name] = {
                    'records_before': before_merge,
                    'records_after': after_merge,
                    'external_records': len(ext_data),
                    'match_rate': len(merged_data[merged_data[merge_key].isin(ext_data[merge_key])]) / before_merge
                }
                
            merge_info['final_records'] = len(merged_data)
            
            # Add merge info as metadata
            merged_data.attrs['merge_info'] = merge_info
            
            return merged_data
            
        except Exception as e:
            logger.error(f"Error merging external data: {e}")
            return hrv_data
    
    def perform_multivariate_regression(self,
                                      data: pd.DataFrame,
                                      outcome_variable: str,
                                      predictor_variables: List[str],
                                      confounders: List[str] = None,
                                      interaction_terms: List[Tuple[str, str]] = None) -> Dict[str, Any]:
        """
        Perform multivariate regression analysis with confounder assessment.
        
        Args:
            data: Dataset with all variables
            outcome_variable: Dependent variable name
            predictor_variables: List of primary predictor variables
            confounders: List of potential confounder variables
            interaction_terms: List of interaction terms as (var1, var2) tuples
            
        Returns:
            Comprehensive regression analysis results
        """
        try:
            if not HAS_STATSMODELS:
                return {'error': 'Statsmodels required for multivariate regression'}
                
            results = {
                'outcome_variable': outcome_variable,
                'predictor_variables': predictor_variables,
                'confounders': confounders or [],
                'models': {},
                'confounder_assessment': {},
                'effect_modifications': {},
                'model_diagnostics': {}
            }
            
            # Prepare data
            all_variables = [outcome_variable] + predictor_variables
            if confounders:
                all_variables.extend(confounders)
                
            # Remove rows with missing values
            clean_data = data[all_variables].dropna()
            
            if len(clean_data) == 0:
                return {'error': 'No complete cases available for analysis'}
            
            results['sample_size'] = len(clean_data)
            results['missing_data_info'] = {
                'original_n': len(data),
                'complete_cases_n': len(clean_data),
                'missing_rate': 1 - len(clean_data)/len(data)
            }
            
            # Model 1: Predictors only (crude model)
            crude_formula = f"{outcome_variable} ~ " + " + ".join(predictor_variables)
            crude_model = sm.OLS.from_formula(crude_formula, data=clean_data).fit()
            results['models']['crude'] = {
                'formula': crude_formula,
                'summary': crude_model.summary(),
                'coefficients': crude_model.params.to_dict(),
                'p_values': crude_model.pvalues.to_dict(),
                'conf_intervals': crude_model.conf_int().to_dict(),
                'r_squared': crude_model.rsquared,
                'aic': crude_model.aic,
                'bic': crude_model.bic
            }
            
            # Model 2: Predictors + confounders (adjusted model)
            if confounders:
                adjusted_formula = crude_formula + " + " + " + ".join(confounders)
                adjusted_model = sm.OLS.from_formula(adjusted_formula, data=clean_data).fit()
                results['models']['adjusted'] = {
                    'formula': adjusted_formula,
                    'summary': adjusted_model.summary(),
                    'coefficients': adjusted_model.params.to_dict(),
                    'p_values': adjusted_model.pvalues.to_dict(),
                    'conf_intervals': adjusted_model.conf_int().to_dict(),
                    'r_squared': adjusted_model.rsquared,
                    'aic': adjusted_model.aic,
                    'bic': adjusted_model.bic
                }
                
                # Assess confounding
                results['confounder_assessment'] = self._assess_confounding(
                    crude_model, adjusted_model, predictor_variables
                )
            
            # Model 3: Include interaction terms if specified
            if interaction_terms:
                interaction_formula = adjusted_formula if confounders else crude_formula
                interaction_terms_str = [f"{var1}*{var2}" for var1, var2 in interaction_terms]
                interaction_formula += " + " + " + ".join(interaction_terms_str)
                
                interaction_model = sm.OLS.from_formula(interaction_formula, data=clean_data).fit()
                results['models']['interaction'] = {
                    'formula': interaction_formula,
                    'summary': interaction_model.summary(),
                    'coefficients': interaction_model.params.to_dict(),
                    'p_values': interaction_model.pvalues.to_dict(),
                    'conf_intervals': interaction_model.conf_int().to_dict(),
                    'r_squared': interaction_model.rsquared,
                    'aic': interaction_model.aic,
                    'bic': interaction_model.bic
                }
                
                # Assess effect modification
                results['effect_modifications'] = self._assess_effect_modification(
                    interaction_model, interaction_terms
                )
            
            # Model diagnostics
            final_model = interaction_model if interaction_terms else (adjusted_model if confounders else crude_model)
            results['model_diagnostics'] = self._perform_regression_diagnostics(final_model, clean_data)
            
            # Model comparison
            results['model_comparison'] = self._compare_regression_models(results['models'])
            
            return results
            
        except Exception as e:
            logger.error(f"Error in multivariate regression: {e}")
            return {'error': str(e)}
    
    def _assess_confounding(self, crude_model, adjusted_model, predictor_variables: List[str]) -> Dict[str, Any]:
        """Assess confounding by comparing crude and adjusted models."""
        confounding_assessment = {}
        
        for predictor in predictor_variables:
            if predictor in crude_model.params and predictor in adjusted_model.params:
                crude_coef = crude_model.params[predictor]
                adjusted_coef = adjusted_model.params[predictor]
                
                # Calculate percent change in coefficient
                if crude_coef != 0:
                    percent_change = abs((adjusted_coef - crude_coef) / crude_coef) * 100
                else:
                    percent_change = 0
                
                # Assess significance of confounding
                confounding_magnitude = "none" if percent_change < 10 else \
                                      "mild" if percent_change < 20 else \
                                      "moderate" if percent_change < 50 else "severe"
                
                confounding_assessment[predictor] = {
                    'crude_coefficient': crude_coef,
                    'adjusted_coefficient': adjusted_coef,
                    'percent_change': percent_change,
                    'confounding_magnitude': confounding_magnitude,
                    'interpretation': self._interpret_confounding(percent_change)
                }
        
        return confounding_assessment
    
    def _assess_effect_modification(self, interaction_model, interaction_terms: List[Tuple[str, str]]) -> Dict[str, Any]:
        """Assess effect modification from interaction terms."""
        effect_modifications = {}
        
        for var1, var2 in interaction_terms:
            interaction_term = f"{var1}:{var2}"
            
            if interaction_term in interaction_model.params:
                interaction_coef = interaction_model.params[interaction_term]
                interaction_p = interaction_model.pvalues[interaction_term]
                
                # Assess significance of interaction
                significant = interaction_p < 0.05
                magnitude = abs(interaction_coef)
                
                effect_modifications[f"{var1}_x_{var2}"] = {
                    'interaction_coefficient': interaction_coef,
                    'p_value': interaction_p,
                    'significant': significant,
                    'magnitude': magnitude,
                    'interpretation': self._interpret_interaction(interaction_coef, interaction_p)
                }
        
        return effect_modifications
    
    def _perform_regression_diagnostics(self, model, data: pd.DataFrame) -> Dict[str, Any]:
        """Perform comprehensive regression diagnostics."""
        diagnostics = {}
        
        try:
            # Residual analysis
            residuals = model.resid
            fitted_values = model.fittedvalues
            
            diagnostics['residual_analysis'] = {
                'residual_mean': float(np.mean(residuals)),
                'residual_std': float(np.std(residuals)),
                'residual_range': (float(np.min(residuals)), float(np.max(residuals)))
            }
            
            # Normality tests
            from scipy.stats import shapiro, jarque_bera
            
            if len(residuals) <= 5000:  # Shapiro-Wilk has sample size limit
                shapiro_stat, shapiro_p = shapiro(residuals)
                diagnostics['normality_tests'] = {
                    'shapiro_wilk': {'statistic': shapiro_stat, 'p_value': shapiro_p}
                }
            
            jb_stat, jb_p = jarque_bera(residuals)
            diagnostics.setdefault('normality_tests', {})['jarque_bera'] = {
                'statistic': jb_stat, 'p_value': jb_p
            }
            
            # Heteroscedasticity tests
            if HAS_STATSMODELS:
                from statsmodels.stats.diagnostic import het_breuschpagan, het_white
                
                # Breusch-Pagan test
                bp_lm, bp_lm_p, bp_f, bp_f_p = het_breuschpagan(residuals, model.model.exog)
                diagnostics['heteroscedasticity_tests'] = {
                    'breusch_pagan': {'lm_statistic': bp_lm, 'lm_p_value': bp_lm_p}
                }
                
                # White test
                white_lm, white_lm_p, white_f, white_f_p = het_white(residuals, model.model.exog)
                diagnostics['heteroscedasticity_tests']['white'] = {
                    'lm_statistic': white_lm, 'lm_p_value': white_lm_p
                }
            
            # Multicollinearity assessment (VIF)
            if len(model.model.exog[0]) > 1:  # More than just intercept
                from statsmodels.stats.outliers_influence import variance_inflation_factor
                
                vif_data = pd.DataFrame()
                vif_data["Variable"] = model.model.exog_names[1:]  # Exclude intercept
                vif_data["VIF"] = [variance_inflation_factor(model.model.exog, i) 
                                 for i in range(1, model.model.exog.shape[1])]
                
                diagnostics['multicollinearity'] = {
                    'vif_values': vif_data.to_dict('records'),
                    'max_vif': float(vif_data["VIF"].max()),
                    'multicollinearity_concern': vif_data["VIF"].max() > 10
                }
            
            # Outlier detection
            from statsmodels.stats.outliers_influence import OLSInfluence
            
            influence = OLSInfluence(model)
            cooks_d = influence.cooks_distance[0]
            leverage = influence.hat_matrix_diag
            
            # Identify outliers using Cook's distance and leverage
            n = len(data)
            p = model.model.exog.shape[1]
            
            cooks_threshold = 4 / n
            leverage_threshold = 2 * p / n
            
            outliers = {
                'cooks_distance_outliers': int(np.sum(cooks_d > cooks_threshold)),
                'high_leverage_points': int(np.sum(leverage > leverage_threshold)),
                'max_cooks_distance': float(np.max(cooks_d)),
                'max_leverage': float(np.max(leverage))
            }
            
            diagnostics['outliers'] = outliers
            
        except Exception as e:
            logger.warning(f"Some regression diagnostics failed: {e}")
            diagnostics['warning'] = str(e)
        
        return diagnostics
    
    def _compare_regression_models(self, models: Dict[str, Any]) -> Dict[str, Any]:
        """Compare different regression models."""
        comparison = {}
        
        model_names = list(models.keys())
        
        if len(model_names) > 1:
            # Compare R-squared values
            r_squared_values = {name: models[name]['r_squared'] for name in model_names}
            best_r_squared = max(r_squared_values, key=r_squared_values.get)
            
            # Compare AIC values (lower is better)
            aic_values = {name: models[name]['aic'] for name in model_names}
            best_aic = min(aic_values, key=aic_values.get)
            
            # Compare BIC values (lower is better)
            bic_values = {name: models[name]['bic'] for name in model_names}
            best_bic = min(bic_values, key=bic_values.get)
            
            comparison = {
                'r_squared_comparison': r_squared_values,
                'best_r_squared_model': best_r_squared,
                'aic_comparison': aic_values,
                'best_aic_model': best_aic,
                'bic_comparison': bic_values,
                'best_bic_model': best_bic,
                'recommended_model': self._recommend_best_model(best_aic, best_bic, best_r_squared)
            }
        
        return comparison
    
    def _interpret_confounding(self, percent_change: float) -> str:
        """Interpret the magnitude of confounding."""
        if percent_change < 10:
            return "No meaningful confounding detected"
        elif percent_change < 20:
            return "Mild confounding present - consider including confounders"
        elif percent_change < 50:
            return "Moderate confounding - confounders should be included"
        else:
            return "Severe confounding - confounders must be included"
    
    def _interpret_interaction(self, interaction_coef: float, interaction_p: float) -> str:
        """Interpret interaction effects."""
        if interaction_p >= 0.05:
            return "No significant interaction detected"
        else:
            direction = "positive" if interaction_coef > 0 else "negative"
            magnitude = "weak" if abs(interaction_coef) < 0.1 else \
                       "moderate" if abs(interaction_coef) < 0.5 else "strong"
            return f"Significant {direction} {magnitude} interaction effect"
    
    def _recommend_best_model(self, best_aic: str, best_bic: str, best_r_squared: str) -> str:
        """Recommend the best model based on multiple criteria."""
        # Count votes for each model
        votes = {}
        for model in [best_aic, best_bic, best_r_squared]:
            votes[model] = votes.get(model, 0) + 1
        
        # Find model with most votes
        best_model = max(votes, key=votes.get)
        
        if votes[best_model] >= 2:
            return best_model
        else:
            # If tie, prefer model with best AIC (better for prediction)
            return best_aic
    
    def analyze_contextual_factors(self,
                                 merged_data: pd.DataFrame,
                                 hrv_variables: List[str],
                                 contextual_variables: List[str],
                                 grouping_variables: List[str] = None) -> Dict[str, Any]:
        """
        Analyze the impact of contextual factors on HRV variables.
        
        Args:
            merged_data: Dataset with HRV and contextual variables
            hrv_variables: List of HRV outcome variables
            contextual_variables: List of contextual predictor variables
            grouping_variables: Variables for subgroup analyses
            
        Returns:
            Comprehensive contextual analysis results
        """
        try:
            results = {
                'hrv_variables': hrv_variables,
                'contextual_variables': contextual_variables,
                'variable_analyses': {},
                'correlation_analysis': {},
                'subgroup_analyses': {},
                'summary_insights': {}
            }
            
            # Individual variable analyses
            for hrv_var in hrv_variables:
                if hrv_var not in merged_data.columns:
                    continue
                    
                var_results = {}
                
                # Univariate associations
                for context_var in contextual_variables:
                    if context_var not in merged_data.columns:
                        continue
                    
                    # Determine analysis type based on variable types
                    if merged_data[context_var].dtype in ['object', 'category']:
                        # Categorical contextual variable
                        var_results[context_var] = self._analyze_categorical_context(
                            merged_data, hrv_var, context_var
                        )
                    else:
                        # Continuous contextual variable
                        var_results[context_var] = self._analyze_continuous_context(
                            merged_data, hrv_var, context_var
                        )
                
                # Multivariate analysis
                available_contexts = [var for var in contextual_variables 
                                    if var in merged_data.columns]
                
                if len(available_contexts) > 1:
                    var_results['multivariate'] = self.perform_multivariate_regression(
                        merged_data,
                        outcome_variable=hrv_var,
                        predictor_variables=available_contexts[:3],  # Limit to avoid overfitting
                        confounders=grouping_variables if grouping_variables else None
                    )
                
                results['variable_analyses'][hrv_var] = var_results
            
            # Overall correlation analysis
            all_numeric_vars = hrv_variables + [var for var in contextual_variables 
                                              if merged_data[var].dtype in ['int64', 'float64']]
            
            if len(all_numeric_vars) > 1:
                correlation_matrix = merged_data[all_numeric_vars].corr()
                results['correlation_analysis'] = {
                    'correlation_matrix': correlation_matrix.to_dict(),
                    'strongest_correlations': self._find_strongest_correlations(
                        correlation_matrix, hrv_variables, contextual_variables
                    )
                }
            
            # Subgroup analyses
            if grouping_variables:
                results['subgroup_analyses'] = self._perform_subgroup_analyses(
                    merged_data, hrv_variables, contextual_variables, grouping_variables
                )
            
            # Generate summary insights
            results['summary_insights'] = self._generate_contextual_insights(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in contextual analysis: {e}")
            return {'error': str(e)}
    
    def _analyze_categorical_context(self, data: pd.DataFrame, hrv_var: str, context_var: str) -> Dict[str, Any]:
        """Analyze association between HRV variable and categorical contextual factor."""
        try:
            # ANOVA or Kruskal-Wallis depending on normality
            groups = [group[hrv_var].dropna() for name, group in data.groupby(context_var)]
            
            if len(groups) < 2:
                return {'error': 'Insufficient groups for analysis'}
            
            # Test normality of groups
            normal_groups = all(len(group) >= 8 and stats.shapiro(group)[1] > 0.05 
                              for group in groups if len(group) >= 3)
            
            if normal_groups and all(len(group) >= 8 for group in groups):
                # ANOVA
                f_stat, p_value = stats.f_oneway(*groups)
                test_name = "ANOVA"
                statistic = f_stat
            else:
                # Kruskal-Wallis
                h_stat, p_value = stats.kruskal(*groups)
                test_name = "Kruskal-Wallis"
                statistic = h_stat
            
            # Effect size (eta-squared approximation)
            group_means = [np.mean(group) for group in groups]
            overall_mean = np.mean([val for group in groups for val in group])
            ss_between = sum(len(group) * (np.mean(group) - overall_mean)**2 for group in groups)
            ss_total = sum((val - overall_mean)**2 for group in groups for val in group)
            eta_squared = ss_between / ss_total if ss_total > 0 else 0
            
            # Post-hoc comparisons if significant
            post_hoc = None
            if p_value < 0.05 and len(groups) > 2:
                post_hoc = self._perform_post_hoc_comparisons(data, hrv_var, context_var)
            
            return {
                'test': test_name,
                'statistic': statistic,
                'p_value': p_value,
                'eta_squared': eta_squared,
                'group_statistics': {
                    f'group_{i}': {'mean': np.mean(group), 'std': np.std(group), 'n': len(group)}
                    for i, group in enumerate(groups)
                },
                'post_hoc': post_hoc,
                'interpretation': self._interpret_group_differences(p_value, eta_squared)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _analyze_continuous_context(self, data: pd.DataFrame, hrv_var: str, context_var: str) -> Dict[str, Any]:
        """Analyze association between HRV variable and continuous contextual factor."""
        try:
            clean_data = data[[hrv_var, context_var]].dropna()
            
            if len(clean_data) < 10:
                return {'error': 'Insufficient data for analysis'}
            
            x = clean_data[context_var]
            y = clean_data[hrv_var]
            
            # Pearson and Spearman correlations
            pearson_r, pearson_p = stats.pearsonr(x, y)
            spearman_rho, spearman_p = stats.spearmanr(x, y)
            
            # Simple linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            
            # Determine which correlation to report based on linearity
            linearity_appropriate = abs(pearson_r) >= abs(spearman_rho) * 0.9
            
            return {
                'pearson_correlation': {'r': pearson_r, 'p_value': pearson_p},
                'spearman_correlation': {'rho': spearman_rho, 'p_value': spearman_p},
                'linear_regression': {
                    'slope': slope,
                    'intercept': intercept,
                    'r_squared': r_value**2,
                    'p_value': p_value,
                    'standard_error': std_err
                },
                'recommended_test': 'pearson' if linearity_appropriate else 'spearman',
                'sample_size': len(clean_data),
                'interpretation': self._interpret_correlation(
                    pearson_r if linearity_appropriate else spearman_rho,
                    pearson_p if linearity_appropriate else spearman_p
                )
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _find_strongest_correlations(self, corr_matrix: pd.DataFrame, 
                                   hrv_vars: List[str], context_vars: List[str]) -> List[Dict]:
        """Find strongest correlations between HRV and contextual variables."""
        strongest_correlations = []
        
        for hrv_var in hrv_vars:
            if hrv_var not in corr_matrix.columns:
                continue
                
            for context_var in context_vars:
                if context_var not in corr_matrix.columns:
                    continue
                    
                corr_value = corr_matrix.loc[hrv_var, context_var]
                if not np.isnan(corr_value):
                    strongest_correlations.append({
                        'hrv_variable': hrv_var,
                        'contextual_variable': context_var,
                        'correlation': corr_value,
                        'abs_correlation': abs(corr_value)
                    })
        
        # Sort by absolute correlation strength
        strongest_correlations.sort(key=lambda x: x['abs_correlation'], reverse=True)
        
        return strongest_correlations[:10]  # Top 10 strongest correlations
    
    def _perform_post_hoc_comparisons(self, data: pd.DataFrame, hrv_var: str, context_var: str) -> Dict[str, Any]:
        """Perform post-hoc pairwise comparisons."""
        try:
            if HAS_PINGOUIN:
                import pingouin as pg
                post_hoc = pg.pairwise_tukey(data=data, dv=hrv_var, between=context_var)
                return {
                    'method': 'Tukey HSD',
                    'results': post_hoc.to_dict('records')
                }
            else:
                # Manual pairwise t-tests with Bonferroni correction
                groups = data.groupby(context_var)[hrv_var].apply(list).to_dict()
                group_names = list(groups.keys())
                
                comparisons = []
                n_comparisons = len(group_names) * (len(group_names) - 1) // 2
                
                for i in range(len(group_names)):
                    for j in range(i+1, len(group_names)):
                        group1 = groups[group_names[i]]
                        group2 = groups[group_names[j]]
                        
                        t_stat, p_val = stats.ttest_ind(group1, group2)
                        adjusted_p = min(1.0, p_val * n_comparisons)  # Bonferroni correction
                        
                        comparisons.append({
                            'group1': group_names[i],
                            'group2': group_names[j],
                            't_statistic': t_stat,
                            'p_value': p_val,
                            'adjusted_p_value': adjusted_p,
                            'significant': adjusted_p < 0.05
                        })
                
                return {
                    'method': 'Pairwise t-tests with Bonferroni correction',
                    'results': comparisons
                }
                
        except Exception as e:
            return {'error': str(e)}
    
    def _perform_subgroup_analyses(self, data: pd.DataFrame, hrv_vars: List[str], 
                                 context_vars: List[str], group_vars: List[str]) -> Dict[str, Any]:
        """Perform subgroup analyses based on grouping variables."""
        subgroup_results = {}
        
        for group_var in group_vars:
            if group_var not in data.columns:
                continue
                
            subgroup_results[group_var] = {}
            
            for group_value in data[group_var].unique():
                if pd.isna(group_value):
                    continue
                    
                subgroup_data = data[data[group_var] == group_value]
                
                if len(subgroup_data) < 20:  # Minimum sample size for subgroup
                    continue
                
                subgroup_analysis = self.analyze_contextual_factors(
                    subgroup_data, hrv_vars[:2], context_vars[:2]  # Limit for computational efficiency
                )
                
                subgroup_results[group_var][str(group_value)] = {
                    'n': len(subgroup_data),
                    'analysis': subgroup_analysis
                }
        
        return subgroup_results
    
    def _generate_contextual_insights(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary insights from contextual analysis."""
        insights = {
            'key_findings': [],
            'methodological_notes': [],
            'recommendations': []
        }
        
        # Analyze strongest associations
        if 'correlation_analysis' in results and 'strongest_correlations' in results['correlation_analysis']:
            strongest = results['correlation_analysis']['strongest_correlations'][:3]
            
            for corr in strongest:
                if corr['abs_correlation'] > 0.5:
                    direction = "positive" if corr['correlation'] > 0 else "negative"
                    insights['key_findings'].append(
                        f"Strong {direction} association between {corr['hrv_variable']} "
                        f"and {corr['contextual_variable']} (r = {corr['correlation']:.3f})"
                    )
        
        # Analyze multivariate results
        for hrv_var, analyses in results.get('variable_analyses', {}).items():
            if 'multivariate' in analyses and 'models' in analyses['multivariate']:
                models = analyses['multivariate']['models']
                
                if 'adjusted' in models:
                    r_squared = models['adjusted']['r_squared']
                    if r_squared > 0.2:
                        insights['key_findings'].append(
                            f"Contextual factors explain {r_squared*100:.1f}% "
                            f"of variance in {hrv_var}"
                        )
        
        # Methodological recommendations
        insights['methodological_notes'] = [
            "Multiple comparisons performed - consider family-wise error correction",
            "Causal inference limited by observational design",
            "Results may be influenced by unmeasured confounders"
        ]
        
        insights['recommendations'] = [
            "Validate findings in independent dataset",
            "Consider longitudinal analysis if temporal data available",
            "Explore potential mediating mechanisms"
        ]
        
        return insights
    
    def _interpret_group_differences(self, p_value: float, eta_squared: float) -> str:
        """Interpret group differences."""
        if p_value >= 0.05:
            return "No significant group differences detected"
        
        effect_size = "small" if eta_squared < 0.06 else \
                     "medium" if eta_squared < 0.14 else "large"
        
        return f"Significant group differences with {effect_size} effect size (η² = {eta_squared:.3f})"
    
    def _interpret_correlation(self, correlation: float, p_value: float) -> str:
        """Interpret correlation results."""
        if p_value >= 0.05:
            return "No significant association detected"
        
        abs_corr = abs(correlation)
        strength = "weak" if abs_corr < 0.3 else \
                  "moderate" if abs_corr < 0.7 else "strong"
        
        direction = "positive" if correlation > 0 else "negative"
        
        return f"Significant {direction} {strength} association (r = {correlation:.3f})"

# Provide backwards-compatible alias expected by tests
AdvancedHRVStatistics = AdvancedStats