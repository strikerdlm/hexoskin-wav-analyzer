"""
HRV Time Series Forecasting using ARIMA Models

This module provides time series forecasting capabilities for HRV metrics including:
- ARIMA model fitting and automatic parameter selection
- Seasonal ARIMA (SARIMA) for data with seasonal patterns
- Prophet-based forecasting for long-term trend analysis  
- Cross-validation and forecast accuracy assessment
- SOL trend prediction and autonomic adaptation forecasting
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Dict, Any, Optional, List, Tuple
import logging
from dataclasses import dataclass
import warnings

logger = logging.getLogger(__name__)

# Time series analysis imports (with fallbacks)
try:
    import pmdarima as pm
    from pmdarima import auto_arima
    HAS_PMDARIMA = True
    logger.info("pmdarima successfully imported - automatic ARIMA selection available")
except ImportError:
    HAS_PMDARIMA = False
    logger.warning("pmdarima not available")
    
try:
    from prophet import Prophet
    HAS_PROPHET = True
    logger.info("Prophet successfully imported - Prophet forecasting available")
except ImportError:
    HAS_PROPHET = False
    logger.warning("Prophet not available")
    
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.tsa.stattools import adfuller, kpss
    HAS_STATSMODELS = True
    logger.info("statsmodels available for ARIMA modeling")
except ImportError:
    HAS_STATSMODELS = False
    logger.warning("statsmodels not available")

@dataclass
class ForecastResult:
    """Container for forecasting results."""
    forecast_values: np.ndarray
    forecast_dates: pd.DatetimeIndex
    confidence_intervals: Tuple[np.ndarray, np.ndarray]
    model_name: str
    model_params: Dict[str, Any]
    fit_metrics: Dict[str, float]
    residual_diagnostics: Dict[str, Any]
    
@dataclass
class ModelComparison:
    """Container for model comparison results."""
    models: Dict[str, Any]
    performance_metrics: pd.DataFrame
    best_model_name: str
    ensemble_forecast: np.ndarray
    ensemble_confidence: Tuple[np.ndarray, np.ndarray]

class HRVForecasting:
    """Advanced time series forecasting for HRV metrics."""
    
    def __init__(self, 
                 seasonal_periods: int = 4,  # Assuming 4 SOLs per seasonal cycle
                 confidence_level: float = 0.95):
        """
        Initialize HRV forecasting analyzer.
        
        Args:
            seasonal_periods: Number of periods in a seasonal cycle
            confidence_level: Confidence level for forecast intervals
        """
        self.seasonal_periods = seasonal_periods
        self.confidence_level = confidence_level
        self.fitted_models = {}
        
        # Log availability status
        capabilities = []
        if HAS_STATSMODELS:
            capabilities.append("ARIMA")
        if HAS_PMDARIMA:
            capabilities.append("Auto-ARIMA")
        if HAS_PROPHET:
            capabilities.append("Prophet")
        
        logger.info(f"HRVForecasting initialized with capabilities: {', '.join(capabilities) if capabilities else 'Basic only'}")
            
    def prepare_time_series(self,
                          hrv_data: pd.DataFrame,
                          value_col: str,
                          time_col: str,
                          subject_col: str = 'subject',
                          aggregation: str = 'mean') -> Dict[str, pd.Series]:
        """
        Prepare time series data for forecasting.
        
        Args:
            hrv_data: DataFrame with HRV metrics
            value_col: Column name for the metric to forecast
            time_col: Column name for time variable (e.g., 'Sol')
            subject_col: Column name for subject identifier
            aggregation: How to aggregate multiple subjects ('mean', 'median', 'individual')
            
        Returns:
            Dictionary mapping subject IDs to time series
        """
        try:
            time_series = {}
            
            if aggregation == 'individual':
                # Create separate time series for each subject
                for subject in hrv_data[subject_col].unique():
                    subject_data = hrv_data[hrv_data[subject_col] == subject].copy()
                    subject_data = subject_data.sort_values(time_col)
                    
                    if len(subject_data) >= 3:  # Minimum for time series
                        ts = pd.Series(
                            subject_data[value_col].values,
                            index=subject_data[time_col].values,
                            name=f"{subject}_{value_col}"
                        )
                        time_series[subject] = ts.dropna()
                        
            else:
                # Aggregate across subjects
                if aggregation == 'mean':
                    agg_func = 'mean'
                elif aggregation == 'median':
                    agg_func = 'median'
                else:
                    agg_func = 'mean'
                    
                grouped = hrv_data.groupby(time_col)[value_col].agg(agg_func)
                
                if len(grouped) >= 3:
                    ts = pd.Series(grouped.values, index=grouped.index, name=f"aggregated_{value_col}")
                    time_series['aggregated'] = ts.dropna()
                    
            logger.info(f"Prepared {len(time_series)} time series for forecasting")
            return time_series
            
        except Exception as e:
            logger.error(f"Error preparing time series: {e}")
            return {}
            
    def fit_arima_model(self,
					  ts_data: pd.Series,
					  order: Tuple[int, int, int] = None,
					  seasonal_order: Tuple[int, int, int, int] = None,
					  auto_select: bool = True,
					  forecast_steps: int = None) -> ForecastResult:
		"""
		Fit ARIMA model to time series data.
		
		Args:
		    ts_data: Time series data
		    order: ARIMA order (p, d, q)
		    seasonal_order: Seasonal ARIMA order (P, D, Q, s)
		    auto_select: Whether to automatically select optimal parameters
		    forecast_steps: Optional number of steps to forecast (compatibility)
		"""
		if not HAS_STATSMODELS:
			logger.error("statsmodels required for ARIMA modeling")
			return self._create_empty_forecast_result("ARIMA")
		try:
			stationarity_results = self._check_stationarity(ts_data)
			logger.info(f"Stationarity test results: {stationarity_results}")
			if auto_select and HAS_PMDARIMA:
				model = auto_arima(
					ts_data,
					seasonal=True if seasonal_order else False,
					m=self.seasonal_periods,
					stepwise=True,
					suppress_warnings=True,
					error_action='ignore',
					max_p=3, max_q=3, max_P=2, max_Q=2,
					max_d=2, max_D=1,
					information_criterion='aic',
					trace=False
				)
				fitted_model = model
				model_params = {
					'order': model.order,
					'seasonal_order': model.seasonal_order,
					'aic': model.aic(),
					'bic': model.bic()
				}
			else:
				if order is None:
					order = (1, 1, 1)
				from statsmodels.tsa.arima.model import ARIMA
				if seasonal_order is not None:
					from statsmodels.tsa.statespace.sarimax import SARIMAX
					fitted_model = SARIMAX(ts_data, order=order, seasonal_order=seasonal_order)
				else:
					fitted_model = ARIMA(ts_data, order=order)
				fitted_model = fitted_model.fit()
				model_params = {
					'order': order,
					'seasonal_order': seasonal_order,
					'aic': fitted_model.aic,
					'bic': fitted_model.bic
				}
			# Generate forecast
			steps = forecast_steps if isinstance(forecast_steps, int) and forecast_steps > 0 else min(len(ts_data) // 2, 10)
			forecast_result = fitted_model.get_forecast(steps=steps)
			forecast_values = forecast_result.predicted_mean
			confidence_intervals = forecast_result.conf_int()
			# Create forecast dates
			if isinstance(ts_data.index, pd.DatetimeIndex):
				last_date = ts_data.index[-1]
				forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=steps, freq='D')
			else:
				last_index = ts_data.index[-1]
				forecast_dates = pd.Index(range(last_index + 1, last_index + 1 + steps))
			# Compute fit metrics
			fitted_values = fitted_model.fittedvalues
			residuals = fitted_model.resid
			fit_metrics = {
				'aic': float(fitted_model.aic if hasattr(fitted_model, 'aic') else model_params['aic']),
				'bic': float(fitted_model.bic if hasattr(fitted_model, 'bic') else model_params['bic']),
				'rmse': float(np.sqrt(np.mean(residuals**2))),
				'mae': float(np.mean(np.abs(residuals)))
			}
			residual_diagnostics = self._compute_residual_diagnostics(residuals)
			self.fitted_models[f"arima_{ts_data.name}"] = fitted_model
			return ForecastResult(
				forecast_values=forecast_values.values,
				forecast_dates=forecast_dates,
				confidence_intervals=(
					confidence_intervals.iloc[:, 0].values,
					confidence_intervals.iloc[:, 1].values
				),
				model_name="ARIMA",
				model_params=model_params,
				fit_metrics=fit_metrics,
				residual_diagnostics=residual_diagnostics
			)
		except Exception as e:
			logger.error(f"Error fitting ARIMA model: {e}")
			return self._create_empty_forecast_result("ARIMA")

	def fit_prophet_model(self,
					   ts_data: pd.DataFrame | pd.Series,
					   include_seasonality: bool = True,
					   yearly_seasonality: bool = False,
					   forecast_days: int = 14,
					   include_weekly_seasonality: bool = True,
					   include_yearly_seasonality: bool = False) -> ForecastResult:
		"""Fit Prophet model for trend and seasonality analysis. Compatible with tests accepting DataFrame with 'ds','y'."""
		if not HAS_PROPHET:
			logger.error("Prophet library not available")
			return self._create_empty_forecast_result("Prophet")
		try:
			# Normalize input
			if isinstance(ts_data, pd.Series):
				prophet_df = pd.DataFrame({'ds': pd.to_datetime(ts_data.index) if not isinstance(ts_data.index, pd.DatetimeIndex) else ts_data.index, 'y': ts_data.values})
			else:
				prophet_df = ts_data.rename(columns={'y': 'y', 'ds': 'ds'})
			# Initialize Prophet model
			model = Prophet(
				interval_width=self.confidence_level,
				daily_seasonality=False,
				weekly_seasonality=include_weekly_seasonality,
				yearly_seasonality=include_yearly_seasonality or yearly_seasonality
			)
			if include_seasonality and len(prophet_df) >= 2 * self.seasonal_periods:
				model.add_seasonality(name='sol_cycle', period=self.seasonal_periods, fourier_order=2)
			model.fit(prophet_df)
			future = model.make_future_dataframe(periods=forecast_days, freq='D')
			forecast = model.predict(future)
			forecast_values = forecast['yhat'].tail(forecast_days).values
			forecast_dates = forecast['ds'].tail(forecast_days)
			confidence_intervals = (
				forecast['yhat_lower'].tail(forecast_days).values,
				forecast['yhat_upper'].tail(forecast_days).values
			)
			fitted_values = forecast['yhat'].iloc[:-forecast_days].values
			residuals = prophet_df['y'].values - fitted_values[: len(prophet_df['y'].values)]
			fit_metrics = {
				'rmse': float(np.sqrt(np.mean(residuals**2))),
				'mae': float(np.mean(np.abs(residuals)))
			}
			residual_diagnostics = self._compute_residual_diagnostics(pd.Series(residuals))
			model_params = {
				'interval_width': self.confidence_level,
				'weekly_seasonality': include_weekly_seasonality,
				'yearly_seasonality': include_yearly_seasonality or yearly_seasonality
			}
			self.fitted_models["prophet"] = model
			return ForecastResult(
				forecast_values=forecast_values,
				forecast_dates=forecast_dates,
				confidence_intervals=confidence_intervals,
				model_name="Prophet",
				model_params=model_params,
				fit_metrics=fit_metrics,
				residual_diagnostics=residual_diagnostics
			)
		except Exception as e:
			logger.error(f"Error fitting Prophet model: {e}")
			return self._create_empty_forecast_result("Prophet")

	def compare_models(self,
				   ts_data: pd.Series,
				   models: List[str] = None,
				   cross_validation: bool = True,
				   test_size: float = 0.2,
				   forecast_horizon: int = 7) -> ModelComparison:
		"""Compare multiple forecasting models. Extra params kept for compatibility (ignored)."""
		try:
			if models is None:
				models = ['arima']
				if HAS_PROPHET:
					models.append('prophet')
			results = {}
			performance_metrics = []
			for model_name in models:
				try:
					if model_name == 'arima':
						result = self.fit_arima_model(ts_data)
					elif model_name == 'prophet':
						result = self.fit_prophet_model(ts_data)
					elif model_name == 'linear_trend':
						result = self._fit_exponential_smoothing(ts_data)
					else:
						continue
					results[model_name] = result
					performance_metrics.append({
						'model': model_name,
						'rmse': result.fit_metrics.get('rmse', float('inf')),
						'mae': result.fit_metrics.get('mae', float('inf')),
						'mape': result.fit_metrics.get('mape', float('inf')),
						'aic': result.fit_metrics.get('aic', float('inf')),
						'bic': result.fit_metrics.get('bic', float('inf'))
					})
				except Exception as e:
					logger.warning(f"Error fitting {model_name}: {e}")
			if not performance_metrics:
				logger.error("No models could be fitted successfully")
				return ModelComparison({}, pd.DataFrame(), "", np.array([]), (np.array([]), np.array([])))
			perf_df = pd.DataFrame(performance_metrics).sort_values('rmse')
			best_model_name = perf_df.iloc[0]['model']
			if len(results) > 1:
				all_forecasts = []
				all_lower_bounds = []
				all_upper_bounds = []
				for r in results.values():
					if len(r.forecast_values) > 0:
						all_forecasts.append(r.forecast_values)
						all_lower_bounds.append(r.confidence_intervals[0])
						all_upper_bounds.append(r.confidence_intervals[1])
				if all_forecasts:
					min_length = min(len(f) for f in all_forecasts)
					all_forecasts = [f[:min_length] for f in all_forecasts]
					all_lower_bounds = [f[:min_length] for f in all_lower_bounds]
					all_upper_bounds = [f[:min_length] for f in all_upper_bounds]
					ensemble_forecast = np.mean(all_forecasts, axis=0)
					ensemble_lower = np.mean(all_lower_bounds, axis=0)
					ensemble_upper = np.mean(all_upper_bounds, axis=0)
				else:
					ensemble_forecast = np.array([])
					ensemble_lower = ensemble_upper = np.array([])
			else:
				best_result = results[best_model_name]
				ensemble_forecast = best_result.forecast_values
				ensemble_lower, ensemble_upper = best_result.confidence_intervals
			return ModelComparison(results, perf_df, best_model_name, ensemble_forecast, (ensemble_lower, ensemble_upper))
		except Exception as e:
			logger.error(f"Error comparing models: {e}")
			return ModelComparison({}, pd.DataFrame(), "", np.array([]), (np.array([]), np.array([])))
            
    def predict_sol_adaptation(self,
                              hrv_time_series: Dict[str, pd.Series],
                              adaptation_metric: str = 'rmssd',
                              forecast_sols: int = 5) -> Dict[str, Any]:
        """
        Predict HRV adaptation trends across SOL missions.
        
        Args:
            hrv_time_series: Dictionary of HRV time series by subject
            adaptation_metric: HRV metric to analyze for adaptation
            forecast_sols: Number of future SOLs to forecast
            
        Returns:
            SOL adaptation predictions
        """
        try:
            adaptation_results = {}
            
            for subject_id, ts_data in hrv_time_series.items():
                if len(ts_data) < 4:  # Need minimum data for trend analysis
                    continue
                    
                # Fit model for this subject
                comparison = self.compare_models(ts_data)
                
                if comparison.best_model_name:
                    best_result = comparison.models[comparison.best_model_name]
                    
                    # Analyze adaptation pattern
                    adaptation_analysis = self._analyze_adaptation_pattern(ts_data, best_result)
                    
                    # Predict future adaptation
                    future_prediction = self._predict_future_adaptation(
                        ts_data, best_result, forecast_sols
                    )
                    
                    adaptation_results[subject_id] = {
                        'current_trend': adaptation_analysis['trend_direction'],
                        'adaptation_rate': adaptation_analysis['adaptation_rate'],
                        'adaptation_phase': adaptation_analysis['adaptation_phase'],
                        'forecast_values': future_prediction['forecast_values'],
                        'forecast_confidence': future_prediction['confidence_intervals'],
                        'predicted_adaptation': future_prediction['adaptation_prediction'],
                        'model_performance': comparison.performance_metrics.iloc[0].to_dict()
                    }
                    
            # Aggregate results
            if adaptation_results:
                group_trends = self._aggregate_adaptation_trends(adaptation_results)
                adaptation_results['group_summary'] = group_trends
                
            return adaptation_results
            
        except Exception as e:
            logger.error(f"Error predicting SOL adaptation: {e}")
            return {'error': str(e)}
            
    def _check_stationarity(self, ts_data: pd.Series) -> Dict[str, Any]:
        """Check time series stationarity using ADF and KPSS tests."""
        try:
            results = {}
            
            if HAS_STATSMODELS:
                # Augmented Dickey-Fuller test
                adf_result = adfuller(ts_data.dropna())
                results['adf'] = {
                    'statistic': adf_result[0],
                    'p_value': adf_result[1],
                    'is_stationary': adf_result[1] < 0.05
                }
                
                # KPSS test
                kpss_result = kpss(ts_data.dropna())
                results['kpss'] = {
                    'statistic': kpss_result[0],
                    'p_value': kpss_result[1],
                    'is_stationary': kpss_result[1] > 0.05
                }
                
            return results
            
        except Exception as e:
            logger.error(f"Error checking stationarity: {e}")
            return {}
            
    def _compute_residual_diagnostics(self, residuals: pd.Series) -> Dict[str, Any]:
        """Compute diagnostic statistics for model residuals."""
        try:
            diagnostics = {}
            
            # Basic residual statistics
            diagnostics['mean_residual'] = float(residuals.mean())
            diagnostics['std_residual'] = float(residuals.std())
            diagnostics['skewness'] = float(residuals.skew())
            diagnostics['kurtosis'] = float(residuals.kurtosis())
            
            # Ljung-Box test for residual autocorrelation
            if HAS_STATSMODELS and len(residuals) > 10:
                try:
                    lb_result = acorr_ljungbox(residuals, lags=min(10, len(residuals)//4))
                    diagnostics['ljung_box_pvalue'] = float(lb_result['lb_pvalue'].iloc[-1])
                    diagnostics['residuals_uncorrelated'] = diagnostics['ljung_box_pvalue'] > 0.05
                except:
                    diagnostics['ljung_box_pvalue'] = 1.0
                    diagnostics['residuals_uncorrelated'] = True
                    
            return diagnostics
            
        except Exception as e:
            logger.error(f"Error computing residual diagnostics: {e}")
            return {}
            
    def _analyze_adaptation_pattern(self, ts_data: pd.Series, forecast_result: ForecastResult) -> Dict[str, Any]:
        """Analyze autonomic adaptation pattern."""
        try:
            # Linear trend analysis
            x = np.arange(len(ts_data))
            trend_slope, trend_intercept = np.polyfit(x, ts_data.values, 1)
            
            # Determine trend direction
            if abs(trend_slope) < 0.01 * np.std(ts_data):
                trend_direction = 'stable'
            elif trend_slope > 0:
                trend_direction = 'increasing'
            else:
                trend_direction = 'decreasing'
                
            # Adaptation rate (change per SOL)
            adaptation_rate = abs(trend_slope)
            
            # Determine adaptation phase
            if len(ts_data) < 4:
                adaptation_phase = 'insufficient_data'
            else:
                # Look at first half vs second half
                mid_point = len(ts_data) // 2
                first_half_mean = ts_data.iloc[:mid_point].mean()
                second_half_mean = ts_data.iloc[mid_point:].mean()
                
                change_magnitude = abs(second_half_mean - first_half_mean) / first_half_mean
                
                if change_magnitude < 0.1:
                    adaptation_phase = 'adapted'
                elif trend_direction == 'stable':
                    adaptation_phase = 'stabilizing'
                else:
                    adaptation_phase = 'adapting'
                    
            return {
                'trend_direction': trend_direction,
                'trend_slope': trend_slope,
                'adaptation_rate': adaptation_rate,
                'adaptation_phase': adaptation_phase,
                'r_squared': np.corrcoef(x, ts_data.values)[0, 1]**2
            }
            
        except Exception as e:
            logger.error(f"Error analyzing adaptation pattern: {e}")
            return {}
            
    def _predict_future_adaptation(self, ts_data: pd.Series, forecast_result: ForecastResult, forecast_sols: int) -> Dict[str, Any]:
        """Predict future adaptation based on current trends."""
        try:
            # Use forecast values
            forecast_values = forecast_result.forecast_values[:forecast_sols]
            confidence_intervals = (
                forecast_result.confidence_intervals[0][:forecast_sols],
                forecast_result.confidence_intervals[1][:forecast_sols]
            )
            
            # Predict adaptation outcome
            current_mean = ts_data.iloc[-3:].mean()  # Recent baseline
            forecast_mean = np.mean(forecast_values)
            
            relative_change = (forecast_mean - current_mean) / current_mean
            
            if abs(relative_change) < 0.05:
                adaptation_prediction = 'stable_adaptation'
            elif relative_change > 0.1:
                adaptation_prediction = 'positive_adaptation'
            elif relative_change < -0.1:
                adaptation_prediction = 'negative_adaptation'
            else:
                adaptation_prediction = 'mild_change'
                
            return {
                'forecast_values': forecast_values,
                'confidence_intervals': confidence_intervals,
                'relative_change': relative_change,
                'adaptation_prediction': adaptation_prediction
            }
            
        except Exception as e:
            logger.error(f"Error predicting future adaptation: {e}")
            return {}
            
    def _aggregate_adaptation_trends(self, individual_results: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate individual adaptation trends for group analysis."""
        try:
            # Collect trends from all subjects
            trends = []
            phases = []
            rates = []
            
            for subject_id, results in individual_results.items():
                if isinstance(results, dict) and 'current_trend' in results:
                    trends.append(results['current_trend'])
                    phases.append(results['adaptation_phase'])
                    rates.append(results['adaptation_rate'])
                    
            # Summarize group patterns
            from collections import Counter
            trend_distribution = Counter(trends)
            phase_distribution = Counter(phases)
            
            group_summary = {
                'dominant_trend': trend_distribution.most_common(1)[0][0] if trend_distribution else 'unknown',
                'trend_distribution': dict(trend_distribution),
                'dominant_phase': phase_distribution.most_common(1)[0][0] if phase_distribution else 'unknown',
                'phase_distribution': dict(phase_distribution),
                'mean_adaptation_rate': np.mean(rates) if rates else 0,
                'n_subjects_analyzed': len(trends)
            }
            
            return group_summary
            
        except Exception as e:
            logger.error(f"Error aggregating adaptation trends: {e}")
            return {}
            
    def _fit_exponential_smoothing(self, ts_data: pd.Series) -> ForecastResult:
        """Fit exponential smoothing model (fallback method)."""
        try:
            # Simple exponential smoothing implementation
            alpha = 0.3  # Smoothing parameter
            
            smoothed = [ts_data.iloc[0]]  # Initialize with first value
            
            for i in range(1, len(ts_data)):
                smoothed.append(alpha * ts_data.iloc[i] + (1 - alpha) * smoothed[i-1])
                
            # Generate forecast
            forecast_steps = min(len(ts_data) // 2, 5)
            last_value = smoothed[-1]
            forecast_values = np.full(forecast_steps, last_value)
            
            # Simple confidence intervals (±10%)
            confidence_width = 0.1 * abs(last_value)
            confidence_intervals = (
                forecast_values - confidence_width,
                forecast_values + confidence_width
            )
            
            # Create forecast dates
            if isinstance(ts_data.index, pd.DatetimeIndex):
                last_date = ts_data.index[-1]
                forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_steps, freq='D')
            else:
                last_index = ts_data.index[-1]
                forecast_dates = pd.Index(range(last_index + 1, last_index + 1 + forecast_steps))
                
            # Compute fit metrics
            residuals = ts_data.values - np.array(smoothed)
            fit_metrics = {
                'rmse': float(np.sqrt(np.mean(residuals**2))),
                'mae': float(np.mean(np.abs(residuals)))
            }
            
            model_params = {'alpha': alpha}
            
            return ForecastResult(
                forecast_values=forecast_values,
                forecast_dates=forecast_dates,
                confidence_intervals=confidence_intervals,
                model_name="Exponential Smoothing",
                model_params=model_params,
                fit_metrics=fit_metrics,
                residual_diagnostics=self._compute_residual_diagnostics(pd.Series(residuals))
            )
            
        except Exception as e:
            logger.error(f"Error in exponential smoothing: {e}")
            return self._create_empty_forecast_result("Exponential Smoothing")
            
    def _create_empty_forecast_result(self, model_name: str) -> ForecastResult:
        """Create empty forecast result for error cases."""
        return ForecastResult(
            forecast_values=np.array([]),
            forecast_dates=pd.DatetimeIndex([]),
            confidence_intervals=(np.array([]), np.array([])),
            model_name=model_name,
            model_params={},
            fit_metrics={},
            residual_diagnostics={}
        ) 

    def auto_arima_forecast(self,
                           time_series: pd.Series,
                           steps_ahead: int = 6,
                           seasonal: bool = True,
                           suppress_warnings: bool = True) -> Optional[ForecastResult]:
        """
        Advanced automatic ARIMA model selection and forecasting using pmdarima.
        
        Args:
            time_series: Input time series data
            steps_ahead: Number of steps to forecast ahead
            seasonal: Whether to use seasonal ARIMA
            suppress_warnings: Whether to suppress pmdarima warnings
            
        Returns:
            ForecastResult with forecast and diagnostics
        """
        if not HAS_PMDARIMA:
            logger.error("pmdarima not available. Cannot perform auto-ARIMA forecasting.")
            return None
            
        try:
            logger.info(f"Starting auto-ARIMA analysis for time series with {len(time_series)} points")
            
            # Prepare data
            ts_clean = time_series.dropna()
            if len(ts_clean) < 10:
                logger.warning(f"Insufficient data for ARIMA modeling: {len(ts_clean)} points")
                return None
            
            # Configure seasonal parameters
            seasonal_config = {}
            if seasonal and len(ts_clean) >= 2 * self.seasonal_periods:
                seasonal_config = {
                    'seasonal': True,
                    'm': self.seasonal_periods,
                    'max_P': 2,
                    'max_D': 1,
                    'max_Q': 2
                }
                logger.info(f"Using seasonal ARIMA with period {self.seasonal_periods}")
            else:
                seasonal_config = {'seasonal': False}
                logger.info("Using non-seasonal ARIMA")
            
            with warnings.catch_warnings():
                if suppress_warnings:
                    warnings.simplefilter("ignore")
                
                # Auto ARIMA model selection
                model = auto_arima(
                    ts_clean,
                    start_p=0, start_q=0,
                    max_p=5, max_q=5,
                    start_P=0, start_Q=0,
                    max_order=10,
                    stepwise=True,
                    suppress_warnings=suppress_warnings,
                    error_action='ignore',
                    trace=False,
                    **seasonal_config
                )
                
            logger.info(f"Selected model: {model.order} {model.seasonal_order if hasattr(model, 'seasonal_order') else ''}")
            
            # Generate forecasts
            forecast, conf_int = model.predict(n_periods=steps_ahead, 
                                             return_conf_int=True,
                                             alpha=1-self.confidence_level)
            
            # Create forecast dates
            last_date = ts_clean.index[-1] if hasattr(ts_clean.index, 'date') else len(ts_clean)
            if isinstance(last_date, (int, np.integer)):
                forecast_dates = pd.RangeIndex(start=last_date+1, stop=last_date+1+steps_ahead)
            else:
                freq = pd.infer_freq(ts_clean.index)
                if freq is None:
                    # Fallback for irregular time series
                    freq = pd.Timedelta(days=1)
                forecast_dates = pd.date_range(start=last_date + freq, periods=steps_ahead, freq=freq)
            
            # Calculate fit metrics
            fitted_values = model.predict_in_sample()
            residuals = ts_clean - fitted_values
            
            fit_metrics = {
                'aic': model.aic(),
                'bic': model.bic(),
                'rmse': np.sqrt(mean_squared_error(ts_clean, fitted_values)),
                'mae': mean_absolute_error(ts_clean, fitted_values),
                'mape': np.mean(np.abs((ts_clean - fitted_values) / ts_clean)) * 100
            }
            
            # Residual diagnostics
            residual_diagnostics = self._calculate_residual_diagnostics(residuals)
            
            # Model parameters
            model_params = {
                'order': model.order,
                'seasonal_order': getattr(model, 'seasonal_order', None),
                'n_training_points': len(ts_clean)
            }
            
            logger.info(f"Auto-ARIMA completed: AIC={fit_metrics['aic']:.2f}, RMSE={fit_metrics['rmse']:.3f}")
            
            return ForecastResult(
                forecast_values=forecast,
                forecast_dates=forecast_dates,
                confidence_intervals=(conf_int[:, 0], conf_int[:, 1]),
                model_name=f"Auto-ARIMA{model.order}",
                model_params=model_params,
                fit_metrics=fit_metrics,
                residual_diagnostics=residual_diagnostics
            )
            
        except Exception as e:
            logger.error(f"Error in auto-ARIMA forecasting: {e}")
            return None
    
    def prophet_forecast(self,
                        time_series: pd.Series,
                        steps_ahead: int = 6,
                        include_holidays: bool = False,
                        growth: str = 'linear') -> Optional[ForecastResult]:
        """
        Prophet-based forecasting with trend and seasonality detection.
        
        Args:
            time_series: Input time series with datetime index
            steps_ahead: Number of steps to forecast ahead
            include_holidays: Whether to include holiday effects
            growth: Growth model ('linear', 'logistic')
            
        Returns:
            ForecastResult with Prophet forecast and diagnostics
        """
        if not HAS_PROPHET:
            logger.error("Prophet not available. Cannot perform Prophet forecasting.")
            return None
            
        try:
            logger.info(f"Starting Prophet analysis for time series with {len(time_series)} points")
            
            # Prepare data for Prophet
            ts_clean = time_series.dropna()
            if len(ts_clean) < 10:
                logger.warning(f"Insufficient data for Prophet modeling: {len(ts_clean)} points")
                return None
            
            # Convert to Prophet format
            if hasattr(ts_clean.index, 'date'):
                df_prophet = pd.DataFrame({
                    'ds': ts_clean.index,
                    'y': ts_clean.values
                })
            else:
                # Handle non-datetime index
                dates = pd.date_range(start='2023-01-01', periods=len(ts_clean), freq='D')
                df_prophet = pd.DataFrame({
                    'ds': dates,
                    'y': ts_clean.values
                })
                logger.info("Created synthetic datetime index for Prophet")
            
            # Initialize Prophet model
            prophet_params = {
                'growth': growth,
                'daily_seasonality': False,
                'weekly_seasonality': len(ts_clean) >= 14,
                'yearly_seasonality': len(ts_clean) >= 730,
                'seasonality_mode': 'multiplicative' if growth == 'logistic' else 'additive'
            }
            
            model = Prophet(**prophet_params)
            
            # Add custom seasonality if we have enough data points
            if len(ts_clean) >= 4 * self.seasonal_periods:
                model.add_seasonality(
                    name='custom_seasonal',
                    period=self.seasonal_periods,
                    fourier_order=2
                )
                logger.info(f"Added custom seasonality with period {self.seasonal_periods}")
            
            # Fit the model
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(df_prophet)
            
            # Generate future dataframe
            future = model.make_future_dataframe(periods=steps_ahead, freq='D')
            
            # Make predictions
            forecast_df = model.predict(future)
            
            # Extract forecast results
            forecast_values = forecast_df['yhat'].tail(steps_ahead).values
            forecast_dates = forecast_df['ds'].tail(steps_ahead)
            lower_bound = forecast_df['yhat_lower'].tail(steps_ahead).values
            upper_bound = forecast_df['yhat_upper'].tail(steps_ahead).values
            
            # Calculate fit metrics
            fitted_values = forecast_df['yhat'].iloc[:-steps_ahead].values
            residuals = df_prophet['y'].values - fitted_values
            
            fit_metrics = {
                'rmse': np.sqrt(mean_squared_error(df_prophet['y'], fitted_values)),
                'mae': mean_absolute_error(df_prophet['y'], fitted_values),
                'mape': np.mean(np.abs((df_prophet['y'] - fitted_values) / df_prophet['y'])) * 100
            }
            
            # Residual diagnostics
            residual_diagnostics = self._calculate_residual_diagnostics(pd.Series(residuals))
            
            # Model parameters
            model_params = {
                'growth': growth,
                'seasonalities': list(model.seasonalities.keys()),
                'n_training_points': len(ts_clean),
                'prophet_params': prophet_params
            }
            
            logger.info(f"Prophet forecasting completed: RMSE={fit_metrics['rmse']:.3f}")
            
            return ForecastResult(
                forecast_values=forecast_values,
                forecast_dates=forecast_dates,
                confidence_intervals=(lower_bound, upper_bound),
                model_name="Prophet",
                model_params=model_params,
                fit_metrics=fit_metrics,
                residual_diagnostics=residual_diagnostics
            )
            
        except Exception as e:
            logger.error(f"Error in Prophet forecasting: {e}")
            return None

    def _calculate_residual_diagnostics(self, residuals: pd.Series) -> Dict[str, Any]:
        """Calculate comprehensive residual diagnostics."""
        try:
            diagnostics = {}
            
            # Basic statistics
            diagnostics['mean_residual'] = float(residuals.mean())
            diagnostics['std_residual'] = float(residuals.std())
            diagnostics['skewness'] = float(residuals.skew())
            diagnostics['kurtosis'] = float(residuals.kurtosis())
            
            # Normality test (Jarque-Bera)
            from scipy.stats import jarque_bera
            jb_stat, jb_pvalue = jarque_bera(residuals.dropna())
            diagnostics['jarque_bera'] = {'statistic': float(jb_stat), 'p_value': float(jb_pvalue)}
            
            # Autocorrelation test
            if HAS_STATSMODELS and len(residuals) > 10:
                ljung_box = acorr_ljungbox(residuals.dropna(), lags=min(10, len(residuals)//4), return_df=True)
                diagnostics['ljung_box_p_value'] = float(ljung_box['lb_pvalue'].iloc[-1])
            
            return diagnostics
            
        except Exception as e:
            logger.warning(f"Error calculating residual diagnostics: {e}")
            return {'error': str(e)} 

    def analyze_trends(self,
                       ts_data: pd.Series,
                       decomposition_model: str = 'additive') -> Dict[str, Any]:
        """Analyze trend, seasonal, and residual components in a time series.
        Returns keys: trend_component, seasonal_component, residual_component, trend_statistics.
        """
        try:
            values = ts_data.values.astype(float)
            n = len(values)
            if n < 8:
                return {
                    'trend_component': values.tolist(),
                    'seasonal_component': [],
                    'residual_component': (values - np.mean(values)).tolist(),
                    'trend_statistics': {
                        'trend_direction': 'stable',
                        'trend_strength': 0.0,
                        'seasonality_strength': 0.0
                    }
                }
            # Simple moving average trend
            window = max(3, n // 10)
            trend = pd.Series(values).rolling(window=window, center=True, min_periods=1).mean().values
            seasonal = np.zeros_like(values)
            residual = values - trend
            # Stats
            slope = np.polyfit(np.arange(n), values, 1)[0]
            trend_strength = float(np.var(trend) / (np.var(values) + 1e-12))
            seasonality_strength = 0.0
            trend_direction = 'increasing' if slope > 0 else ('decreasing' if slope < 0 else 'stable')
            return {
                'trend_component': trend.tolist(),
                'seasonal_component': seasonal.tolist(),
                'residual_component': residual.tolist(),
                'trend_statistics': {
                    'trend_direction': trend_direction,
                    'trend_strength': trend_strength,
                    'seasonality_strength': seasonality_strength
                }
            }
        except Exception as e:
            logger.error(f"Error in analyze_trends: {e}")
            return {'error': str(e)}

    def validate_forecast_accuracy(self,
                                   train_series: pd.Series,
                                   test_series: pd.Series,
                                   model_type: str = 'arima',
                                   model_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Train specified model on train_series and evaluate on test_series.
        Returns keys: accuracy_metrics, forecast_vs_actual, residual_analysis.
        """
        try:
            model_params = model_params or {}
            if model_type == 'arima':
                order = model_params.get('order', (1, 1, 1))
                result = self.fit_arima_model(train_series, order=order, auto_select=False)
            elif model_type == 'prophet':
                result = self.fit_prophet_model(train_series)
            else:
                result = self._fit_exponential_smoothing(train_series)
            # Forecast horizon is length of test
            h = len(test_series)
            forecast = result.forecast_values[:h] if len(result.forecast_values) >= h else np.resize(result.forecast_values, h)
            # Basic metrics
            mae = float(np.mean(np.abs(test_series.values - forecast)))
            rmse = float(np.sqrt(np.mean((test_series.values - forecast) ** 2)))
            mape = float(np.mean(np.abs((test_series.values - forecast) / (np.where(test_series.values == 0, 1, test_series.values)))) * 100)
            return {
                'accuracy_metrics': {'mae': mae, 'rmse': rmse, 'mape': mape},
                'forecast_vs_actual': {'forecast': forecast.tolist(), 'actual': test_series.values.tolist()},
                'residual_analysis': self._compute_residual_diagnostics(pd.Series(test_series.values - forecast))
            }
        except Exception as e:
            logger.error(f"Error in validate_forecast_accuracy: {e}")
            return {'error': str(e)}

    def fit_mixed_effects_forecast(self,
                                   data_by_subject: Dict[str, pd.Series],
                                   forecast_steps: int = 5,
                                   include_subject_effects: bool = True) -> Dict[str, Any]:
        """Simplified mixed effects style forecasting by blending subject-wise forecasts.
        Returns keys: subject_forecasts (dict), population_forecast (list), subject_effects (dict).
        """
        try:
            subject_forecasts = {}
            subject_effects = {}
            all_forecasts = []
            for subject, series in data_by_subject.items():
                res = self.fit_arima_model(series, order=(1, 1, 1), auto_select=False)
                fc = res.forecast_values[:forecast_steps]
                if len(fc) < forecast_steps:
                    fc = np.resize(fc, forecast_steps)
                subject_forecasts[subject] = fc.tolist()
                subject_effects[subject] = float(np.mean(series.values) - np.mean(list(data_by_subject.values())[0].values)) if include_subject_effects else 0.0
                all_forecasts.append(fc)
            population_forecast = np.mean(np.vstack(all_forecasts), axis=0).tolist() if all_forecasts else []
            return {
                'subject_forecasts': subject_forecasts,
                'population_forecast': population_forecast,
                'subject_effects': subject_effects
            }
        except Exception as e:
            logger.error(f"Error in fit_mixed_effects_forecast: {e}")
            return {'error': str(e)} 