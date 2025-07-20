"""
Comprehensive tests for ML Analysis functionality.

This module tests the machine learning features including
clustering for autonomic phenotypes and time series forecasting.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock
import warnings

import sys
import os
sys.path.append(str(Path(__file__).parent.parent))

try:
    from sklearn.cluster import KMeans
    from sklearn.datasets import make_blobs
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from ml_analysis.clustering import AutonomicPhenotypeClustering
from ml_analysis.forecasting import HRVForecasting

@pytest.mark.skipif(not HAS_SKLEARN, reason="Scikit-learn not available")
class TestAutonomicPhenotypeClustering:
    """Test cases for autonomic phenotype clustering."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.clustering = AutonomicPhenotypeClustering()
        
        # Create sample HRV feature data
        np.random.seed(42)
        n_samples = 100
        self.sample_features = pd.DataFrame({
            'sdnn': np.random.normal(50, 15, n_samples),
            'rmssd': np.random.normal(30, 10, n_samples),
            'pnn50': np.random.normal(15, 8, n_samples),
            'lf_power': np.random.lognormal(6, 0.5, n_samples),
            'hf_power': np.random.lognormal(5.5, 0.6, n_samples),
            'lf_hf_ratio': np.random.gamma(2, 0.8, n_samples),
            'sd1': np.random.normal(22, 7, n_samples),
            'sd2': np.random.normal(68, 20, n_samples)
        })
        
        # Add subject identifiers
        self.sample_features['subject'] = [f'S{i//10 + 1:02d}' for i in range(n_samples)]
        self.sample_features['sol'] = [f'Sol{i%16 + 1:02d}' for i in range(n_samples)]
        
    def test_kmeans_clustering_basic(self):
        """Test basic K-means clustering functionality."""
        features_subset = self.sample_features[['sdnn', 'rmssd', 'lf_hf_ratio']].copy()
        
        result = self.clustering.perform_kmeans_clustering(
            features_subset,
            n_clusters=3,
            feature_columns=['sdnn', 'rmssd', 'lf_hf_ratio']
        )
        
        assert isinstance(result, dict)
        assert 'cluster_labels' in result
        assert 'cluster_centers' in result
        assert 'inertia' in result
        assert 'silhouette_score' in result
        assert 'interpretation' in result
        
        # Check cluster labels
        labels = result['cluster_labels']
        assert len(labels) == len(features_subset)
        assert set(labels).issubset({0, 1, 2})  # Should have 3 clusters
        
        # Check cluster centers
        centers = result['cluster_centers']
        assert centers.shape == (3, 3)  # 3 clusters, 3 features
        
        # Check silhouette score range
        assert -1 <= result['silhouette_score'] <= 1
        
    def test_hierarchical_clustering(self):
        """Test hierarchical clustering functionality."""
        features_subset = self.sample_features[['sdnn', 'rmssd', 'lf_hf_ratio']].copy()
        
        result = self.clustering.perform_hierarchical_clustering(
            features_subset,
            n_clusters=3,
            feature_columns=['sdnn', 'rmssd', 'lf_hf_ratio'],
            linkage_method='ward'
        )
        
        assert isinstance(result, dict)
        assert 'cluster_labels' in result
        assert 'dendrogram_data' in result
        assert 'silhouette_score' in result
        assert 'interpretation' in result
        
        labels = result['cluster_labels']
        assert len(labels) == len(features_subset)
        assert len(set(labels)) <= 3  # Should have at most 3 clusters
        
    def test_optimal_clusters_detection(self):
        """Test optimal number of clusters detection."""
        features_subset = self.sample_features[['sdnn', 'rmssd', 'lf_hf_ratio']].copy()
        
        result = self.clustering.find_optimal_clusters(
            features_subset,
            feature_columns=['sdnn', 'rmssd', 'lf_hf_ratio'],
            max_clusters=8,
            methods=['elbow', 'silhouette']
        )
        
        assert isinstance(result, dict)
        assert 'elbow_scores' in result
        assert 'silhouette_scores' in result
        assert 'optimal_k_elbow' in result
        assert 'optimal_k_silhouette' in result
        assert 'recommendations' in result
        
        # Check that optimal k values are reasonable
        assert 2 <= result['optimal_k_elbow'] <= 8
        assert 2 <= result['optimal_k_silhouette'] <= 8
        
    def test_phenotype_interpretation(self):
        """Test autonomic phenotype interpretation."""
        # Create mock cluster centers with distinct characteristics
        cluster_centers = np.array([
            [30, 20, 3.0],  # High sympathetic (high LF/HF)
            [60, 45, 0.8],  # High parasympathetic (low LF/HF, high RMSSD)
            [45, 30, 1.5]   # Balanced
        ])
        
        feature_names = ['sdnn', 'rmssd', 'lf_hf_ratio']
        
        interpretation = self.clustering.interpret_autonomic_phenotypes(
            cluster_centers, 
            feature_names
        )
        
        assert isinstance(interpretation, dict)
        assert len(interpretation) == 3  # 3 clusters
        
        for cluster_id, desc in interpretation.items():
            assert isinstance(desc, dict)
            assert 'phenotype' in desc
            assert 'characteristics' in desc
            assert 'description' in desc
            
    def test_clustering_validation(self):
        """Test clustering validation metrics."""
        features_subset = self.sample_features[['sdnn', 'rmssd', 'lf_hf_ratio']].copy()
        
        # Perform clustering first
        clustering_result = self.clustering.perform_kmeans_clustering(
            features_subset,
            n_clusters=3,
            feature_columns=['sdnn', 'rmssd', 'lf_hf_ratio']
        )
        
        labels = clustering_result['cluster_labels']
        
        validation_result = self.clustering.validate_clustering(
            features_subset[['sdnn', 'rmssd', 'lf_hf_ratio']],
            labels
        )
        
        assert isinstance(validation_result, dict)
        assert 'silhouette_score' in validation_result
        assert 'calinski_harabasz_score' in validation_result
        assert 'davies_bouldin_score' in validation_result
        assert 'cluster_stability' in validation_result
        
        # Check score ranges
        assert -1 <= validation_result['silhouette_score'] <= 1
        assert validation_result['calinski_harabasz_score'] >= 0
        assert validation_result['davies_bouldin_score'] >= 0
        
    def test_feature_importance_analysis(self):
        """Test feature importance analysis for clustering."""
        features_subset = self.sample_features[['sdnn', 'rmssd', 'lf_hf_ratio']].copy()
        
        importance_result = self.clustering.analyze_feature_importance(
            features_subset,
            feature_columns=['sdnn', 'rmssd', 'lf_hf_ratio'],
            n_clusters=3
        )
        
        assert isinstance(importance_result, dict)
        assert 'feature_importance' in importance_result
        assert 'cluster_separation_power' in importance_result
        
        importance_scores = importance_result['feature_importance']
        assert len(importance_scores) == 3  # 3 features
        assert all(score >= 0 for score in importance_scores.values())
        
    def test_clustering_error_handling(self):
        """Test error handling in clustering methods."""
        # Test with insufficient data
        small_features = self.sample_features.iloc[:5].copy()
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = self.clustering.perform_kmeans_clustering(
                small_features,
                n_clusters=3,  # More clusters than data points
                feature_columns=['sdnn', 'rmssd', 'lf_hf_ratio']
            )
            
        # Should handle gracefully
        assert isinstance(result, dict)
        
        # Test with missing features
        incomplete_features = self.sample_features[['sdnn']].copy()
        
        result = self.clustering.perform_kmeans_clustering(
            incomplete_features,
            n_clusters=2,
            feature_columns=['sdnn', 'missing_feature']  # One feature missing
        )
        
        # Should handle missing features
        assert isinstance(result, dict)

class TestHRVForecasting:
    """Test cases for HRV forecasting functionality."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.forecasting = HRVForecasting()
        
        # Create sample time series data
        np.random.seed(42)
        n_timepoints = 100
        
        # Simulate HRV time series with trend and seasonality
        time_index = pd.date_range('2024-01-01', periods=n_timepoints, freq='D')
        
        # Base trend with noise
        trend = np.linspace(45, 55, n_timepoints)
        seasonal = 5 * np.sin(2 * np.pi * np.arange(n_timepoints) / 14)  # Weekly pattern
        noise = np.random.normal(0, 2, n_timepoints)
        
        self.sample_hrv_series = pd.Series(
            trend + seasonal + noise,
            index=time_index,
            name='sdnn'
        )
        
        # Multi-subject data
        subjects = ['S01', 'S02', 'S03']
        self.multi_subject_data = {}
        
        for subject in subjects:
            subject_trend = trend + np.random.normal(0, 5, n_timepoints)
            subject_series = pd.Series(
                subject_trend + seasonal + np.random.normal(0, 2, n_timepoints),
                index=time_index,
                name='sdnn'
            )
            self.multi_subject_data[subject] = subject_series
            
    def test_arima_forecasting_basic(self):
        """Test basic ARIMA forecasting functionality."""
        result = self.forecasting.fit_arima_model(
            self.sample_hrv_series,
            order=(1, 1, 1),
            forecast_steps=10
        )
        
        assert isinstance(result, dict)
        assert 'model_summary' in result
        assert 'forecast_values' in result
        assert 'confidence_intervals' in result
        assert 'model_diagnostics' in result
        
        # Check forecast values
        forecast = result['forecast_values']
        assert len(forecast) == 10
        assert all(isinstance(val, (int, float)) for val in forecast)
        
        # Check confidence intervals
        ci = result['confidence_intervals']
        assert ci.shape == (10, 2)  # 10 forecasts, lower and upper bounds
        assert np.all(ci[:, 0] <= ci[:, 1])  # Lower <= Upper bounds
        
    def test_prophet_forecasting(self):
        """Test Prophet forecasting functionality."""
        # Create prophet-compatible dataframe
        prophet_df = pd.DataFrame({
            'ds': self.sample_hrv_series.index,
            'y': self.sample_hrv_series.values
        })
        
        result = self.forecasting.fit_prophet_model(
            prophet_df,
            forecast_days=14,
            include_weekly_seasonality=True,
            include_yearly_seasonality=False
        )
        
        assert isinstance(result, dict)
        assert 'model_summary' in result
        assert 'forecast_dataframe' in result
        assert 'components' in result
        assert 'performance_metrics' in result
        
        # Check forecast dataframe
        forecast_df = result['forecast_dataframe']
        assert len(forecast_df) >= 14  # At least forecast days
        assert 'yhat' in forecast_df.columns  # Forecast values
        assert 'yhat_lower' in forecast_df.columns  # Lower bounds
        assert 'yhat_upper' in forecast_df.columns  # Upper bounds
        
    def test_model_comparison(self):
        """Test model comparison functionality."""
        models_to_compare = ['arima', 'prophet', 'linear_trend']
        
        comparison_result = self.forecasting.compare_models(
            self.sample_hrv_series,
            models=models_to_compare,
            test_size=0.2,
            forecast_horizon=7
        )
        
        assert isinstance(comparison_result, dict)
        assert 'model_performances' in comparison_result
        assert 'best_model' in comparison_result
        assert 'recommendations' in comparison_result
        
        # Check performance metrics for each model
        performances = comparison_result['model_performances']
        for model_name in models_to_compare:
            if model_name in performances:
                assert 'mae' in performances[model_name]
                assert 'rmse' in performances[model_name]
                assert 'mape' in performances[model_name]
                
        # Best model should be one of the compared models
        assert comparison_result['best_model'] in models_to_compare
        
    def test_trend_analysis(self):
        """Test trend analysis and decomposition."""
        trend_result = self.forecasting.analyze_trends(
            self.sample_hrv_series,
            decomposition_model='additive'
        )
        
        assert isinstance(trend_result, dict)
        assert 'trend_component' in trend_result
        assert 'seasonal_component' in trend_result
        assert 'residual_component' in trend_result
        assert 'trend_statistics' in trend_result
        
        # Check component lengths
        original_length = len(self.sample_hrv_series)
        assert len(trend_result['trend_component']) <= original_length
        assert len(trend_result['seasonal_component']) <= original_length
        assert len(trend_result['residual_component']) <= original_length
        
        # Check trend statistics
        trend_stats = trend_result['trend_statistics']
        assert 'trend_direction' in trend_stats
        assert 'trend_strength' in trend_stats
        assert 'seasonality_strength' in trend_stats
        
    def test_sol_adaptation_prediction(self):
        """Test SOL adaptation prediction functionality."""
        # Simulate adaptation pattern
        sol_days = np.arange(1, 17)  # 16 SOL days
        
        # Create adaptation pattern (initial decrease, then recovery)
        adaptation_pattern = 50 - 10 * np.exp(-sol_days/5) + np.random.normal(0, 2, len(sol_days))
        
        sol_series = pd.Series(adaptation_pattern, index=sol_days, name='adaptation_metric')
        
        prediction_result = self.forecasting.predict_sol_adaptation(
            sol_series,
            target_sols=[17, 18, 19, 20],
            adaptation_model='exponential'
        )
        
        assert isinstance(prediction_result, dict)
        assert 'predicted_values' in prediction_result
        assert 'adaptation_curve_params' in prediction_result
        assert 'recovery_timeline' in prediction_result
        assert 'confidence_bounds' in prediction_result
        
        # Check predictions
        predictions = prediction_result['predicted_values']
        assert len(predictions) == 4  # 4 target SOLs
        
        # Check adaptation parameters
        params = prediction_result['adaptation_curve_params']
        assert 'baseline' in params
        assert 'amplitude' in params
        assert 'time_constant' in params
        
    def test_multi_subject_forecasting(self):
        """Test multi-subject forecasting with mixed effects."""
        mixed_results = self.forecasting.fit_mixed_effects_forecast(
            self.multi_subject_data,
            forecast_steps=5,
            include_subject_effects=True
        )
        
        assert isinstance(mixed_results, dict)
        assert 'subject_forecasts' in mixed_results
        assert 'population_forecast' in mixed_results
        assert 'subject_effects' in mixed_results
        
        # Check individual subject forecasts
        subject_forecasts = mixed_results['subject_forecasts']
        for subject in self.multi_subject_data.keys():
            if subject in subject_forecasts:
                assert len(subject_forecasts[subject]) == 5
                
        # Check population forecast
        pop_forecast = mixed_results['population_forecast']
        assert len(pop_forecast) == 5
        
    def test_forecasting_error_handling(self):
        """Test error handling in forecasting methods."""
        # Test with very short time series
        short_series = self.sample_hrv_series.iloc[:5]
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = self.forecasting.fit_arima_model(
                short_series,
                order=(2, 1, 2),  # Complex model for short series
                forecast_steps=3
            )
            
        # Should handle gracefully
        assert isinstance(result, dict)
        
        # Test with constant series (no variation)
        constant_series = pd.Series(
            np.ones(50), 
            index=pd.date_range('2024-01-01', periods=50, freq='D'),
            name='constant'
        )
        
        result = self.forecasting.fit_arima_model(
            constant_series,
            order=(1, 0, 1),
            forecast_steps=5
        )
        
        # Should handle constant series
        assert isinstance(result, dict)
        
    def test_forecast_validation(self):
        """Test forecast validation and accuracy metrics."""
        # Split data for validation
        split_point = int(0.8 * len(self.sample_hrv_series))
        train_series = self.sample_hrv_series.iloc[:split_point]
        test_series = self.sample_hrv_series.iloc[split_point:]
        
        validation_result = self.forecasting.validate_forecast_accuracy(
            train_series,
            test_series,
            model_type='arima',
            model_params={'order': (1, 1, 1)}
        )
        
        assert isinstance(validation_result, dict)
        assert 'accuracy_metrics' in validation_result
        assert 'forecast_vs_actual' in validation_result
        assert 'residual_analysis' in validation_result
        
        # Check accuracy metrics
        metrics = validation_result['accuracy_metrics']
        assert 'mae' in metrics
        assert 'rmse' in metrics
        assert 'mape' in metrics
        assert all(metric >= 0 for metric in metrics.values())
        
        # Check forecast vs actual comparison
        comparison = validation_result['forecast_vs_actual']
        assert len(comparison['forecast']) == len(test_series)
        assert len(comparison['actual']) == len(test_series)

class TestMLAnalysisIntegration:
    """Integration tests for ML analysis components."""
    
    def setup_method(self):
        """Setup for each test method."""
        if not HAS_SKLEARN:
            pytest.skip("Scikit-learn not available")
            
        self.clustering = AutonomicPhenotypeClustering()
        self.forecasting = HRVForecasting()
        
    def test_clustering_to_forecasting_pipeline(self):
        """Test pipeline from clustering to personalized forecasting."""
        # Step 1: Generate sample HRV data for multiple subjects
        np.random.seed(42)
        n_subjects = 20
        n_timepoints = 50
        
        subjects_data = {}
        features_list = []
        
        for i in range(n_subjects):
            subject_id = f'S{i+1:02d}'
            
            # Generate subject-specific HRV time series
            base_hrv = np.random.normal(50, 10)  # Individual baseline
            time_series = base_hrv + np.random.normal(0, 5, n_timepoints)
            dates = pd.date_range('2024-01-01', periods=n_timepoints, freq='D')
            
            subjects_data[subject_id] = pd.Series(time_series, index=dates)
            
            # Extract features for clustering
            features = {
                'subject': subject_id,
                'mean_hrv': np.mean(time_series),
                'std_hrv': np.std(time_series),
                'trend': np.polyfit(range(n_timepoints), time_series, 1)[0],
                'cv': np.std(time_series) / np.mean(time_series)
            }
            features_list.append(features)
        
        features_df = pd.DataFrame(features_list)
        
        # Step 2: Perform clustering
        clustering_result = self.clustering.perform_kmeans_clustering(
            features_df,
            n_clusters=3,
            feature_columns=['mean_hrv', 'std_hrv', 'trend', 'cv']
        )
        
        # Step 3: Group subjects by cluster
        features_df['cluster'] = clustering_result['cluster_labels']
        clustered_subjects = features_df.groupby('cluster')['subject'].apply(list).to_dict()
        
        # Step 4: Perform cluster-specific forecasting
        cluster_forecasts = {}
        
        for cluster_id, subject_list in clustered_subjects.items():
            if len(subject_list) >= 2:  # Need multiple subjects for mixed effects
                cluster_data = {subj: subjects_data[subj] for subj in subject_list[:3]}
                
                forecast_result = self.forecasting.fit_mixed_effects_forecast(
                    cluster_data,
                    forecast_steps=7
                )
                
                cluster_forecasts[cluster_id] = forecast_result
        
        # Verify results
        assert len(cluster_forecasts) >= 1  # Should have at least one cluster with forecasts
        
        for cluster_id, forecast_result in cluster_forecasts.items():
            assert 'subject_forecasts' in forecast_result
            assert 'population_forecast' in forecast_result
            assert len(forecast_result['population_forecast']) == 7
            
    def test_phenotype_based_personalization(self):
        """Test personalized analysis based on autonomic phenotypes."""
        # Create synthetic data with known phenotype patterns
        np.random.seed(42)
        
        # Sympathetic-dominant phenotype
        sympathetic_features = pd.DataFrame({
            'lf_hf_ratio': np.random.normal(3.0, 0.5, 30),
            'sdnn': np.random.normal(35, 8, 30),
            'rmssd': np.random.normal(20, 5, 30),
            'subject': [f'SYM{i:02d}' for i in range(30)]
        })
        
        # Parasympathetic-dominant phenotype  
        parasympathetic_features = pd.DataFrame({
            'lf_hf_ratio': np.random.normal(0.8, 0.3, 30),
            'sdnn': np.random.normal(55, 10, 30),
            'rmssd': np.random.normal(45, 8, 30),
            'subject': [f'PAR{i:02d}' for i in range(30)]
        })
        
        # Balanced phenotype
        balanced_features = pd.DataFrame({
            'lf_hf_ratio': np.random.normal(1.5, 0.4, 30),
            'sdnn': np.random.normal(45, 8, 30),
            'rmssd': np.random.normal(32, 6, 30),
            'subject': [f'BAL{i:02d}' for i in range(30)]
        })
        
        # Combine all phenotypes
        all_features = pd.concat([sympathetic_features, parasympathetic_features, balanced_features])
        
        # Perform clustering
        clustering_result = self.clustering.perform_kmeans_clustering(
            all_features,
            n_clusters=3,
            feature_columns=['lf_hf_ratio', 'sdnn', 'rmssd']
        )
        
        # Interpret phenotypes
        phenotype_interpretation = self.clustering.interpret_autonomic_phenotypes(
            clustering_result['cluster_centers'],
            ['lf_hf_ratio', 'sdnn', 'rmssd']
        )
        
        # Verify phenotype identification
        assert len(phenotype_interpretation) == 3
        
        for cluster_id, interpretation in phenotype_interpretation.items():
            assert 'phenotype' in interpretation
            assert 'characteristics' in interpretation
            assert interpretation['phenotype'] in ['sympathetic', 'parasympathetic', 'balanced']
            
        # Create personalized forecasting based on phenotypes
        all_features['cluster'] = clustering_result['cluster_labels']
        
        # Generate time series for each phenotype
        phenotype_time_series = {}
        
        for cluster_id in range(3):
            cluster_subjects = all_features[all_features['cluster'] == cluster_id]['subject'].tolist()
            
            # Generate characteristic time series for this phenotype
            if cluster_id == 0:  # Assume sympathetic
                base_pattern = np.random.normal(40, 12, 50)  # More variable, lower baseline
            elif cluster_id == 1:  # Assume parasympathetic  
                base_pattern = np.random.normal(60, 8, 50)   # Less variable, higher baseline
            else:  # Balanced
                base_pattern = np.random.normal(50, 10, 50)  # Moderate variability and baseline
                
            dates = pd.date_range('2024-01-01', periods=50, freq='D')
            phenotype_time_series[cluster_id] = pd.Series(base_pattern, index=dates)
        
        # Forecast for each phenotype
        phenotype_forecasts = {}
        
        for cluster_id, time_series in phenotype_time_series.items():
            forecast_result = self.forecasting.fit_arima_model(
                time_series,
                order=(1, 1, 1),
                forecast_steps=10
            )
            phenotype_forecasts[cluster_id] = forecast_result
        
        # Verify personalized forecasts
        assert len(phenotype_forecasts) == 3
        
        for cluster_id, forecast in phenotype_forecasts.items():
            assert 'forecast_values' in forecast
            assert len(forecast['forecast_values']) == 10

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 