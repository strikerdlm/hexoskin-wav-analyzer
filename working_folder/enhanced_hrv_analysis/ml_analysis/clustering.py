"""
HRV Clustering Analysis for Autonomic Phenotype Identification

This module provides clustering capabilities for identifying distinct
autonomic nervous system phenotypes based on HRV metrics using:
- K-means clustering with optimal cluster number selection
- Hierarchical clustering with dendrogram visualization  
- HDBSCAN for density-based clustering
- Cluster validation and interpretation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.model_selection import cross_val_score
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
from typing import Dict, Any, Optional, List, Tuple
import logging
from dataclasses import dataclass

# Optional advanced clustering imports
try:
    import hdbscan
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False
    
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

logger = logging.getLogger(__name__)

@dataclass
class ClusterResult:
    """Container for clustering analysis results."""
    cluster_labels: np.ndarray
    cluster_centers: np.ndarray
    n_clusters: int
    silhouette_score: float
    calinski_harabasz_score: float
    davies_bouldin_score: float
    inertia: Optional[float]
    cluster_sizes: Dict[int, int]
    cluster_profiles: pd.DataFrame

@dataclass 
class ClusterValidation:
    """Container for cluster validation metrics."""
    silhouette_scores: List[float]
    calinski_harabasz_scores: List[float]
    davies_bouldin_scores: List[float]
    elbow_scores: List[float]
    optimal_k: int
    validation_method: str

class HRVClustering:
    """Advanced clustering analysis for HRV-based autonomic phenotypes."""
    
    def __init__(self, 
                 standardize: bool = True,
                 random_state: int = 42):
        """
        Initialize HRV clustering analyzer.
        
        Args:
            standardize: Whether to standardize features before clustering
            random_state: Random state for reproducibility
        """
        self.standardize = standardize
        self.random_state = random_state
        self.scaler = StandardScaler() if standardize else None
        self.cluster_model = None
        self.cluster_results = None
        
    def find_optimal_clusters(self,
                            hrv_data: pd.DataFrame,
                            max_k: int = 10,
                            method: str = 'kmeans',
                            validation_metrics: List[str] = None) -> ClusterValidation:
        """
        Find optimal number of clusters using multiple validation metrics.
        
        Args:
            hrv_data: DataFrame with HRV metrics
            max_k: Maximum number of clusters to test
            method: Clustering method ('kmeans', 'hierarchical')
            validation_metrics: List of validation metrics to use
            
        Returns:
            Cluster validation results
        """
        try:
            if validation_metrics is None:
                validation_metrics = ['silhouette', 'calinski_harabasz', 'davies_bouldin', 'elbow']
                
            # Prepare data
            X = self._prepare_data(hrv_data)
            if X is None:
                return ClusterValidation([], [], [], [], 2, "error")
                
            # Test different numbers of clusters
            k_range = range(2, min(max_k + 1, len(X)))
            silhouette_scores = []
            calinski_harabasz_scores = []
            davies_bouldin_scores = []
            elbow_scores = []
            
            for k in k_range:
                try:
                    if method == 'kmeans':
                        model = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
                        labels = model.fit_predict(X)
                        centers = model.cluster_centers_
                        elbow_scores.append(model.inertia_)
                    elif method == 'hierarchical':
                        model = AgglomerativeClustering(n_clusters=k)
                        labels = model.fit_predict(X)
                        centers = self._compute_cluster_centers(X, labels)
                        elbow_scores.append(self._compute_within_cluster_sum_squares(X, labels, centers))
                    else:
                        continue
                        
                    # Compute validation metrics
                    if 'silhouette' in validation_metrics and len(np.unique(labels)) > 1:
                        sil_score = silhouette_score(X, labels)
                        silhouette_scores.append(sil_score)
                    else:
                        silhouette_scores.append(0)
                        
                    if 'calinski_harabasz' in validation_metrics and len(np.unique(labels)) > 1:
                        ch_score = calinski_harabasz_score(X, labels)
                        calinski_harabasz_scores.append(ch_score)
                    else:
                        calinski_harabasz_scores.append(0)
                        
                    if 'davies_bouldin' in validation_metrics and len(np.unique(labels)) > 1:
                        db_score = davies_bouldin_score(X, labels)
                        davies_bouldin_scores.append(db_score)
                    else:
                        davies_bouldin_scores.append(float('inf'))
                        
                except Exception as e:
                    logger.warning(f"Error testing k={k}: {e}")
                    silhouette_scores.append(0)
                    calinski_harabasz_scores.append(0)
                    davies_bouldin_scores.append(float('inf'))
                    elbow_scores.append(float('inf'))
                    
            # Determine optimal k using multiple criteria
            optimal_k = self._determine_optimal_k(
                silhouette_scores, calinski_harabasz_scores, 
                davies_bouldin_scores, elbow_scores, k_range
            )
            
            return ClusterValidation(
                silhouette_scores=silhouette_scores,
                calinski_harabasz_scores=calinski_harabasz_scores,
                davies_bouldin_scores=davies_bouldin_scores,
                elbow_scores=elbow_scores,
                optimal_k=optimal_k,
                validation_method=method
            )
            
        except Exception as e:
            logger.error(f"Error in optimal cluster finding: {e}")
            return ClusterValidation([], [], [], [], 2, "error")
            
    def perform_kmeans_clustering(self,
                                hrv_data: pd.DataFrame,
                                n_clusters: int = None,
                                feature_selection: List[str] = None) -> ClusterResult:
        """
        Perform K-means clustering on HRV metrics.
        
        Args:
            hrv_data: DataFrame with HRV metrics
            n_clusters: Number of clusters (if None, will be determined automatically)
            feature_selection: List of specific features to use for clustering
            
        Returns:
            Clustering results
        """
        try:
            # Feature selection
            if feature_selection is not None:
                available_features = [f for f in feature_selection if f in hrv_data.columns]
                if not available_features:
                    logger.error("None of the specified features found in data")
                    return self._create_empty_cluster_result()
                clustering_data = hrv_data[available_features]
            else:
                clustering_data = hrv_data.select_dtypes(include=[np.number])
                
            # Prepare data
            X = self._prepare_data(clustering_data)
            if X is None:
                return self._create_empty_cluster_result()
                
            # Determine number of clusters if not specified
            if n_clusters is None:
                validation_results = self.find_optimal_clusters(clustering_data)
                n_clusters = validation_results.optimal_k
                logger.info(f"Automatically selected {n_clusters} clusters")
                
            # Perform K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
            cluster_labels = kmeans.fit_predict(X)
            cluster_centers = kmeans.cluster_centers_
            
            # Store model
            self.cluster_model = kmeans
            
            # Compute validation metrics
            sil_score = silhouette_score(X, cluster_labels) if len(np.unique(cluster_labels)) > 1 else 0
            ch_score = calinski_harabasz_score(X, cluster_labels) if len(np.unique(cluster_labels)) > 1 else 0
            db_score = davies_bouldin_score(X, cluster_labels) if len(np.unique(cluster_labels)) > 1 else float('inf')
            
            # Compute cluster sizes
            unique_labels, counts = np.unique(cluster_labels, return_counts=True)
            cluster_sizes = dict(zip(unique_labels, counts))
            
            # Compute cluster profiles
            cluster_profiles = self._compute_cluster_profiles(clustering_data, cluster_labels)
            
            # Transform centers back to original scale if standardized
            if self.scaler is not None:
                cluster_centers_original = self.scaler.inverse_transform(cluster_centers)
            else:
                cluster_centers_original = cluster_centers
                
            result = ClusterResult(
                cluster_labels=cluster_labels,
                cluster_centers=cluster_centers_original,
                n_clusters=n_clusters,
                silhouette_score=sil_score,
                calinski_harabasz_score=ch_score,
                davies_bouldin_score=db_score,
                inertia=kmeans.inertia_,
                cluster_sizes=cluster_sizes,
                cluster_profiles=cluster_profiles
            )
            
            self.cluster_results = result
            return result
            
        except Exception as e:
            logger.error(f"Error in K-means clustering: {e}")
            return self._create_empty_cluster_result()
            
    def perform_hierarchical_clustering(self,
                                      hrv_data: pd.DataFrame,
                                      n_clusters: int = None,
                                      linkage_method: str = 'ward',
                                      distance_metric: str = 'euclidean') -> ClusterResult:
        """
        Perform hierarchical clustering on HRV metrics.
        
        Args:
            hrv_data: DataFrame with HRV metrics  
            n_clusters: Number of clusters
            linkage_method: Linkage method ('ward', 'complete', 'average', 'single')
            distance_metric: Distance metric for linkage
            
        Returns:
            Clustering results
        """
        try:
            # Prepare data
            X = self._prepare_data(hrv_data)
            if X is None:
                return self._create_empty_cluster_result()
                
            # Determine number of clusters if not specified
            if n_clusters is None:
                validation_results = self.find_optimal_clusters(hrv_data, method='hierarchical')
                n_clusters = validation_results.optimal_k
                
            # Perform hierarchical clustering
            hierarchical = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage=linkage_method,
                metric=distance_metric if linkage_method != 'ward' else 'euclidean'
            )
            
            cluster_labels = hierarchical.fit_predict(X)
            
            # Compute cluster centers
            cluster_centers = self._compute_cluster_centers(X, cluster_labels)
            
            # Store model
            self.cluster_model = hierarchical
            
            # Compute validation metrics
            sil_score = silhouette_score(X, cluster_labels) if len(np.unique(cluster_labels)) > 1 else 0
            ch_score = calinski_harabasz_score(X, cluster_labels) if len(np.unique(cluster_labels)) > 1 else 0
            db_score = davies_bouldin_score(X, cluster_labels) if len(np.unique(cluster_labels)) > 1 else float('inf')
            
            # Compute cluster sizes
            unique_labels, counts = np.unique(cluster_labels, return_counts=True)
            cluster_sizes = dict(zip(unique_labels, counts))
            
            # Compute cluster profiles
            cluster_profiles = self._compute_cluster_profiles(hrv_data, cluster_labels)
            
            # Transform centers back to original scale if standardized
            if self.scaler is not None:
                cluster_centers_original = self.scaler.inverse_transform(cluster_centers)
            else:
                cluster_centers_original = cluster_centers
                
            result = ClusterResult(
                cluster_labels=cluster_labels,
                cluster_centers=cluster_centers_original,
                n_clusters=n_clusters,
                silhouette_score=sil_score,
                calinski_harabasz_score=ch_score,
                davies_bouldin_score=db_score,
                inertia=None,
                cluster_sizes=cluster_sizes,
                cluster_profiles=cluster_profiles
            )
            
            self.cluster_results = result
            return result
            
        except Exception as e:
            logger.error(f"Error in hierarchical clustering: {e}")
            return self._create_empty_cluster_result()
            
    def perform_hdbscan_clustering(self,
                                 hrv_data: pd.DataFrame,
                                 min_cluster_size: int = 5,
                                 min_samples: int = None) -> ClusterResult:
        """
        Perform HDBSCAN density-based clustering.
        
        Args:
            hrv_data: DataFrame with HRV metrics
            min_cluster_size: Minimum cluster size
            min_samples: Minimum samples in neighborhood
            
        Returns:
            Clustering results
        """
        if not HAS_HDBSCAN:
            logger.error("HDBSCAN library not available")
            return self._create_empty_cluster_result()
            
        try:
            # Prepare data
            X = self._prepare_data(hrv_data)
            if X is None:
                return self._create_empty_cluster_result()
                
            # Set default min_samples
            if min_samples is None:
                min_samples = min_cluster_size
                
            # Perform HDBSCAN clustering
            hdb = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric='euclidean'
            )
            
            cluster_labels = hdb.fit_predict(X)
            
            # Handle noise points (labeled as -1)
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            
            if n_clusters == 0:
                logger.warning("HDBSCAN found no clusters - all points classified as noise")
                return self._create_empty_cluster_result()
                
            # Compute cluster centers (excluding noise points)
            cluster_centers = self._compute_cluster_centers(X, cluster_labels, exclude_noise=True)
            
            # Store model
            self.cluster_model = hdb
            
            # Compute validation metrics (excluding noise points)
            valid_mask = cluster_labels != -1
            if np.sum(valid_mask) > 0 and len(np.unique(cluster_labels[valid_mask])) > 1:
                sil_score = silhouette_score(X[valid_mask], cluster_labels[valid_mask])
                ch_score = calinski_harabasz_score(X[valid_mask], cluster_labels[valid_mask])
                db_score = davies_bouldin_score(X[valid_mask], cluster_labels[valid_mask])
            else:
                sil_score = ch_score = 0
                db_score = float('inf')
                
            # Compute cluster sizes
            unique_labels, counts = np.unique(cluster_labels, return_counts=True)
            cluster_sizes = dict(zip(unique_labels, counts))
            
            # Compute cluster profiles
            cluster_profiles = self._compute_cluster_profiles(hrv_data, cluster_labels)
            
            # Transform centers back to original scale if standardized
            if self.scaler is not None and len(cluster_centers) > 0:
                cluster_centers_original = self.scaler.inverse_transform(cluster_centers)
            else:
                cluster_centers_original = cluster_centers
                
            result = ClusterResult(
                cluster_labels=cluster_labels,
                cluster_centers=cluster_centers_original,
                n_clusters=n_clusters,
                silhouette_score=sil_score,
                calinski_harabasz_score=ch_score,
                davies_bouldin_score=db_score,
                inertia=None,
                cluster_sizes=cluster_sizes,
                cluster_profiles=cluster_profiles
            )
            
            self.cluster_results = result
            return result
            
        except Exception as e:
            logger.error(f"Error in HDBSCAN clustering: {e}")
            return self._create_empty_cluster_result()
            
    def interpret_clusters(self,
                         cluster_result: ClusterResult,
                         feature_names: List[str] = None) -> Dict[str, Any]:
        """
        Interpret cluster characteristics and provide autonomic phenotype descriptions.
        
        Args:
            cluster_result: Clustering results
            feature_names: Names of features used for clustering
            
        Returns:
            Cluster interpretation results
        """
        try:
            interpretations = {}
            
            # Define autonomic phenotype characteristics
            autonomic_indicators = {
                'parasympathetic_dominance': ['rmssd', 'pnn50', 'hf_power', 'hf_nu', 'sd1'],
                'sympathetic_dominance': ['lf_nu', 'lf_hf_ratio', 'mean_hr', 'stress_index'],
                'balanced_autonomic': ['sdnn', 'total_power', 'vlf_power'],
                'low_variability': ['sdnn', 'rmssd', 'total_power']  # Lower values indicate reduced variability
            }
            
            for cluster_id in range(cluster_result.n_clusters):
                cluster_profile = cluster_result.cluster_profiles.iloc[cluster_id]
                cluster_size = cluster_result.cluster_sizes.get(cluster_id, 0)
                
                # Analyze cluster characteristics
                cluster_interpretation = {
                    'cluster_id': cluster_id,
                    'cluster_size': cluster_size,
                    'percentage_of_population': (cluster_size / len(cluster_result.cluster_labels)) * 100,
                    'dominant_characteristics': [],
                    'phenotype_classification': 'unclassified',
                    'key_metrics': {}
                }
                
                # Extract key metrics
                for metric in cluster_profile.index:
                    if not pd.isna(cluster_profile[metric]):
                        cluster_interpretation['key_metrics'][metric] = float(cluster_profile[metric])
                        
                # Determine dominant autonomic characteristics
                parasympathetic_score = 0
                sympathetic_score = 0
                balance_score = 0
                variability_score = 0
                
                for characteristic, indicators in autonomic_indicators.items():
                    present_indicators = [ind for ind in indicators if ind in cluster_profile.index]
                    
                    if present_indicators:
                        # Normalize scores based on relative values across all clusters
                        if len(present_indicators) > 0:
                            avg_score = cluster_profile[present_indicators].mean()
                            
                            if characteristic == 'parasympathetic_dominance':
                                parasympathetic_score = avg_score
                            elif characteristic == 'sympathetic_dominance':
                                sympathetic_score = avg_score
                            elif characteristic == 'balanced_autonomic':
                                balance_score = avg_score
                            elif characteristic == 'low_variability':
                                variability_score = -avg_score  # Invert for low variability
                                
                # Classify autonomic phenotype
                scores = {
                    'parasympathetic': parasympathetic_score,
                    'sympathetic': sympathetic_score,
                    'balanced': balance_score,
                    'low_variability': variability_score
                }
                
                dominant_pattern = max(scores.keys(), key=lambda k: scores[k])
                
                phenotype_descriptions = {
                    'parasympathetic': 'Parasympathetic-Dominant Phenotype: High vagal tone, good recovery capacity',
                    'sympathetic': 'Sympathetic-Dominant Phenotype: Elevated stress response, higher arousal',
                    'balanced': 'Balanced Autonomic Phenotype: Well-regulated autonomic function',
                    'low_variability': 'Reduced Variability Phenotype: Limited autonomic flexibility'
                }
                
                cluster_interpretation['phenotype_classification'] = dominant_pattern
                cluster_interpretation['phenotype_description'] = phenotype_descriptions.get(
                    dominant_pattern, 'Unclassified autonomic pattern'
                )
                cluster_interpretation['autonomic_scores'] = scores
                
                interpretations[f'cluster_{cluster_id}'] = cluster_interpretation
                
            # Overall cluster analysis
            interpretations['overall_analysis'] = {
                'total_clusters': cluster_result.n_clusters,
                'silhouette_score': cluster_result.silhouette_score,
                'cluster_quality': self._assess_cluster_quality(cluster_result.silhouette_score),
                'phenotype_distribution': {
                    interp['phenotype_classification']: interp['cluster_size'] 
                    for interp in interpretations.values() 
                    if isinstance(interp, dict) and 'phenotype_classification' in interp
                }
            }
            
            return interpretations
            
        except Exception as e:
            logger.error(f"Error interpreting clusters: {e}")
            return {'error': str(e)}
            
    def predict_cluster(self, new_hrv_data: pd.DataFrame) -> np.ndarray:
        """
        Predict cluster membership for new HRV data.
        
        Args:
            new_hrv_data: New HRV metrics to classify
            
        Returns:
            Predicted cluster labels
        """
        try:
            if self.cluster_model is None:
                logger.error("No clustering model available. Fit a model first.")
                return np.array([])
                
            # Prepare new data
            X_new = self._prepare_data(new_hrv_data, fit_scaler=False)
            if X_new is None:
                return np.array([])
                
            # Predict clusters
            if hasattr(self.cluster_model, 'predict'):
                predictions = self.cluster_model.predict(X_new)
            else:
                # For models without predict method, find nearest cluster center
                if self.cluster_results is None:
                    logger.error("No cluster results available")
                    return np.array([])
                    
                centers = self.cluster_results.cluster_centers
                predictions = []
                
                for point in X_new:
                    distances = [np.linalg.norm(point - center) for center in centers]
                    predictions.append(np.argmin(distances))
                    
                predictions = np.array(predictions)
                
            return predictions
            
        except Exception as e:
            logger.error(f"Error predicting clusters: {e}")
            return np.array([])
            
    def _prepare_data(self, hrv_data: pd.DataFrame, fit_scaler: bool = True) -> Optional[np.ndarray]:
        """Prepare data for clustering analysis."""
        try:
            # Select numeric columns only
            numeric_data = hrv_data.select_dtypes(include=[np.number])
            
            if numeric_data.empty:
                logger.error("No numeric variables found in HRV data")
                return None
                
            # Remove columns with no variation
            numeric_data = numeric_data.loc[:, numeric_data.std() > 0]
            
            if numeric_data.empty:
                logger.error("No variables with sufficient variation")
                return None
                
            # Handle missing values
            if numeric_data.isnull().any().any():
                logger.warning("Missing values found - using median imputation")
                numeric_data = numeric_data.fillna(numeric_data.median())
                
            # Standardize if requested
            if self.standardize:
                if fit_scaler:
                    X = self.scaler.fit_transform(numeric_data)
                else:
                    if self.scaler is None:
                        logger.error("Scaler not fitted - call with fit_scaler=True first")
                        return None
                    X = self.scaler.transform(numeric_data)
            else:
                X = numeric_data.values
                
            return X
            
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            return None
            
    def _compute_cluster_centers(self, X: np.ndarray, labels: np.ndarray, exclude_noise: bool = False) -> np.ndarray:
        """Compute cluster centers."""
        try:
            unique_labels = np.unique(labels)
            
            if exclude_noise and -1 in unique_labels:
                unique_labels = unique_labels[unique_labels != -1]
                
            centers = []
            for label in unique_labels:
                cluster_points = X[labels == label]
                if len(cluster_points) > 0:
                    center = np.mean(cluster_points, axis=0)
                    centers.append(center)
                    
            return np.array(centers)
            
        except Exception as e:
            logger.error(f"Error computing cluster centers: {e}")
            return np.array([])
            
    def _compute_cluster_profiles(self, hrv_data: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
        """Compute cluster profiles (mean values for each cluster)."""
        try:
            numeric_data = hrv_data.select_dtypes(include=[np.number])
            data_with_labels = numeric_data.copy()
            data_with_labels['cluster'] = labels
            
            # Compute mean for each cluster
            cluster_profiles = data_with_labels.groupby('cluster').mean()
            
            return cluster_profiles
            
        except Exception as e:
            logger.error(f"Error computing cluster profiles: {e}")
            return pd.DataFrame()
            
    def _determine_optimal_k(self,
                           silhouette_scores: List[float],
                           calinski_harabasz_scores: List[float], 
                           davies_bouldin_scores: List[float],
                           elbow_scores: List[float],
                           k_range: range) -> int:
        """Determine optimal number of clusters using multiple criteria."""
        try:
            if not silhouette_scores:
                return 2
                
            # Find best k for each metric
            best_silhouette_k = k_range[np.argmax(silhouette_scores)]
            best_ch_k = k_range[np.argmax(calinski_harabasz_scores)]
            best_db_k = k_range[np.argmin(davies_bouldin_scores)]
            
            # Elbow method
            if len(elbow_scores) > 2:
                # Find elbow using second derivative
                diffs = np.diff(elbow_scores)
                second_diffs = np.diff(diffs)
                if len(second_diffs) > 0:
                    elbow_k = k_range[np.argmax(second_diffs) + 2]
                else:
                    elbow_k = best_silhouette_k
            else:
                elbow_k = best_silhouette_k
                
            # Vote-based selection
            candidates = [best_silhouette_k, best_ch_k, best_db_k, elbow_k]
            
            # Count votes for each k
            from collections import Counter
            votes = Counter(candidates)
            
            # Return most voted k, or best silhouette score if tie
            optimal_k = votes.most_common(1)[0][0]
            
            return optimal_k
            
        except Exception as e:
            logger.error(f"Error determining optimal k: {e}")
            return 2
            
    def _compute_within_cluster_sum_squares(self, X: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> float:
        """Compute within-cluster sum of squares for elbow method."""
        try:
            wcss = 0
            for i, center in enumerate(centers):
                cluster_points = X[labels == i]
                if len(cluster_points) > 0:
                    wcss += np.sum((cluster_points - center) ** 2)
            return wcss
        except:
            return float('inf')
            
    def _assess_cluster_quality(self, silhouette_score: float) -> str:
        """Assess clustering quality based on silhouette score."""
        if silhouette_score >= 0.7:
            return "Excellent clustering"
        elif silhouette_score >= 0.5:
            return "Good clustering"
        elif silhouette_score >= 0.3:
            return "Fair clustering"
        else:
            return "Poor clustering"
            
    def _create_empty_cluster_result(self) -> ClusterResult:
        """Create empty cluster result for error cases."""
        return ClusterResult(
            cluster_labels=np.array([]),
            cluster_centers=np.array([]),
            n_clusters=0,
            silhouette_score=0,
            calinski_harabasz_score=0,
            davies_bouldin_score=float('inf'),
            inertia=None,
            cluster_sizes={},
            cluster_profiles=pd.DataFrame()
        ) 