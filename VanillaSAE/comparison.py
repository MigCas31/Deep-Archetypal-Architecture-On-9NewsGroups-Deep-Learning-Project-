import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import NMF
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.manifold import TSNE
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy
import umap
from pathlib import Path
import sys
import pandas as pd

# Add src to path
sys.path.append('src')

from src.models.vanilla_sae import VanillaSAE
from src.data.dataset import create_dataloaders
from src.utils.config import load_config

class DeepArchetypalAnalysis:
    """Extract archetypal analysis from trained SAE"""
    
    def __init__(self, model_path, config_path, device):
        self.model = self._load_model(model_path, config_path, device)
        self.device = device
        self.hidden_representations = None
        self.reconstructions = None
        
    def _load_model(self, model_path, config_path, device):
        """Load trained SAE model"""
        config = load_config(config_path)
        
        # Get input dimension from data
        train_loader, _, _ = create_dataloaders(
            data_path=config['data']['data_path'],
            batch_size=1,
            test_size=config['data'].get('test_size', 0.2),
            val_size=config['data'].get('val_size', 0.1),
            random_state=config['data'].get('random_state', 42)
        )
        sample_batch = next(iter(train_loader))
        input_dim = sample_batch.shape[1]
        
        # Create and load model
        model = VanillaSAE(
            input_dim=input_dim,
            hidden_dim=config['model']['hidden_dim'],
            activation=config['model']['activation'],
            sparsity_penalty=config['model']['sparsity_penalty'],
            tie_weights=config['model'].get('tie_weights', False),
            dropout_rate=config['model'].get('dropout_rate', 0.0)
        )
        
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        
        return model
        
    def extract_deep_archetypes(self, X):
        print("Extracting deep archetypal representations...")
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        hidden_reps = []
        reconstructions = []
        
        with torch.no_grad():
            batch_size = 100
            for i in range(0, len(X), batch_size):
                batch = X_tensor[i:i+batch_size]
                reconstruction, hidden = self.model(batch)
                
                hidden_reps.append(hidden.cpu().numpy())
                reconstructions.append(reconstruction.cpu().numpy())
        
        self.hidden_representations = np.vstack(hidden_reps)
        self.reconstructions = np.vstack(reconstructions)
        
        print(f"Deep representations shape: {self.hidden_representations.shape}")
        return self.hidden_representations, self.reconstructions

class LinearArchetypalAnalysis:
    """Linear archetypal analysis using NMF"""
    
    def __init__(self, n_components=8):
        self.n_components = n_components
        self.nmf = None
        self.mixing_weights = None
        self.archetypes = None
        
    def fit_transform(self, X):
        """Fit linear archetypal analysis"""
        print(f"Fitting linear archetypal analysis with {self.n_components} components...")
        
        # Ensure non-negative data
        X_pos = X - X.min() + 0.001
        
        self.nmf = NMF(n_components=self.n_components, random_state=42, max_iter=300)
        self.mixing_weights = self.nmf.fit_transform(X_pos)
        self.archetypes = self.nmf.components_
        
        # Reconstruct data
        reconstructions = self.mixing_weights @ self.archetypes
        
        print(f"Linear representations shape: {self.mixing_weights.shape}")
        print(f"Linear reconstruction error: {self.nmf.reconstruction_err_:.4f}")
        
        return self.mixing_weights, reconstructions

class ArchetypalComparisonFramework:
    """Framework to compare deep vs linear archetypal analysis"""
    
    def __init__(self, deep_analyzer, linear_analyzer):
        self.deep_analyzer = deep_analyzer
        self.linear_analyzer = linear_analyzer
        self.results = {}
        
    def evaluate_reconstruction_quality(self, X_original, deep_recon, linear_recon):
        """Evaluate reconstruction quality metrics"""
        print("Evaluating reconstruction quality...")
        
        # MSE
        deep_mse = np.mean((X_original - deep_recon) ** 2)
        linear_mse = np.mean((X_original - linear_recon) ** 2)
        
        # MAE  
        deep_mae = np.mean(np.abs(X_original - deep_recon))
        linear_mae = np.mean(np.abs(X_original - linear_recon))
        
        # Correlation (sample-wise)
        deep_corrs = []
        linear_corrs = []
        
        for i in range(min(len(X_original), 1000)):  # Limit for speed
            deep_corr = np.corrcoef(X_original[i], deep_recon[i])[0, 1]
            linear_corr = np.corrcoef(X_original[i], linear_recon[i])[0, 1]
            
            if not np.isnan(deep_corr):
                deep_corrs.append(deep_corr)
            if not np.isnan(linear_corr):
                linear_corrs.append(linear_corr)
        
        deep_corr_mean = np.mean(deep_corrs)
        linear_corr_mean = np.mean(linear_corrs)
        
        return {
            'deep_mse': deep_mse,
            'linear_mse': linear_mse,
            'deep_mae': deep_mae,
            'linear_mae': linear_mae,
            'deep_correlation': deep_corr_mean,
            'linear_correlation': linear_corr_mean,
            'mse_improvement': (linear_mse - deep_mse) / linear_mse * 100,
            'correlation_improvement': (deep_corr_mean - linear_corr_mean) / linear_corr_mean * 100
        }
    
    def evaluate_sparsity_patterns(self, deep_hidden, linear_weights):
        """Evaluate sparsity characteristics"""
        print("Evaluating sparsity patterns...")
        
        # Sparsity ratios (fraction of near-zero activations)
        deep_sparsity = (np.abs(deep_hidden) < 0.01).mean()
        linear_sparsity = (np.abs(linear_weights) < 0.01).mean()
        
        # L1/L2 ratios (indicates sparsity preference)
        deep_l1_l2 = np.mean(np.sum(np.abs(deep_hidden), axis=1) / np.sqrt(np.sum(deep_hidden**2, axis=1)))
        linear_l1_l2 = np.mean(np.sum(np.abs(linear_weights), axis=1) / np.sqrt(np.sum(linear_weights**2, axis=1)))
        
        # Effective dimensionality (based on entropy)
        deep_entropy = entropy(np.mean(np.abs(deep_hidden), axis=0) + 1e-12)
        linear_entropy = entropy(np.mean(np.abs(linear_weights), axis=0) + 1e-12)
        
        # Feature usage diversity (how many features are actively used)
        deep_active_features = (np.mean(np.abs(deep_hidden), axis=0) > 0.01).sum()
        linear_active_features = (np.mean(np.abs(linear_weights), axis=0) > 0.01).sum()
        
        return {
            'deep_sparsity': deep_sparsity,
            'linear_sparsity': linear_sparsity,
            'deep_l1_l2_ratio': deep_l1_l2,
            'linear_l1_l2_ratio': linear_l1_l2,
            'deep_entropy': deep_entropy,
            'linear_entropy': linear_entropy,
            'deep_active_features': deep_active_features,
            'linear_active_features': linear_active_features,
            'total_deep_features': deep_hidden.shape[1],
            'total_linear_features': linear_weights.shape[1]
        }
    
    def evaluate_clustering_performance(self, deep_hidden, linear_weights, n_clusters=9):
        """Evaluate clustering quality of representations"""
        print("Evaluating clustering performance...")
        
        # K-means clustering on both representations
        deep_kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        linear_kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        
        deep_clusters = deep_kmeans.fit_predict(deep_hidden)
        linear_clusters = linear_kmeans.fit_predict(linear_weights)
        
        # Silhouette scores (higher is better)
        deep_silhouette = silhouette_score(deep_hidden[:5000], deep_clusters[:5000])  # Limit for speed
        linear_silhouette = silhouette_score(linear_weights[:5000], linear_clusters[:5000])
        
        # Inertia (within-cluster sum of squares, lower is better)
        deep_inertia = deep_kmeans.inertia_
        linear_inertia = linear_kmeans.inertia_
        
        # Cluster agreement (how similar are the clusterings)
        cluster_agreement = adjusted_rand_score(deep_clusters, linear_clusters)
        cluster_nmi = normalized_mutual_info_score(deep_clusters, linear_clusters)
        
        return {
            'deep_silhouette': deep_silhouette,
            'linear_silhouette': linear_silhouette,
            'deep_inertia': deep_inertia,
            'linear_inertia': linear_inertia,
            'cluster_agreement_ari': cluster_agreement,
            'cluster_agreement_nmi': cluster_nmi,
            'deep_clusters': deep_clusters,
            'linear_clusters': linear_clusters
        }
    
    def evaluate_nonlinearity_capture(self, X_original, deep_hidden, linear_weights):
        """Evaluate how well each method captures non-linear patterns"""
        print("Evaluating non-linearity capture...")
        
        # Mutual information between original features and representations
        from sklearn.feature_selection import mutual_info_regression
        
        # Sample subset for speed
        n_samples = min(2000, len(X_original))
        indices = np.random.choice(len(X_original), n_samples, replace=False)
        
        X_sample = X_original[indices]
        deep_sample = deep_hidden[indices]
        linear_sample = linear_weights[indices]
        
        # Average mutual information
        deep_mi_scores = []
        linear_mi_scores = []
        
        # For each original feature, compute MI with all representation features
        for i in range(min(100, X_sample.shape[1])):  # Limit features for speed
            if i % 20 == 0:
                print(f"Processing feature {i}/100...")
                
            original_feature = X_sample[:, i]
            
            # MI with deep representations
            deep_mi = mutual_info_regression(deep_sample, original_feature, random_state=42)
            deep_mi_scores.append(np.max(deep_mi))  # Best MI for this original feature
            
            # MI with linear representations  
            linear_mi = mutual_info_regression(linear_sample, original_feature, random_state=42)
            linear_mi_scores.append(np.max(linear_mi))
        
        deep_mi_mean = np.mean(deep_mi_scores)
        linear_mi_mean = np.mean(linear_mi_scores)
        
        # Feature interaction capture (based on pairwise correlations)
        deep_feature_interactions = np.mean(np.abs(np.corrcoef(deep_sample.T)))
        linear_feature_interactions = np.mean(np.abs(np.corrcoef(linear_sample.T)))
        
        return {
            'deep_mi_mean': deep_mi_mean,
            'linear_mi_mean': linear_mi_mean,
            'deep_feature_interactions': deep_feature_interactions,
            'linear_feature_interactions': linear_feature_interactions,
            'mi_improvement': (deep_mi_mean - linear_mi_mean) / linear_mi_mean * 100
        }
    
    def evaluate_interpretability(self, deep_hidden, linear_weights, linear_archetypes):
        """Evaluate interpretability of learned representations"""
        print("Evaluating interpretability...")
        
        # Archetypal coherence (how consistent are archetypal patterns)
        # For deep: use PCA to find interpretable directions
        from sklearn.decomposition import PCA
        
        # Deep archetypal analysis: find principal directions in hidden space
        deep_pca = PCA(n_components=min(8, deep_hidden.shape[1]))
        deep_components = deep_pca.fit_transform(deep_hidden)
        deep_explained_variance = deep_pca.explained_variance_ratio_
        
        # Linear archetypes are already interpretable
        linear_explained_variance = np.var(linear_weights, axis=0) / np.sum(np.var(linear_weights, axis=0))
        
        # Dominant pattern concentration (how concentrated are the archetypal patterns)
        deep_dominant = np.max(np.abs(deep_hidden), axis=1)
        linear_dominant = np.max(linear_weights, axis=1)
        
        deep_concentration = np.mean(deep_dominant / (np.sum(np.abs(deep_hidden), axis=1) + 1e-8))
        linear_concentration = np.mean(linear_dominant / (np.sum(linear_weights, axis=1) + 1e-8))
        
        return {
            'deep_explained_variance': deep_explained_variance[:5].tolist(),  # Top 5 components
            'linear_explained_variance': sorted(linear_explained_variance, reverse=True)[:5],
            'deep_cumulative_variance': np.cumsum(deep_explained_variance)[:5].tolist(),
            'linear_cumulative_variance': np.cumsum(sorted(linear_explained_variance, reverse=True))[:5],
            'deep_concentration': deep_concentration,
            'linear_concentration': linear_concentration,
            'deep_n_components': deep_hidden.shape[1],
            'linear_n_components': linear_weights.shape[1]
        }
    
    def run_comprehensive_comparison(self, X):
        # Extract representations
        deep_hidden, deep_recon = self.deep_analyzer.extract_deep_archetypes(X)
        linear_weights, linear_recon = self.linear_analyzer.fit_transform(X)
        
        # Run all evaluations
        self.results = {}
        
        print("\n1. Reconstruction Quality")
        self.results['reconstruction'] = self.evaluate_reconstruction_quality(X, deep_recon, linear_recon)
        
        print("\n2. Sparsity Patterns")  
        self.results['sparsity'] = self.evaluate_sparsity_patterns(deep_hidden, linear_weights)
        
        print("\n3. Clustering Performance")
        self.results['clustering'] = self.evaluate_clustering_performance(deep_hidden, linear_weights)
        
        print("\n4. Non-linearity Capture")
        self.results['nonlinearity'] = self.evaluate_nonlinearity_capture(X, deep_hidden, linear_weights)
        
        print("\n5. Interpretability")
        self.results['interpretability'] = self.evaluate_interpretability(
            deep_hidden, linear_weights, self.linear_analyzer.archetypes)
        
        return self.results, deep_hidden, linear_weights
    
    def create_comparison_visualizations(self, X, deep_hidden, linear_weights, save_dir):
        """Create comprehensive comparison visualizations"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Performance comparison radar chart
        self._create_performance_radar(save_dir)
        
        # Representation space comparison (UMAP)
        self._create_representation_comparison(deep_hidden, linear_weights, save_dir)
        
        # Reconstruction quality comparison
        self._create_reconstruction_comparison(X, save_dir)
        
        # Sparsity analysis
        self._create_sparsity_analysis(deep_hidden, linear_weights, save_dir)
        
        
        print(f"Comparison visualizations saved to: {save_dir}")
    
    def _create_performance_radar(self, save_dir):
        """Create radar chart comparing overall performance"""
        metrics = ['Reconstruction', 'Clustering', 'Sparsity', 'Non-linearity', 'Interpretability']
        
        # Normalize metrics to 0-1 scale
        deep_scores = [
            1 - self.results['reconstruction']['deep_mse'] / self.results['reconstruction']['linear_mse'],
            self.results['clustering']['deep_silhouette'] / max(self.results['clustering']['deep_silhouette'], 
                                                               self.results['clustering']['linear_silhouette']),
            1 - self.results['sparsity']['deep_sparsity'],
            self.results['nonlinearity']['deep_mi_mean'] / max(self.results['nonlinearity']['deep_mi_mean'],
                                                              self.results['nonlinearity']['linear_mi_mean']),
            self.results['interpretability']['deep_concentration']
        ]
        
        linear_scores = [
            1 - self.results['reconstruction']['linear_mse'] / self.results['reconstruction']['linear_mse'],
            self.results['clustering']['linear_silhouette'] / max(self.results['clustering']['deep_silhouette'],
                                                                 self.results['clustering']['linear_silhouette']),
            1 - self.results['sparsity']['linear_sparsity'],
            self.results['nonlinearity']['linear_mi_mean'] / max(self.results['nonlinearity']['deep_mi_mean'],
                                                                self.results['nonlinearity']['linear_mi_mean']),
            self.results['interpretability']['linear_concentration']
        ]
        
        # Radar chart
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1] 
        
        deep_scores += deep_scores[:1]
        linear_scores += linear_scores[:1]
        
        ax.plot(angles, deep_scores, 'o-', linewidth=2, label='Deep SAE', color='red')
        ax.fill(angles, deep_scores, alpha=0.25, color='red')
        
        ax.plot(angles, linear_scores, 'o-', linewidth=2, label='Linear NMF', color='blue')
        ax.fill(angles, linear_scores, alpha=0.25, color='blue')
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1)
        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        ax.set_title('Deep vs Linear Archetypal Analysis\nPerformance Comparison', size=16, pad=20)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'performance_radar_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_representation_comparison(self, deep_hidden, linear_weights, save_dir):
        """Create UMAP comparison of representation spaces"""
        print("Creating representation space comparison...")
        
        # Sample for speed
        n_samples = min(2000, len(deep_hidden))
        indices = np.random.choice(len(deep_hidden), n_samples, replace=False)
        
        deep_sample = deep_hidden[indices]
        linear_sample = linear_weights[indices]
        
        # UMAP projections
        umap_deep = umap.UMAP(n_components=2, random_state=42)
        umap_linear = umap.UMAP(n_components=2, random_state=42)
        
        deep_2d = umap_deep.fit_transform(deep_sample)
        linear_2d = umap_linear.fit_transform(linear_sample)
        
        # Plot comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        scatter1 = ax1.scatter(deep_2d[:, 0], deep_2d[:, 1], c=range(len(deep_2d)), 
                              cmap='viridis', alpha=0.6, s=10)
        ax1.set_title(f'Deep SAE Representations\n({deep_hidden.shape[1]} dimensions)')
        ax1.set_xlabel('UMAP1')
        ax1.set_ylabel('UMAP2')
        
        scatter2 = ax2.scatter(linear_2d[:, 0], linear_2d[:, 1], c=range(len(linear_2d)), 
                              cmap='viridis', alpha=0.6, s=10)
        ax2.set_title(f'Linear NMF Representations\n({linear_weights.shape[1]} components)')
        ax2.set_xlabel('UMAP1')
        ax2.set_ylabel('UMAP2')
        
        plt.tight_layout()
        plt.savefig(save_dir / 'representation_space_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_reconstruction_comparison(self, X, save_dir):
        """Create reconstruction quality comparison plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Reconstruction error comparison
        ax = axes[0, 0]
        methods = ['Deep SAE', 'Linear NMF']
        mse_values = [self.results['reconstruction']['deep_mse'], 
                     self.results['reconstruction']['linear_mse']]
        bars = ax.bar(methods, mse_values, color=['red', 'blue'], alpha=0.7)
        ax.set_title('Reconstruction MSE Comparison')
        ax.set_ylabel('MSE')
        for bar, val in zip(bars, mse_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.4f}', ha='center', va='bottom')
        
        # Correlation comparison
        ax = axes[0, 1]
        corr_values = [self.results['reconstruction']['deep_correlation'],
                      self.results['reconstruction']['linear_correlation']]
        bars = ax.bar(methods, corr_values, color=['red', 'blue'], alpha=0.7)
        ax.set_title('Reconstruction Correlation Comparison')
        ax.set_ylabel('Correlation')
        ax.set_ylim(0, 1)
        for bar, val in zip(bars, corr_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.3f}', ha='center', va='bottom')
        
        # Improvement metrics
        ax = axes[1, 0]
        improvements = ['MSE Improvement (%)', 'Correlation Improvement (%)']
        values = [self.results['reconstruction']['mse_improvement'],
                 self.results['reconstruction']['correlation_improvement']]
        colors = ['green' if v > 0 else 'red' for v in values]
        bars = ax.bar(improvements, values, color=colors, alpha=0.7)
        ax.set_title('Deep SAE Improvements over Linear')
        ax.set_ylabel('Improvement (%)')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, 
                   bar.get_height() + (0.5 if val > 0 else -0.5),
                   f'{val:.1f}%', ha='center', va='bottom' if val > 0 else 'top')
        
        # Summary metrics
        ax = axes[1, 1]
        summary_metrics = ['Reconstruction\nQuality', 'Feature\nEfficiency', 'Overall\nPerformance']
        # Calculate composite scores
        recon_score = 1 if self.results['reconstruction']['deep_correlation'] > self.results['reconstruction']['linear_correlation'] else 0
        efficiency_score = 1 if self.results['sparsity']['deep_active_features'] > self.results['sparsity']['linear_active_features'] else 0
        overall_score = (recon_score + efficiency_score) / 2
        
        scores = [recon_score, efficiency_score, overall_score]
        bars = ax.bar(summary_metrics, scores, color=['green' if s > 0.5 else 'red' for s in scores], alpha=0.7)
        ax.set_title('Deep SAE Advantage Analysis')
        ax.set_ylabel('Score (1=Deep Better, 0=Linear Better)')
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'reconstruction_quality_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_sparsity_analysis(self, deep_hidden, linear_weights, save_dir):
        """Create sparsity pattern analysis"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Sparsity distributions
        ax = axes[0, 0]
        deep_sparsity_per_sample = (np.abs(deep_hidden) < 0.01).mean(axis=1)
        linear_sparsity_per_sample = (np.abs(linear_weights) < 0.01).mean(axis=1)
        
        ax.hist(deep_sparsity_per_sample, bins=30, alpha=0.7, label='Deep SAE', color='red')
        ax.hist(linear_sparsity_per_sample, bins=30, alpha=0.7, label='Linear NMF', color='blue')
        ax.set_title('Sparsity Distribution per Sample')
        ax.set_xlabel('Sparsity Ratio')
        ax.set_ylabel('Frequency')
        ax.legend()
        
        # Feature activation patterns
        ax = axes[0, 1]
        deep_feature_activity = np.mean(np.abs(deep_hidden) > 0.01, axis=0)
        linear_feature_activity = np.mean(np.abs(linear_weights) > 0.01, axis=0)
        
        ax.plot(deep_feature_activity, label='Deep SAE', color='red', alpha=0.7)
        ax.plot(linear_feature_activity, label='Linear NMF', color='blue', alpha=0.7)
        ax.set_title('Feature Activation Frequency')
        ax.set_xlabel('Feature Index')
        ax.set_ylabel('Activation Frequency')
        ax.legend()
        
        # L1/L2 ratio comparison
        ax = axes[0, 2]
        deep_l1_l2 = np.sum(np.abs(deep_hidden), axis=1) / np.sqrt(np.sum(deep_hidden**2, axis=1))
        linear_l1_l2 = np.sum(np.abs(linear_weights), axis=1) / np.sqrt(np.sum(linear_weights**2, axis=1))
        
        ax.hist(deep_l1_l2, bins=30, alpha=0.7, label='Deep SAE', color='red')
        ax.hist(linear_l1_l2, bins=30, alpha=0.7, label='Linear NMF', color='blue')
        ax.set_title('L1/L2 Ratio Distribution')
        ax.set_xlabel('L1/L2 Ratio')
        ax.set_ylabel('Frequency')
        ax.legend()
        
        # Effective dimensionality
        ax = axes[1, 0]
        methods = ['Deep SAE', 'Linear NMF']
        entropies = [self.results['sparsity']['deep_entropy'], 
                    self.results['sparsity']['linear_entropy']]
        bars = ax.bar(methods, entropies, color=['red', 'blue'], alpha=0.7)
        ax.set_title('Effective Dimensionality (Entropy)')
        ax.set_ylabel('Entropy')
        
        # Active features comparison
        ax = axes[1, 1]
        active_features = [self.results['sparsity']['deep_active_features'],
                          self.results['sparsity']['linear_active_features']]
        total_features = [self.results['sparsity']['total_deep_features'],
                         self.results['sparsity']['total_linear_features']]
        
        bars = ax.bar(methods, active_features, color=['red', 'blue'], alpha=0.7)
        ax.set_title('Number of Active Features')
        ax.set_ylabel('Active Features')
        
        # Add total features as text
        for i, (bar, active, total) in enumerate(zip(bars, active_features, total_features)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                   f'{active}/{total}', ha='center', va='bottom')
        
        # Sparsity efficiency
        ax = axes[1, 2]
        efficiency_deep = active_features[0] / total_features[0]
        efficiency_linear = active_features[1] / total_features[1]
        
        efficiencies = [efficiency_deep, efficiency_linear]
        bars = ax.bar(methods, efficiencies, color=['red', 'blue'], alpha=0.7)
        ax.set_title('Feature Efficiency\n(Active/Total)')
        ax.set_ylabel('Efficiency Ratio')
        ax.set_ylim(0, 1)
        
        for bar, val in zip(bars, efficiencies):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(save_dir / 'sparsity_pattern_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    
    
    
    def comprehensive_report(self):
        # Save comprehensive report to file
        report_path = self.save_dir / 'comprehensive_report.txt'
        with open(report_path, 'w') as f:
            f.write("\n" + "="*80 + "\n")
            f.write("COMPREHENSIVE DEEP vs LINEAR ARCHETYPAL ANALYSIS REPORT\n")
            f.write("="*80 + "\n")
            
        
            # Reconstruction Quality
            f.write("RECONSTRUCTION QUALITY:")
            f.write(f"  Deep SAE MSE:     {self.results['reconstruction']['deep_mse']:.6f}")
            f.write(f"  Linear NMF MSE:   {self.results['reconstruction']['linear_mse']:.6f}")
            f.write(f"  MSE Improvement:  {self.results['reconstruction']['mse_improvement']:+.1f}%")
            f.write(f"  Deep Correlation: {self.results['reconstruction']['deep_correlation']:.4f}")
            f.write(f"  Linear Correlation: {self.results['reconstruction']['linear_correlation']:.4f}")

            # Sparsity Analysis
            f.write(f"SPARSITY PATTERNS:")
            f.write(f"  Deep Sparsity:    {self.results['sparsity']['deep_sparsity']:.1%}")
            f.write(f"  Linear Sparsity:  {self.results['sparsity']['linear_sparsity']:.1%}")
            f.write(f"  Deep Features:    {self.results['sparsity']['deep_active_features']}/{self.results['sparsity']['total_deep_features']}")
            f.write(f"  Linear Features:  {self.results['sparsity']['linear_active_features']}/{self.results['sparsity']['total_linear_features']}")

            # Clustering Performance
            f.write(f"CLUSTERING PERFORMANCE:")
            f.write(f"  Deep Silhouette:  {self.results['clustering']['deep_silhouette']:.4f}")
            f.write(f"  Linear Silhouette: {self.results['clustering']['linear_silhouette']:.4f}")
            f.write(f"  Cluster Agreement: {self.results['clustering']['cluster_agreement_ari']:.4f} (ARI)")

            # Non-linearity
            f.write(f"NON-LINEARITY CAPTURE:")
            f.write(f"  Deep MI Score:    {self.results['nonlinearity']['deep_mi_mean']:.4f}")
            f.write(f"  Linear MI Score:  {self.results['nonlinearity']['linear_mi_mean']:.4f}")
            f.write(f"  MI Improvement:   {self.results['nonlinearity']['mi_improvement']:+.1f}%")


def main():
    print("Deep vs Linear Archetypal Analysis Comparison Framework")
    print("="*80)
    
    # Load data
    config = load_config('configs/default.yaml')
    train_loader, val_loader, test_loader = create_dataloaders(
        data_path=config['data']['data_path'],
        batch_size=1000,
        test_size=config['data'].get('test_size', 0.2),
        val_size=config['data'].get('val_size', 0.1),
        random_state=config['data'].get('random_state', 42)
    )
    
    # Collect data
    all_data = []
    for data in [train_loader, val_loader, test_loader]:
        for batch in data:
            all_data.append(batch.numpy())
    
    X = np.vstack(all_data)
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Initialize analyzers
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test different models
    models_to_test = [
        ('results_DL/best_model.pt', 'best_model'),
        ('results_DL/final_model.pt', 'final_model')
    ]
    
    for model_path, model_name in models_to_test:
        if Path(model_path).exists():
            print(f"Analyzing {model_name}...")
            
            deep_analyzer = DeepArchetypalAnalysis(model_path, 'configs/default.yaml', device)
            linear_analyzer = LinearArchetypalAnalysis(n_components=8)
            
            # Create comparison framework
            framework = ArchetypalComparisonFramework(deep_analyzer, linear_analyzer)
            
            # Run comprehensive comparison
            results, deep_hidden, linear_weights = framework.run_comprehensive_comparison(X)
            
            # Create visualizations
            save_dir = f"deep_vs_linear_comparison_{model_name}"
            framework.create_comparison_visualizations(X, deep_hidden, linear_weights, save_dir)
            
            # Print report
            framework.print_comprehensive_report()
            
            

if __name__ == "__main__":
    main()