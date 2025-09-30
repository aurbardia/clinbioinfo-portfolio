#!/usr/bin/env python3
"""
Pipeline Ómico - Análisis de Expresión Génica
Demo para ClinBioinfo - Bioinformática Traslacional
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def generar_datos_omicos():
    """Genera dataset sintético de expresión génica"""
    print("🧬 Generando datos ómicos sintéticos...")
    
    np.random.seed(42)
    n_genes, n_samples = 100, 60
    
    # Crear nombres de genes
    genes = [f"GENE_{i:03d}" for i in range(1, n_genes + 1)]
    
    # Generar datos con 3 grupos
    grupos = ['Control'] * 20 + ['Enfermedad_A'] * 20 + ['Enfermedad_B'] * 20
    
    expression_data = []
    for i, grupo in enumerate(grupos):
        if grupo == 'Control':
            sample_expr = np.random.normal(5, 1, n_genes)
        elif grupo == 'Enfermedad_A':
            sample_expr = np.random.normal(5, 1, n_genes)
            sample_expr[:20] += 3  # Sobreexpresar primeros 20 genes
        else:  # Enfermedad_B
            sample_expr = np.random.normal(5, 1, n_genes)
            sample_expr[20:40] += 3  # Sobreexpresar genes 20-40
        
        expression_data.append(sample_expr)
    
    # Crear DataFrame
    df = pd.DataFrame(
        np.array(expression_data).T,
        index=genes,
        columns=[f"SAMPLE_{i:03d}" for i in range(1, n_samples + 1)]
    )
    
    metadata = pd.DataFrame({
        'sample_id': df.columns,
        'grupo': grupos
    })
    
    df.to_csv('data_expression.csv')
    metadata.to_csv('metadata_samples.csv', index=False)
    
    print(f"✅ Dataset generado: {n_genes} genes x {n_samples} muestras")
    return df, metadata

def analisis_pca_clustering(df, metadata):
    """Análisis PCA y clustering"""
    print("\n🔍 Análisis PCA y Clustering...")
    
    # Normalizar datos
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df.T)
    
    # PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data_scaled)
    
    # K-means clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(data_scaled)
    
    # Visualizaciones
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # PCA por grupo real
    colors = {'Control': 'blue', 'Enfermedad_A': 'red', 'Enfermedad_B': 'green'}
    for grupo in metadata['grupo'].unique():
        mask = metadata['grupo'] == grupo
        axes[0].scatter(pca_result[mask, 0], pca_result[mask, 1], 
                       c=colors[grupo], label=grupo, alpha=0.7)
    axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    axes[0].set_title('PCA - Grupos Reales')
    axes[0].legend()
    
    # PCA por clusters
    scatter = axes[1].scatter(pca_result[:, 0], pca_result[:, 1], 
                             c=clusters, cmap='viridis', alpha=0.7)
    axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    axes[1].set_title('PCA - Clusters K-means')
    
    plt.tight_layout()
    plt.savefig('analisis_omico.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Varianza explicada PC1-PC2: {pca.explained_variance_ratio_[:2].sum():.1%}")

def main():
    """Pipeline principal"""
    print("🧬 PIPELINE ÓMICO - ClinBioinfo Demo")
    print("=" * 50)
    
    df, metadata = generar_datos_omicos()
    analisis_pca_clustering(df, metadata)
    
    print("\n✅ Pipeline ómico completado")

if __name__ == "__main__":
    main()