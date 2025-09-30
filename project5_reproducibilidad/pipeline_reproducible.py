#!/usr/bin/env python3
"""
Pipeline Científico Reproducible - ClinBioinfo Demo
Ejemplo de análisis biomédico con principios FAIR
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime

def generar_datos_reproducibles():
    """Genera dataset sintético reproducible"""
    print("🔄 Generando datos reproducibles...")
    
    # Semilla fija para reproducibilidad
    np.random.seed(42)
    
    n_samples, n_features = 200, 10
    
    # Generar datos sintéticos
    X = np.random.randn(n_samples, n_features)
    y = X.sum(axis=1) + 0.1 * np.random.randn(n_samples)
    
    # Crear DataFrame
    feature_names = [f'feature_{i:02d}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    df['patient_id'] = [f'PAT_{i:04d}' for i in range(n_samples)]
    
    # Crear directorio de datos
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/dataset_reproducible.csv', index=False)
    
    # Metadatos
    metadata = {
        'filename': 'dataset_reproducible.csv',
        'n_samples': n_samples,
        'n_features': n_features,
        'generation_date': datetime.now().isoformat(),
        'random_seed': 42,
        'description': 'Dataset sintético para demostración de reproducibilidad'
    }
    
    with open('data/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Dataset generado: {n_samples} muestras, {n_features} características")
    return df

def analisis_reproducible(df):
    """Análisis estadístico reproducible"""
    print("📊 Ejecutando análisis reproducible...")
    
    # Crear directorio de salida
    os.makedirs('output', exist_ok=True)
    
    # Estadísticas descriptivas
    stats = df.describe()
    stats.to_csv('output/estadisticas.csv')
    
    # Correlaciones
    corr_matrix = df.select_dtypes(include=[np.number]).corr()
    
    # Visualizaciones
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histograma de la variable objetivo
    axes[0].hist(df['target'], bins=20, alpha=0.7, color='skyblue')
    axes[0].set_title('Distribución Variable Objetivo')
    axes[0].set_xlabel('Valor')
    axes[0].set_ylabel('Frecuencia')
    
    # Heatmap de correlaciones
    im = axes[1].imshow(corr_matrix, cmap='coolwarm', aspect='auto')
    axes[1].set_title('Matriz de Correlación')
    axes[1].set_xticks(range(len(corr_matrix.columns)))
    axes[1].set_yticks(range(len(corr_matrix.columns)))
    axes[1].set_xticklabels(corr_matrix.columns, rotation=45)
    axes[1].set_yticklabels(corr_matrix.columns)
    
    plt.tight_layout()
    plt.savefig('output/analisis_reproducible.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Resultados
    resultados = {
        'mean_target': float(df['target'].mean()),
        'std_target': float(df['target'].std()),
        'n_samples': len(df),
        'analysis_date': datetime.now().isoformat()
    }
    
    with open('output/resultados.json', 'w') as f:
        json.dump(resultados, f, indent=2)
    
    print("✅ Análisis completado")
    return resultados

def generar_reporte_final(resultados):
    """Genera reporte final"""
    print("📋 Generando reporte final...")
    
    reporte = f"""# Reporte de Análisis Reproducible

## Resumen
- **Fecha de análisis**: {resultados['analysis_date']}
- **Muestras analizadas**: {resultados['n_samples']}
- **Media objetivo**: {resultados['mean_target']:.4f}
- **Desviación estándar**: {resultados['std_target']:.4f}

## Archivos generados
- data/dataset_reproducible.csv
- data/metadata.json
- output/estadisticas.csv
- output/resultados.json
- output/analisis_reproducible.png

## Reproducibilidad
Este análisis es completamente reproducible usando:
- Semilla aleatoria fija (42)
- Versiones específicas de librerías
- Documentación completa de parámetros
"""
    
    with open('output/reporte_final.md', 'w') as f:
        f.write(reporte)
    
    print("✅ Reporte final generado")

def main():
    """Pipeline principal"""
    print("🔄 PIPELINE REPRODUCIBLE - ClinBioinfo Demo")
    print("=" * 50)
    
    # Ejecutar pipeline
    df = generar_datos_reproducibles()
    resultados = analisis_reproducible(df)
    generar_reporte_final(resultados)
    
    print("\n✅ Pipeline reproducible completado")
    print("📁 Ver carpeta 'output' para resultados")

if __name__ == "__main__":
    main()