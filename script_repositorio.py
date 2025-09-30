from genericpath import exists
import os
import sys

def maker_estructura_repositorio():
     
    if not os.path.exists('.git'):
        print("No se detecta repositorio Git")
        print("Ejecuta el script en el repo clonado de Github")

        return False

    carpetas = [
        'project1_vigilancia_genomica',
        'project2_iRWD_SQL',
        'project3_ommics_pipeline',
        'project4_dashboard'
        'project5_reproducibilidad'
        ]
    for carpeta in carpetas:
        os.makedirs(carpeta, exist_ok=True)

    # Contenido de archivos
    archivos = {
        # README principal
        'README.md': '''# Portffolio Análisis Bioinformática

**Autor**: Aurora Barroso Díaz  
**Objetivo**: Demostrar competencias técnicas alineadas con las líneas de investigación en epigenética y bioinformática
## Sobre este portafolio

Este repositorio contiene **5 mini-proyectos** que reflejan mi experiencia en:
- **Vigilancia genómica y epidemiológica**
- **Integración de datos clínicos (iRWD)**  
- **Análisis ómico y bioinformática traslacional**
- **Dashboards interactivos para medicina de precisión**
- **Reproducibilidad y control de versiones en ciencia**

## Mi experiencia previa

- **3 años** en el Servicio de Electromedicina - Hospital Universitario Virgen del Rocío
- Coordinación de instalación de equipos médicos críticos
- Gestión integral de alertas sanitarias y documentación regulatoria
- Mantenimiento y gestión de equipos médicos
- Liderazgo de plataforma de divulgación **Electroforma**
- Automatización de procesos administrativos y ofimáticos 
rogramación: **Python, Java, SQL**

## Experiencia en Bioinformática

- Desarrollo personal de conocimientos en análisis de bioinformática y epigenética
- Experiencia académica en entornos, aplicaciones y lenguajes informáticos de interés, Python, R, Matlab.
- Experiencia en control de versiones con TortoiseSVN.

## Estructura del portafolio

### [Proyecto 1: Vigilancia Genómica](./project1_vigilancia_genomica/)
Simulación de pipeline para detección de mutaciones en secuencias virales (inspirado en SIEGA).
- **Tecnologías**: Python, Biopython, alineamiento de secuencias
- **Aplicación**: Vigilancia epidemiológica de variantes

### [Proyecto 2: iRWD - Datos Clínicos](./project2_iRWD_SQL/)
Integración y análisis de datos de mundo real (Real-World Data) con SQL.
- **Tecnologías**: Python, SQLite, pandas
- **Aplicación**: Cohortes clínicas y estudios observacionales

### [Proyecto 3: Pipeline Ómico](./project3_omics_pipeline/)
Análisis de expresión génica con clustering y reducción dimensional.
- **Tecnologías**: Python, scikit-learn, matplotlib, Jupyter
- **Aplicación**: Biomarcadores y medicina personalizada

### [Proyecto 4: Dashboard Interactivo](./project4_dashboard/)
Visualización web interactiva para datos epidemiológicos.
- **Tecnologías**: Streamlit, plotly, pandas
- **Aplicación**: Monitorización en tiempo real

### [Proyecto 5: Reproducibilidad Científica](./project5_reproducibilidad/)
Ejemplo de pipeline reproducible con control de dependencias.
- **Tecnologías**: Python, R, Docker, requirements.txt
- **Aplicación**: FAIR data y open science


## Cómo ejecutar

Cada proyecto tiene su propio README con instrucciones específicas. En general:

## Contacto

- **Email**: auorbarroso@gmail.com
- **LinkedIn**: Aurora María Barroso Díaz
- **GitHub**: http://github.com/aurbardia/

---
''',

        # Requirements.txt
        'requirements.txt': '''# Dependencias generales del portafolio ClinBioinfo
pandas==2.0.3
numpy==1.24.3
matplotlib==3.7.2
seaborn==0.12.2
scikit-learn==1.3.0
biopython==1.81
streamlit==1.25.0
plotly==5.15.0
jupyter==1.0.0
sqlite3''',

        # Proyecto 1 - README
        'project1_vigilancia_genomica/README.md': '''# Proyecto 1: Análisis Genómico

## Descripción
Simulación de pipeline para **detección de mutaciones** en secuencias virales, inspirado en el sistema **SIEGA** de ClinBioinfo para vigilancia epidemiológica.

## Objetivo
Demostrar capacidades en:
- Procesamiento de secuencias FASTA
- Alineamiento de secuencias
- Detección automática de variantes
- Generación de reportes epidemiológicos

## Tecnologías
- **Python 3.8+**
- **Biopython** (manejo de secuencias)
- **pandas** (análisis de datos)
- **matplotlib** (visualización)

## Instalación y ejecución

```bash
# Instalar dependencias
pip install biopython pandas matplotlib

# Ejecutar análisis
python script_align.py

# Salida esperada:
# - Reporte de mutaciones detectadas
# - Gráfico de distribución de variantes
# - Archivo CSV con resultados
```


Este proyecto simula el flujo de trabajo del **sistema SIEGA** para:
- Vigilancia de variantes SARS-CoV-2
- Trazabilidad de bacterias resistentes
- Análisis epidemiológico automatizado''',

        # Proyecto 1 - Script
        'project1_vigilancia_genomica/script_align.py': '''#!/usr/bin/env python3
"""
Pipeline de Vigilancia Genómica
Detección automática de mutaciones en secuencias virales
"""

from Bio import SeqIO
from Bio.Seq import Seq
import pandas as pd
import matplotlib.pyplot as plt
import os

def generar_datos_sinteticos():
    """Genera archivo FASTA sintético para demo"""
    # Secuencia de referencia (simulando fragmento viral)
    referencia = "ATGCGATCGTAGCTAGCTAGCTAGCGATCGATCGTAGCTAGCTAGCTAGCGATCGATCGTAGCTAGCTAGCTAGC"
    
    secuencias_demo = [
        ("REF_001", referencia),
        ("VAR_002", referencia.replace("ATGC", "TTGC")),  # Mutación A>T
        ("VAR_003", referencia.replace("GATC", "CATC")),  # Mutación G>C
        ("VAR_004", referencia.replace("CGTA", "CGAA")),  # Mutación T>A
        ("VAR_005", referencia.replace("GCTA", "ACTA")),  # Mutación G>A
        ("VAR_006", referencia[:30] + "AAAA" + referencia[34:]),  # Inserción
        ("VAR_007", referencia.replace("TAGC", "AAGC")),  # Mutación T>A
        ("VAR_008", referencia.replace("ATCG", "ATAG")),  # Mutación C>A
        ("VAR_009", referencia.replace("GCGC", "ACGC")),  # Mutación G>A
        ("VAR_010", referencia.replace("CTAG", "CTGG"))   # Mutación A>G
    ]
    
    with open("datos_sinteticos.fasta", "w") as f:
        for seq_id, secuencia in secuencias_demo:
            f.write(f">{seq_id}\\n{secuencia}\\n")
    
    print(" Datos sintéticos generados: datos_sinteticos.fasta")

def cargar_secuencias(archivo_fasta):
    """Carga secuencias desde archivo FASTA"""
    secuencias = []
    for record in SeqIO.parse(archivo_fasta, "fasta"):
        secuencias.append({
            'id': record.id,
            'secuencia': str(record.seq),
            'longitud': len(record.seq)
        })
    print(f" Cargadas {len(secuencias)} secuencias")
    return secuencias

def detectar_mutaciones(secuencias, referencia_idx=0):
    """Detecta mutaciones comparando con secuencia de referencia"""
    referencia = secuencias[referencia_idx]['secuencia']
    mutaciones = []
    
    for i, seq_data in enumerate(secuencias):
        if i == referencia_idx:
            continue
            
        secuencia = seq_data['secuencia']
        seq_id = seq_data['id']
        
        # Comparación posición por posición
        for pos, (ref_base, seq_base) in enumerate(zip(referencia, secuencia)):
            if ref_base != seq_base:
                mutaciones.append({
                    'secuencia_id': seq_id,
                    'posicion': pos + 1,
                    'referencia': ref_base,
                    'mutacion': seq_base,
                    'tipo': f"{ref_base}{pos+1}{seq_base}"
                })
    
    print(f" Detectadas {len(mutaciones)} mutaciones")
    return mutaciones

def generar_reporte(mutaciones):
    """Genera reporte de mutaciones en CSV y visualización"""
    if not mutaciones:
        print(" No se encontraron mutaciones")
        return
    
    # DataFrame con mutaciones
    df_mut = pd.DataFrame(mutaciones)
    
    # Guardar CSV
    df_mut.to_csv('reporte_mutaciones.csv', index=False)
    print(" Reporte guardado: reporte_mutaciones.csv")
    
    # Visualización
    plt.figure(figsize=(12, 6))
    
    # Gráfico 1: Distribución de posiciones
    plt.subplot(1, 2, 1)
    plt.hist(df_mut['posicion'], bins=20, alpha=0.7, color='skyblue')
    plt.xlabel('Posición en genoma')
    plt.ylabel('Número de mutaciones')
    plt.title('Distribución de mutaciones por posición')
    
    # Gráfico 2: Tipos de mutación más frecuentes
    plt.subplot(1, 2, 2)
    top_mutaciones = df_mut['tipo'].value_counts().head(10)
    top_mutaciones.plot(kind='bar', color='lightcoral')
    plt.xlabel('Tipo de mutación')
    plt.ylabel('Frecuencia')
    plt.title('Top 10 mutaciones detectadas')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('analisis_mutaciones.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Resumen estadístico
    print("\\n RESUMEN EPIDEMIOLÓGICO")
    print(f"Total mutaciones: {len(mutaciones)}")
    print(f"Secuencias analizadas: {df_mut['secuencia_id'].nunique()}")
    print(f"Posiciones afectadas: {df_mut['posicion'].nunique()}")

def main():
    """Pipeline principal"""
    print(" PIPELINE DE VIGILANCIA GENÓMICA - ClinBioinfo Demo")
    print("=" * 60)
    
    # Verificar archivo de datos
    archivo_fasta = "datos_sinteticos.fasta"
    if not os.path.exists(archivo_fasta):
        print(f" No se encuentra {archivo_fasta}")
        print("Generando datos sintéticos...")
        generar_datos_sinteticos()
    
    # Ejecutar pipeline
    secuencias = cargar_secuencias(archivo_fasta)
    mutaciones = detectar_mutaciones(secuencias)
    generar_reporte(mutaciones)
    
    print("\\n Pipeline completado exitosamente")
    print("Archivos generados:")
    print("  - reporte_mutaciones.csv")
    print("  - analisis_mutaciones.png")

if __name__ == "__main__":
    main()''',

        # Proyecto 2 - README
        'project2_iRWD_SQL/README.md': '''#  Proyecto 2: iRWD - Datos Clínicos con SQL

## Descripción
Simulación de **integración y análisis de datos de mundo real (Real-World Data)** 

## Tecnologías
- **Python 3.8+**
- **SQLite** (base de datos)
- **pandas** (análisis de datos)
- **matplotlib/seaborn** (visualización)

## Instalación y ejecución

```bash
# Instalar dependencias
pip install pandas matplotlib seaborn

# Crear base de datos sintética y ejecutar análisis
python consulta.py

# Salida esperada:
# - Base de datos SQLite con 500 pacientes
# - Análisis de cohortes por edad/sexo
# - Gráficos de distribuciones clínicas
```


Este proyecto simula el flujo de trabajo de **iRWD** para:
- Estudios observacionales retrospectivos
- Análisis de cohortes clínicas
- Integración de datos hospitalarios''',

        # Proyecto 2 - Script (versión simplificada)
        'project2_iRWD_SQL/consulta.py': '''#!/usr/bin/env python3
"""
iRWD - Análisis de Datos Clínicos con SQL
Demo para ClinBioinfo - Real World Data Analysis
"""

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import random
from datetime import datetime, timedelta

def crear_base_datos_sintetica():
    """Crea base de datos SQLite con datos clínicos sintéticos"""
    print("Creando base de datos clínica sintética...")
    
    # Conectar a SQLite
    conn = sqlite3.connect("clinica_synthetic.db")
    cursor = conn.cursor()
    
    # Crear tabla pacientes
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS pacientes (
        id INTEGER PRIMARY KEY,
        edad INTEGER,
        sexo TEXT,
        hospital TEXT,
        diagnostico TEXT,
        dias_hospitalizacion INTEGER
    )
    """)
    
    # Generar datos sintéticos
    hospitales = ["H. Virgen del Rocío", "H. Reina Sofía", "H. Puerta del Mar"]
    diagnosticos = ["Diabetes", "Hipertensión", "EPOC", "Cardiopatía", "Nefropatía"]
    
    pacientes_data = []
    for i in range(1, 501):  # 500 pacientes
        pacientes_data.append((
            i,
            random.randint(18, 85),
            random.choice(["M", "F"]),
            random.choice(hospitales),
            random.choice(diagnosticos),
            random.randint(1, 30)
        ))
    
    cursor.executemany("""
    INSERT OR REPLACE INTO pacientes 
    (id, edad, sexo, hospital, diagnostico, dias_hospitalizacion)
    VALUES (?, ?, ?, ?, ?, ?)
    """, pacientes_data)
    
    conn.commit()
    conn.close()
    print("Base de datos creada: clinica_synthetic.db")

def analizar_cohortes():
    """Realiza análisis de cohortes clínicas"""
    print("\\n ANÁLISIS DE COHORTES")
    print("=" * 40)
    
    conn = sqlite3.connect("clinica_synthetic.db")
    
    # Análisis 1: Distribución por edad y sexo
    query1 = """
    SELECT 
        CASE 
            WHEN edad < 30 THEN '18-29'
            WHEN edad < 50 THEN '30-49'
            WHEN edad < 70 THEN '50-69'
            ELSE '70+'
        END as grupo_edad,
        sexo,
        COUNT(*) as total_pacientes
    FROM pacientes 
    GROUP BY grupo_edad, sexo
    ORDER BY grupo_edad, sexo
    """
    
    df_demo = pd.read_sql_query(query1, conn)
    print("Distribución demográfica:")
    print(df_demo)
    
    # Análisis 2: Top diagnósticos
    query2 = """
    SELECT diagnostico, COUNT(*) as frecuencia
    FROM pacientes
    GROUP BY diagnostico
    ORDER BY frecuencia DESC
    """
    
    df_diag = pd.read_sql_query(query2, conn)
    print("\\nTop diagnósticos:")
    print(df_diag)
    
    # Crear visualizaciones
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Gráfico 1: Distribución por diagnóstico
    df_diag.plot(x='diagnostico', y='frecuencia', kind='bar', ax=axes[0])
    axes[0].set_title('Distribución por Diagnóstico')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Gráfico 2: Distribución por hospital
    query3 = "SELECT hospital, COUNT(*) as pacientes FROM pacientes GROUP BY hospital"
    df_hosp = pd.read_sql_query(query3, conn)
    df_hosp.plot(x='hospital', y='pacientes', kind='bar', ax=axes[1], color='orange')
    axes[1].set_title('Pacientes por Hospital')
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('analisis_iRWD.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    conn.close()
    print("\\n Análisis completado")

def main():
    """Función principal"""
    crear_base_datos_sintetica()
    analizar_cohortes()

if __name__ == "__main__":
    main()''',

        # Proyecto 3 - README
        'project3_omics_pipeline/README.md': '''# Proyecto 3: Pipeline Ómico - Análisis de Expresión Génica

## Descripción
Pipeline de **bioinformática traslacional** para análisis de datos de expresión génica, clustering de muestras y identificación de biomarcadores.

## Tecnologías
- **Python 3.8+**
- **pandas, scikit-learn, matplotlib, seaborn**

## Instalación y ejecución

```bash
# Instalar dependencias
pip install pandas scikit-learn matplotlib seaborn

# Ejecutar análisis
python clustering.py

# Salida esperada:
# - Dataset sintético de expresión génica
# - Análisis PCA y clustering
# - Identificación de biomarcadores
```


Pipeline típico de **bioinformática traslacional** para:
- Identificación de biomarcadores
- Clasificación molecular de enfermedades
- Análisis de cohortes ómicas''',

        # Proyecto 3 - Script (versión simplificada)
        'project3_omics_pipeline/clustering.py': '''#!/usr/bin/env python3
"""
Pipeline Ómico - Análisis de Expresión Génica
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def generar_datos_omicos():
    """Genera dataset sintético de expresión génica"""
    print("Generando datos ómicos sintéticos...")
    
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
    
    print(f" Dataset generado: {n_genes} genes x {n_samples} muestras")
    return df, metadata

def analisis_pca_clustering(df, metadata):
    """Análisis PCA y clustering"""
    print("\\n Análisis PCA y Clustering...")
    
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
    print(" PIPELINE ÓMICO - ClinBioinfo Demo")
    print("=" * 50)
    
    df, metadata = generar_datos_omicos()
    analisis_pca_clustering(df, metadata)
    
    print("\\n Pipeline ómico completado")

if __name__ == "__main__":
    main()''',

        # Proyecto 4 - README
        'project4_dashboard/README.md': '''# Proyecto 4: Dashboard Interactivo - Vigilancia Epidemiológica

## Descripción
**Dashboard web interactivo** para visualización en tiempo real de datos epidemiológicos y clínicos.

## Tecnologías
- **Streamlit** (framework web)
- **Plotly** (gráficos interactivos)
- **pandas** (manipulación de datos)

## Instalación y ejecución

```bash
# Instalar dependencias
pip install streamlit plotly pandas

# Ejecutar dashboard
streamlit run app_dashboard.py

# Abrir en navegador: http://localhost:8501
```

Dashboard similar a herramientas de:
- **Sistema SIEGA**: Vigilancia genómica
- **Plataforma iRWD**: Análisis clínicos
- **Monitorización hospitalaria**: Métricas en tiempo real''',

        # Proyecto 4 - Script (versión simplificada)
        'project4_dashboard/app_dashboard.py': '''#!/usr/bin/env python3
"""
Dashboard Interactivo - Vigilancia Epidemiológica
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta

# Configuración de página
st.set_page_config(
    page_title="ClinBioinfo Dashboard",
    page_icon="🧬",
    layout="wide"
)

@st.cache_data
def generar_datos_dashboard():
    """Genera datos sintéticos para el dashboard"""
    
    # Datos temporales (últimos 30 días)
    fechas = pd.date_range(
        start=datetime.now() - timedelta(days=30),
        end=datetime.now(),
        freq='D'
    )
    
    hospitales = ["H. Virgen del Rocío", "H. Reina Sofía", "H. Puerta del Mar"]
    
    # Generar casos diarios
    casos_data = []
    for fecha in fechas:
        for hospital in hospitales:
            casos = np.random.poisson(15)  # Media de 15 casos por día
            casos_data.append({
                'fecha': fecha,
                'hospital': hospital,
                'casos_nuevos': casos,
                'tasa_positividad': np.random.uniform(0.05, 0.20)
            })
    
    df_casos = pd.DataFrame(casos_data)
    
    # Datos de variantes
    variantes = ['Alpha', 'Delta', 'Omicron', 'BA.1', 'BA.2']
    variantes_data = []
    
    for _ in range(200):
        variantes_data.append({
            'variante': np.random.choice(variantes),
            'hospital': np.random.choice(hospitales),
            'fecha': np.random.choice(fechas[-14:])  # Últimas 2 semanas
        })
    
    df_variantes = pd.DataFrame(variantes_data)
    
    return df_casos, df_variantes

def main():
    """Función principal del dashboard"""
    
    # Título
    st.title("🧬 ClinBioinfo Dashboard")
    st.markdown("**Plataforma Andaluza de Medicina Computacional**")
    
    # Cargar datos
    df_casos, df_variantes = generar_datos_dashboard()
    
    # Métricas principales
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_casos = df_casos['casos_nuevos'].sum()
        st.metric("📊 Total Casos", f"{total_casos:,}")
    
    with col2:
        casos_ultima_semana = df_casos[df_casos['fecha'] >= datetime.now() - timedelta(days=7)]['casos_nuevos'].sum()
        st.metric("📈 Última Semana", f"{casos_ultima_semana:,}")
    
    with col3:
        tasa_promedio = df_casos['tasa_positividad'].mean()
        st.metric("🎯 Tasa Positividad", f"{tasa_promedio:.1%}")
    
    st.markdown("---")
    
    # Gráficos
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Evolución Temporal")
        casos_diarios = df_casos.groupby('fecha')['casos_nuevos'].sum().reset_index()
        fig_temporal = px.line(casos_diarios, x='fecha', y='casos_nuevos', 
                              title="Casos Nuevos por Día")
        st.plotly_chart(fig_temporal, use_container_width=True)
    
    with col2:
        st.subheader("🏥 Casos por Hospital")
        casos_hospital = df_casos.groupby('hospital')['casos_nuevos'].sum().reset_index()
        fig_hospital = px.bar(casos_hospital, x='hospital', y='casos_nuevos',
                             title="Total Casos por Hospital")
        st.plotly_chart(fig_hospital, use_container_width=True)
    
    # Análisis de variantes
    st.subheader("🦠 Distribución de Variantes")
    
    col1, col2 = st.columns(2)
    
    with col1:
        variantes_count = df_variantes['variante'].value_counts().reset_index()
        fig_variantes = px.pie(variantes_count, values='count', names='variante',
                              title="Distribución de Variantes")
        st.plotly_chart(fig_variantes, use_container_width=True)
    
    with col2:
        st.subheader("📋 Datos Recientes")
        datos_recientes = df_casos.tail(10)[['fecha', 'hospital', 'casos_nuevos', 'tasa_positividad']]
        st.dataframe(datos_recientes, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown(f"**Última actualización**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()''',

        # Proyecto 5 - README
        'project5_reproducibilidad/README.md': '''# Proyecto 5: Reproducibilidad Científica

## Descripción
Ejemplo de **pipeline científico reproducible** con control de dependencias y documentación completa.

## Tecnologías
- **Python 3.8+**
- **Control de versiones**
- **Documentación técnica**
- **Automatización con Makefile**

## Instalación y ejecución

```bash
# Ejecutar pipeline completo
make all

# O ejecutar directamente
python pipeline_reproducible.py

# Salida esperada:
# - Dataset sintético reproducible
# - Análisis estadístico
# - Visualizaciones
# - Reporte final con metadatos
```

## Principios FAIR implementados
- **Findable**: Metadatos descriptivos
- **Accessible**: Formatos estándar
- **Interoperable**: APIs documentadas  
- **Reusable**: Código modular

## Conexión con ClinBioinfo
Demuestra **best practices** para:
- Reproducibilidad en investigación biomédica
- Control de versiones en proyectos colaborativos
- FAIR data principles en bioinformática''',

        # Proyecto 5 - Script (versión simplificada)
        'project5_reproducibilidad/pipeline_reproducible.py': '''#!/usr/bin/env python3
"""
Pipeline Científico Reproducible
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
    print("Generando datos reproducibles...")
    
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
    
    print(f" Dataset generado: {n_samples} muestras, {n_features} características")
    return df

def analisis_reproducible(df):
    """Análisis estadístico reproducible"""
    print(" Ejecutando análisis reproducible...")
    
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
    
    print(" Análisis completado")
    return resultados

def generar_reporte_final(resultados):
    """Genera reporte final"""
    print(" Generando reporte final...")
    
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
    
    print(" Reporte final generado")

def main():
    """Pipeline principal"""
    print(" PIPELINE REPRODUCIBLE - ClinBioinfo Demo")
    print("=" * 50)
    
    # Ejecutar pipeline
    df = generar_datos_reproducibles()
    resultados = analisis_reproducible(df)
    generar_reporte_final(resultados)
    
    print("\\n Pipeline reproducible completado")
    print(" Ver carpeta 'output' para resultados")

if __name__ == "__main__":
    main()''',

        # Proyecto 5 - Makefile
        'project5_reproducibilidad/Makefile': '''# Makefile para Pipeline Reproducible - ClinBioinfo Demo

.PHONY: all clean data analysis report help

# Objetivo por defecto
all: data analysis report

# Ayuda
help:
	@echo "Pipeline Reproducible - ClinBioinfo"
	@echo "Comandos disponibles:"
	@echo "  make all      - Ejecutar pipeline completo"
	@echo "  make data     - Generar datos sintéticos"
	@echo "  make analysis - Ejecutar análisis"
	@echo "  make report   - Generar reporte"
	@echo "  make clean    - Limpiar archivos"

# Generar datos
data:
	@echo "Generando datos..."
	python pipeline_reproducible.py

# Ejecutar análisis
analysis: data
	@echo "Ejecutando análisis..."
	@echo "Análisis completado"

# Generar reporte
report: analysis
	@echo "Generando reporte..."
	@echo "Reporte generado"

# Limpiar archivos
clean:
	@echo "Limpiando archivos..."
	rm -rf data/ output/
	@echo "Limpieza completada"'''
    }
    #creador de archivos
    for ruta_archivo, contenido in archivos.items():
        #Crea un directorio padre si no existe
        directorio = os.path.dirname(ruta_archivo)
        if directorio:
            os.makedirs(directorio, exist_ok= True)

        #escribir los archivos
        with open(ruta_archivo, 'w', encoding = 'utf-8') as f:
            f.write(contenido)

        print(f"Archivo creado: {ruta_archivo}")
    
    print(f"Estructura completa creada")
    print(f"Total archivos: {len(archivos)}")
    print(f"Total carpetas: {len(ruta_archivo)}")

    return True
if __name__ == "__main__":
    print("Generador de portfolio")
    print("=" * 50)

    if maker_estructura_repositorio():
        print("\nPortfolio creado existosamente")
        print("\nPróximos pasos")
        print("1. Instalar dependencias: pip install -r requirements.txt")
        print("2. Probar un proyecto: cd project1_vigilancia_genomica && python script_align.py")
        print("3. Subir a GitHub: git add . && git commit -m 'Initial commit' && git push")

    else:
        print("\n Error crerado portfolio")
        sys.exit(1)
